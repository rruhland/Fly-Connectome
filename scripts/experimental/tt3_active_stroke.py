"""Select a continuation stroke from point reward while keeping the visual meeting plane."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tt3_arena import TT3Arena, VARIANTS, heldout_cases
from tt3_learning import play_point
from tt3_numeric_transfer import PortableMemory, VisualPlayer
from tt3_visual import RoleModel


@dataclass(frozen=True)
class Pulse:
    fraction: float
    window: float = 12.
    lead: float = .9
    require_surface: bool = False

    def displacement(self, speed, remaining):
        """Servo preview for a path with zero position and nonzero terminal velocity."""
        phase = float(np.clip(1. - remaining / self.window, 0., 1.15))
        speed = self.fraction * speed
        position = speed * self.window * phase**2 * (phase - 1.)
        velocity = speed * phase * (3. * phase - 2.)
        return position + velocity / .22


class Recurrence:
    """Recognize a new approach after visually observed motion away and back."""

    def __init__(self):
        self.sign = None
        self.previous_gap = None
        self.near = False
        self.outbound = self.far = self.returning = False
        self.repeat_at = None

    def update(self, gap, rate, frame):
        if self.sign is None:
            if abs(gap) > .045 and abs(rate) > .006 and gap * rate < 0.:
                self.sign = np.copysign(1., gap)
            else:
                return False
        separation, velocity = gap * self.sign, rate * self.sign
        if 0. < separation < .085 and velocity < -.006:
            self.near = True
        if self.near and 0. <= separation < .13 and velocity > .006:
            self.outbound = True
        if self.outbound and separation > .15:
            self.far = True
        if (self.far and self.previous_gap is not None and separation > .09
                and velocity < -.006 and .001 < self.previous_gap - separation < .068):
            if self.repeat_at is None:
                self.repeat_at = frame
            self.returning = True
        self.previous_gap = separation
        return self.returning and separation > -.025 and velocity < -.006


class ActiveStrokeMemory:
    """Apply one numeric stroke while receiving feedback in the shared memory."""

    def __init__(self, carried, pulse):
        self.carried = carried
        self.pulse = pulse
        self.reset()

    def reset(self):
        self.recurrence = Recurrence()
        self.active = []
        self.surface_witness = None
        self.frame = -1

    def choose(self, rng, explore=0.):
        return self.carried.choose(rng, explore)

    def observe(self, option, reward):
        self.carried.observe(option, reward)

    def state(self):
        return dict(carried=self.carried.state(), pulse=asdict(self.pulse))

    def _surface_before_meeting(self, normal, remaining):
        """Check the currently measured or predicted transverse reversal."""
        flight = self.owner.flight
        evidence = getattr(flight, 'evidence', None)
        if evidence is None or evidence.gravity is None or len(flight.tail) < 5:
            return False
        transverse = evidence.gravity - (evidence.gravity @ normal) * normal
        axis = int(np.argmax(np.abs(transverse)))
        accel = transverse[axis]
        if abs(accel) < .00015:
            return False
        direction = np.copysign(1., accel)
        boundary = evidence.boundary(axis, direction)
        if boundary is None:
            return False
        surface, _ = boundary
        path = np.asarray(flight.tail)[:, axis]
        before = (path[-4] - path[-7]) / 3 if len(path) >= 7 else path[-3] - path[-4]
        current = (path[-1] - path[-3]) / 2
        if (before * direction > .003 and current * direction < -.003
                and np.min(np.abs(path[-6:] - surface)) < .018):
            self.surface_witness = self.frame
        if self.surface_witness is not None and self.frame - self.surface_witness < 16:
            return True
        speed = current + abs(accel) * direction
        discriminant = speed**2 + 2. * accel * (surface - path[-1])
        if discriminant <= 0.:
            return False
        when = (direction * np.sqrt(discriminant) - speed) / accel
        return 0. < when < remaining

    def goal(self, option, actor_position, mover_position, mover_velocity, ready_position,
             *, normal=None, **kwargs):
        goal = self.carried.goal(option, actor_position, mover_position, mover_velocity,
                                 ready_position, normal=normal, **kwargs)
        if normal is None or mover_position is None or mover_velocity is None:
            return goal
        separation = float((mover_position - ready_position) @ normal)
        speed = float(mover_velocity @ normal)
        repeat = self.recurrence.update(separation, speed, self.frame)
        if not repeat or not self.pulse.fraction or abs(speed) < .006:
            return goal
        remaining = -separation / speed - self.pulse.lead
        if remaining <= -1.3 or remaining >= self.pulse.window:
            return goal
        if self.pulse.require_surface and not self._surface_before_meeting(normal, remaining + self.pulse.lead):
            return goal
        displacement = self.pulse.displacement(speed, remaining)
        # Keep the command in the locally observed workspace while leaving the
        # perpendicular displacement chosen by the carried relation intact.
        offset = float(np.clip(displacement, -.072, .072))
        self.active.append((self.frame, speed, remaining, offset))
        return goal + offset * normal


class MomentumPlayer(VisualPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory.owner = self

    def reset(self, option=None):
        self.memory.reset()
        super().reset(option)

    def act(self, frames):
        self.memory.frame = self.frame
        return super().act(frames)

    def finish(self):
        result = super().finish()
        result.update(visual_recurrence=self.memory.recurrence.repeat_at,
                      pulse_frames=self.memory.active.copy())
        return result


def run_course(cases, roles, source, pulse, seed, *, save_tape=False):
    memory = ActiveStrokeMemory(PortableMemory.restore(source), pulse)
    agent = MomentumPlayer(roles, memory, seed=seed, forecast=True, recall=True, learn_roles=True)
    view = cases[0].view_seed
    rows = []
    replay = None
    start = perf_counter()
    for number, case in enumerate(cases, 1):
        arena = TT3Arena(case.seed, view_seed=view, variant=case.variant,
                         distractor=case.distractor)
        actions = []

        def capture(_frames, action, _reward):
            actions.append(action.tolist())

        row = play_point(arena, agent, capture=capture if save_tape and replay is None else None)
        case_info = {**asdict(case), 'view_seed': view}
        row.update(case=case_info, point=number)
        rows.append(row)
        if actions and row['legal_returns'] >= 2 and row['rival_returns'] > 0 and row['pulse_frames']:
            replay = dict(**case_info, extra_resets=1, actions=actions, expected=arena.point_result(),
                          events=row['events'], action_sha256=row['action_sha256'], point=number)
    return dict(points=rows, memory=memory.state(), roles=asdict(agent.binder.model),
                wall_seconds=perf_counter() - start, replay=replay)


def summary(courses, *, start=0, stop=None):
    rows = [row for course in courses for row in course['points'][start:stop]]
    return dict(points=len(rows), official_score=sum(row['public_reward'] for row in rows),
                wins=sum(row['points_for'] for row in rows),
                opening_landings=sum(row['legal_returns'] > 0 for row in rows),
                later_landings=sum(max(0, row['legal_returns'] - 1) for row in rows),
                opponent_returns=sum(row['rival_returns'] for row in rows),
                repeat_points=sum(row['legal_returns'] > 1 for row in rows),
                repeat_wins=sum(row['legal_returns'] > 1 and row['points_for'] for row in rows),
                optical_recurs=sum(row['visual_recurrence'] is not None for row in rows),
                pulse_frames=sum(len(row['pulse_frames']) for row in rows),
                cameras=sum(row['frames'] for row in rows), probes=sum(row['probes'] for row in rows))


def run_pool(candidates, matches, points, seed, variant, roles, source, *, save_tape=False):
    results = []
    for pulse in candidates:
        courses = []
        for index in range(matches):
            cases = heldout_cases(seed + index, points, variant=variant)
            courses.append(run_course(cases, roles, source, pulse, seed + index, save_tape=save_tape))
        entry = dict(pulse=asdict(pulse), overall=summary(courses), first_eight=summary(courses, stop=8),
                     wall_seconds=sum(course['wall_seconds'] for course in courses), courses=courses)
        results.append(entry)
        print({key: entry[key] for key in ('pulse', 'overall', 'wall_seconds')}, flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles', type=Path)
    parser.add_argument('numeric', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--practice-seed', type=int, default=786225)
    parser.add_argument('--qualification-seed', type=int, default=589381042)
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--heldout-matches', type=int, default=4)
    parser.add_argument('--heldout-points', type=int, default=16)
    parser.add_argument('--variant', choices=VARIANTS, default='standard')
    parser.add_argument('--fractions', nargs='+', type=float, default=[0., .12, .23, .34, -.12])
    parser.add_argument('--practice-only', action='store_true')
    pars = parser.parse_args(argv)
    raw = json.loads(pars.roles.read_text())
    roles = RoleModel(**raw.get('role_model', raw))
    source = json.loads(pars.numeric.read_text())
    candidates = tuple(Pulse(fraction) for fraction in pars.fractions)
    if candidates[0].fraction != 0.:
        parser.error('start the search with the unchanged zero pulse')

    practice = run_pool(candidates, pars.matches, pars.points, pars.practice_seed,
                        pars.variant, roles, source)
    selected = int(np.argmax([row['overall']['official_score'] for row in practice]))
    qualification = [] if pars.practice_only else run_pool(
        (candidates[0], candidates[selected]), pars.heldout_matches, pars.heldout_points,
        pars.qualification_seed, pars.variant, roles, source, save_tape=True)
    receipt = dict(practice_seed=pars.practice_seed, qualification_seed=pars.qualification_seed,
                   variant=pars.variant, source=source, roles=asdict(roles), selected=selected,
                   practice=practice, qualification=qualification)
    pars.output.parent.mkdir(parents=True, exist_ok=True)
    with pars.output.open('wb') as file:
        with gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as output:
            output.write((json.dumps(receipt, separators=(',', ':')) + '\n').encode())


if __name__ == '__main__':
    main()
