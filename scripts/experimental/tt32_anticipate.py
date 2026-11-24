"""Check camera-guided preparation for the next ball after fixed post-stroke recovery."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor, tt29_vision_twin as vision
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_rally_physics import desired_shot, mujoco
from .tt29_vision_replay import clearance_final
from .tt31_approach import ApproachCheck, JOG_TICKS
from .tt31_curriculum import play_card
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt31_recovery import Recovery


SOURCE = Path('artifacts/tt31/recovery-full.json.gz')
CHECKPOINT = Path('artifacts/tt31/ready-staged-checkpoint.json.gz')
OUTPUT = Path('artifacts/tt32/anticipate-full.json.gz')
MIN_ETA = .255
MAX_ETA = .82
CONTEXT_SCALE = np.array([.12, .18, 1.2, 1.4, 1.2])


def prep_pose(planner, key, q, eta):
    """Use nearby successful approaches as seeds for a new geometric intercept pose."""
    x, z, vx, vy, vz = key
    point = np.array([x, -1.29, z])
    normal = desired_shot(point, np.array([vx, vy, vz]), 1, (0., 1.05), 5.3)
    target = point - .026 * normal
    neighbors = planner.memory.nearest(key, 5)
    seeds = [q]
    initial = []
    for example in neighbors:
        index = int(np.clip(np.rint((example.time_to_contact - .10) /
                                    planner.arena.control_dt), 0, len(example.targets) - 1))
        initial.append(example.targets[index, :5])
    if initial:
        poses = np.stack(initial)
        # Average only across close motion starts. Avoid averaging different
        # joint-space routes to the same blade geometry.
        close = np.max(np.abs(poses - poses[0]), axis=1) < .65
        difference = (np.stack([item.flight for item in neighbors]) - key) / CONTEXT_SCALE
        weights = np.exp(-np.minimum(np.sum(difference ** 2, axis=1), 12.))
        seeds.extend((poses[0], np.average(poses[close], axis=0, weights=weights[close])))
        reachable = np.argmin(np.max(np.abs(poses - q), axis=1))
        seeds.append(poses[reachable])
    candidates = []
    for start in seeds:
        joints, (position, face, *_) = planner.ik.fit(target, normal, start)
        residual = np.linalg.norm(position - target)
        travel = np.max(abs(joints - q))
        if residual < .060 and face @ normal > .80:
            cost = residual / .06 + .55 * (1. - face @ normal) + travel / (3.0 * eta)
            candidates.append((cost, joints, residual))
    if not candidates:
        return None
    _, joints, residual = min(candidates, key=lambda row: row[0])
    return joints, float(residual)


class Anticipate(vision.OpticalPlanner):
    """Use short motion prefixes while waiting for a verified tape for the next ball."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_arm_stroke = False
        self.saw_prior_approach = False
        self.next_approach = False
        self.last_capture = None
        self.preparation = None
        self.checker = None
        self.samples = []
        self.opportunities = []

    def command(self, step, flight, angles, rate, target):
        now = step * self.twin.control_dt
        fresh = (flight is not None and now - flight.time <= .052 and
                 flight.rms < 1.45 and flight.depth_sigma < .10 and flight.frames >= 3)
        if fresh and self.started_arm_stroke:
            p, v = flight.advance(now)
            # A ball already on the near half may leave the cameras briefly
            # through the stroke. A later incoming track on the far half
            # still marks a new approach even if its departure was occluded.
            if (v[1] < -1.30 and p[1] < -.55) or (v[1] > 1.20 and p[1] < 1.20):
                self.saw_prior_approach = True
                self.next_approach = False
            elif v[1] < -1.30 and p[1] > -.28 and self.saw_prior_approach:
                self.opportunities.append(step)
                self.saw_prior_approach = False
                self.next_approach = True
                active_stroke = (self.incoming and self.tape is not None and
                                 0 <= step - self.first < len(self.tape))
                if not active_stroke:
                    self.last_capture = flight.time
                    self.preparation = self.prepare(step, flight, p, v, angles, rate, target)
                    if self.preparation is not None:
                        # The old outgoing tail can end once the next approach
                        # is visible. Give the physical arm three ticks to begin
                        # moving before verifying that next stroke.
                        self.tape = None
                        self.incoming = True
                        self.bounced = False
                        self.previous_try = step - 2
                        return self.preparation[1]
        stock = super().command(step, flight, angles, rate, target)
        active_tape = self.tape is not None and step - self.first < len(self.tape)
        if self.incoming and active_tape:
            self.started_arm_stroke = True
        if not self.next_approach or not self.incoming or active_tape or np.any(stock):
            self.preparation = None
            return stock
        if fresh and flight.time != self.last_capture:
            self.last_capture = flight.time
            self.preparation = self.prepare(step, flight, p, v, angles, rate, target)
        if self.preparation is not None and step < self.preparation[0]:
            return self.preparation[1]
        return stock

    def prepare(self, step, flight, position, velocity, angles, rate, target):
        if velocity[1] > -1.3 or position[1] < -1.02:
            return None
        forecast = self.nominal.flight.path(position, velocity, np.zeros(3), 1,
                                            flight.bounced and position[1] < .1, horizon=.94)
        packet = motor.context(forecast, -1.29)
        if packet is None:
            return None
        key, eta = packet
        if not MIN_ETA <= eta <= MAX_ETA or abs(key[0]) > .40 or not .87 < key[1] < 1.25:
            return None
        result = prep_pose(self.nominal, key, target[:5], eta)
        if result is None:
            return None
        goal, error = result
        delta = goal - target[:5]
        increment = self.twin.control_dt * self.nominal.rates[:5]
        requested = delta / (increment * max(JOG_TICKS, .17 / self.twin.control_dt))
        requested *= min(1., .35 / max(float(np.max(abs(requested))), 1e-8))
        if np.max(abs(requested)) < .025:
            return None
        action = np.r_[requested, 0.]
        bounced = self.bounced
        self._refresh(step, flight, angles, rate, target)
        self.bounced = bounced
        if self.checker is None:
            self.checker = ApproachCheck(self.twin)
        gap = self.checker.clear(action)
        self.samples.append(dict(step=step, capture_s=flight.time, position=position.tolist(),
                                 intercept=key.tolist(), eta_s=eta, fit_m=error,
                                 target=goal.tolist(), action=action.tolist(), nominal_gap_m=gap))
        return None if gap is None else (step + JOG_TICKS, action)


class DelayOnly(Anticipate):
    """Retain the same optical gate and scheduling with stationary preparation."""

    def prepare(self, *args):
        checked = super().prepare(*args)
        if checked is None:
            return None
        until, motion = checked
        hold = np.zeros_like(motion)
        self.samples[-1]['suppressed_action'] = motion.tolist()
        self.samples[-1]['action'] = hold.tolist()
        return until, hold


def replay(row, card):
    arena = replay_arena([card])
    cad = replay_arena([card], detail='cad')
    guard = ArmClearance(cad.model, tolerance=0.)
    shadow = mujoco.MjData(cad.model)
    try:
        arena.launch(row['shot'])
        arena.partner.reset(seed=card['seed'])
        arena.sample_clearance(True)
        arena._cad_reader = guard, shadow
        for command in row['actions']:
            arena.exchange(command, np.zeros(5))
        events = digest_json(arena.events)
        placements = digest_json(arena.point_placements())
        if events != digest_json(row['events']) or placements != digest_json(row['placements']):
            raise ValueError('Physical replay changed events or placements')
        cad_result = clearance_final(arena, guard, shadow)
        if cad_result['samples'] != round(arena.data.time / arena.model.opt.timestep) + 1:
            raise ValueError('CAD replay missed a physics state')
        return dict(events_sha256=events, placements_sha256=placements,
                    clearance=cad_result, motor=arena.motor_audit())
    finally:
        arena.close()
        cad.close()


def audit(reference, card, row, ready, *, delay_only=False):
    prefix = next((i for i, (old, new) in enumerate(zip(reference['actions'], row['actions']))
                   if old != new), None)
    wait = JOG_TICKS if delay_only else 0
    if prefix is not None and not any(0 <= prefix - sample['step'] <= wait and
                                      sample['nominal_gap_m'] is not None for sample in row['preparation']):
        raise ValueError('First action difference was not a checked preparation command')
    if prefix is None and (reference['events'] != row['events'] or
                           len(reference['actions']) != len(row['actions'])):
        raise ValueError('A matching issued tape changed the point')
    reply = None
    if prefix is not None:
        contacts = [e for e in reference['events'] if e['kind'] == 'contact' and e['step'] <= prefix]
        if len(contacts) < 3 or contacts[-1]['side'] != -1 or not any(e['side'] == 1 for e in contacts):
            raise ValueError('Preparation preceded the far player’s reply to an arm strike')
        reply = contacts[-1]['time']
    with centered(ready):
        replays = {name: replay(case, card) for name, case in (('fixed', reference), ('adaptive', row))}
    return dict(first_different_tick=prefix, preceding_far_contact_s=reply, replays=replays)


def run(feeds, source, checkpoint, output, *, delay_control=None):
    fixed = _read(source)
    digest = _hash(checkpoint)
    if fixed['checkpoint_sha256'] != digest or len(fixed['cards']) != len(fixed['rows']):
        raise ValueError('Expected a completed fixed recovery report and its original checkpoint')
    ready = fixed['ready']
    cases = list(zip(fixed['cards'], fixed['rows'], strict=True))
    if delay_control is not None:
        cases = [case for case in cases if case[0]['id'] == delay_control]
        if len(cases) != 1:
            raise ValueError('Select one measured opening for the delay-only control')
    # Inspect games where preparation can engage first. The policy and
    # checkpoint reset at each point.
    cases.sort(key=lambda case: not case[1]['score']['learner'])
    report = dict(schema='tt32-camera-anticipation-1', fixed_sha256=_hash(source),
                  movement_sha256=digest, optics_sha256=digest, ready=ready,
                  feeds_sha256=_hash(feeds), opening_side=1, cards=[case[0] for case in cases], rows=[])
    policies = []
    policy_class = Anticipate
    if delay_control is not None:
        policy_class = DelayOnly
        report['delay_only'] = True

    class Recorded(policy_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            policies.append(self)

    with (centered(ready), patch.object(vision, 'OpticalPlanner', Recorded),
          patch.object(motor, 'RallyObjective', Recovery), TemporaryDirectory() as scratch):
        for card, baseline in cases:
            row, _ = play_card(feeds, card, checkpoint, checkpoint, Path(scratch) / 'point.json.gz',
                               practice=False, cap=24 if fixed['phase'] == 'periodic' else 16)
            observer = policies.pop()
            row['preparation'] = observer.samples
            row['observed_next_approaches'] = observer.opportunities
            row['audit'] = audit(baseline, card, row, ready, delay_only=delay_control is not None)
            report['rows'].append(row)
            _write(output, report)
            print(card['id'], baseline['score']['learner'], row['score']['learner'],
                  len(row['preparation']), row['audit']['first_different_tick'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--delay-control', type=int, help='Measure one opening while holding during preparation')
    args = parser.parse_args()
    output = args.output or (OUTPUT if args.delay_control is None else
                             OUTPUT.with_name('anticipate-delay-control.json.gz'))
    run(args.feeds, args.source, args.checkpoint, output, delay_control=args.delay_control)
