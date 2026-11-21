"""Select the visual meeting delay through ordinary camera-observed exchanges."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from tt3_arena import TT3Arena, heldout_cases
from tt3_numeric_transfer import PortableMemory
from tt3_recovery import RecoveryForecast, RecoveryPlayer
from tt3_recovery_trial import aggregate, play
from tt3_visual import RoleModel


DELAYS = (-3.8, 0., 3.8)
CONDITIONS = ('learned', 'neutral', 'forced_deep')


class ContinuationForecast(RecoveryForecast):
    """Witness another exchange from sustained direction changes near and far from the actor."""
    def __init__(self, owner, evidence, *, require_rebound=True):
        super().__init__(owner, evidence, require_rebound=require_rebound)
        self.track_id = None
        self.last_frame = None
        self.phase = 0
        self.pending = 0
        self.streak = 0
        self.departed = False
        self.continuations = 0
        self.departures = 0

    def update(self, track, frame):
        estimate = super().update(track, frame)
        if track is None or not track.observed or self.owner.normal is None or not self.flight_sign:
            return estimate
        if self.track_id != track.id or self.last_frame != frame - 1:
            self.streak = 0
        self.track_id, self.last_frame = track.id, frame
        normal, ready = self.owner.normal, self.owner.ready
        direction = float(np.asarray(track.velocity) @ normal) * self.flight_sign
        if abs(direction) < .003:
            return estimate
        sign = int(np.sign(direction))
        if sign != self.pending:
            self.pending, self.streak = sign, 1
        else:
            self.streak += 1
        if self.streak < 3 or sign == self.phase:
            return estimate
        distance = abs(float((np.asarray(track.position) - ready) @ normal))
        if self.phase == 1 and sign == -1 and self.owner.recovery_frames and distance < .19:
            self.departures += 1
            self.departed = True
        elif self.phase == -1 and sign == 1 and self.departed:
            self.continuations += distance > .18
            self.departed = False
        self.phase = sign
        return estimate


class DelayMemory:
    """Estimate continuation and public point score for the timing hypotheses independently."""
    def __init__(self, delays=DELAYS):
        self.delays = tuple(map(float, delays))
        self.uses = np.zeros(len(delays), dtype=int)
        self.continuation = np.zeros(len(delays), dtype=int)
        self.public_score = np.zeros(len(delays), dtype=float)

    def scores(self):
        # Shared zero prior tempers hypotheses with very few real opportunities.
        return (self.continuation + .20 * self.public_score) / (self.uses + 2)

    def choose(self, rng):
        if self.uses.min() < 2:
            return int(rng.choice(np.flatnonzero(self.uses == self.uses.min())))
        scores = self.scores() + .23 * np.sqrt(np.log(self.uses.sum() + 2) / (self.uses + 1))
        return int(rng.choice(np.flatnonzero(np.isclose(scores, scores.max()))))

    def observe(self, option, continuation, public_score):
        self.uses[option] += 1
        self.continuation[option] += int(bool(continuation))
        self.public_score[option] += public_score

    def best(self):
        scores = self.scores()
        return min(np.flatnonzero(np.isclose(scores, scores.max())),
                   key=lambda index: (abs(self.delays[index]), index))

    def state(self):
        return dict(delays=list(self.delays), uses=self.uses.tolist(),
                    continuations=self.continuation.tolist(), public_scores=self.public_score.tolist(),
                    estimates=self.scores().tolist(), selected=self.delays[self.best()])


class TimingPlayer(RecoveryPlayer):
    def __init__(self, *args, timing_memory=None, timing_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_memory = timing_memory
        self.timing_rng = np.random.default_rng(timing_seed)
        self.delay_option = None

    def reset(self, option=None):
        if self.timing_memory is not None:
            self.delay_option = self.timing_memory.choose(self.timing_rng)
            self.defer = self.timing_memory.delays[self.delay_option]
        super().reset(option)
        self.flight = ContinuationForecast(self, self.evidence, require_rebound=self.require_rebound)

    def finish(self):
        result = super().finish()
        continuations = int(self.flight.continuations)
        used = bool(self.recovery_frames)
        if used and self.timing_memory is not None:
            self.timing_memory.observe(self.delay_option, continuations, self.reward)
        return dict(**result, visual_continuations=continuations, visual_departures=self.flight.departures,
                    delay_choice=self.defer, delay_exposed=used)


def _course(seed, points, role_state, numeric, *, delay=0., timing_memory=None,
            require_rebound=True):
    schedule = heldout_cases(seed, points)
    player = TimingPlayer(RoleModel(**role_state), PortableMemory.restore(numeric), seed=seed,
                          defer=delay, timing_memory=timing_memory, timing_seed=seed + 50593,
                          recall=True, forecast=True, learn_roles=True, require_rebound=require_rebound)
    rows = []
    for case in schedule:
        arena = TT3Arena(case.seed, view_seed=schedule[0].view_seed, variant=case.variant)
        row = play(arena, player)
        row.update(case=asdict(case), view_seed=schedule[0].view_seed, course_seed=seed)
        rows.append(row)
    return rows


def _summarize(rows):
    diagnostics = [row for row in rows if row['delay_exposed']]
    return dict(**aggregate(rows), exposed_points=len(diagnostics),
                visual_continuations=sum(row['visual_continuations'] > 0 for row in diagnostics),
                matched_continuations=sum(row['visual_continuations'] > 0 and row['rival_returns'] >= 2
                                          for row in diagnostics),
                evaluator_continuations=sum(row['rival_returns'] >= 2 for row in diagnostics))


def run(args):
    roles = json.loads(args.roles.read_text())['role_model']
    numeric = json.loads(args.numeric.read_text())
    memory = DelayMemory()
    training = []
    require_rebound = not args.unrestricted
    for index in range(args.practice_courses):
        rows = _course(args.practice_seed + index, args.points, roles, numeric, timing_memory=memory,
                       require_rebound=require_rebound)
        training.append(dict(summary=_summarize(rows), memory=memory.state(), points=rows))
        print('practice', index, memory.state(), flush=True)
    selected = memory.delays[memory.best()]
    trials = {}
    for name, delay in zip(CONDITIONS, (selected, 0., 3.8), strict=True):
        rows = [_course(args.test_seed + index, args.points, roles, numeric, delay=delay,
                        require_rebound=require_rebound)
                for index in range(args.test_courses)]
        summary = _summarize([row for course in rows for row in course])
        trials[name] = dict(delay=delay, summary=summary, courses=rows)
        print(name, delay, summary, flush=True)
    result = dict(practice_seed=args.practice_seed, test_seed=args.test_seed, points=args.points,
                  practice_courses=args.practice_courses, test_courses=args.test_courses,
                  require_rebound=require_rebound,
                  timing_memory=memory.state(), training=training, trials=trials, roles=roles, numeric=numeric,
                  timing_objective={'visual_continuation': 1., 'ordinary_point_score': .20})
    with gzip.open(args.output, 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--roles', type=Path, default=Path('artifacts/tt3/pong-source.json'))
    parser.add_argument('--numeric', type=Path, default=Path('/workspace/m1b-experiments/tt3-numeric-final/numeric-state.json'))
    parser.add_argument('--practice-seed', type=int, default=42865072)
    parser.add_argument('--test-seed', type=int, default=50706370)
    parser.add_argument('--practice-courses', type=int, default=6)
    parser.add_argument('--test-courses', type=int, default=3)
    parser.add_argument('--points', type=int, default=16)
    parser.add_argument('--unrestricted', action='store_true',
                        help='forecast reflections without masking any pre-rebound meeting times')
    args = parser.parse_args()
    if min(args.practice_courses, args.test_courses) < 1 or args.points < 14:
        parser.error('the timing comparison requires separate practice and test courses of 14 or more points')
    run(args)
