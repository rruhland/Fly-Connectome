"""Build contrastive motor practice from taped donor rallies, then play new points."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory import RATE
from .tt19_flight import FlightLaw
from .tt20_natural import NaturalViewTransport
from .tt21_coach_oracle import PHASE
from .tt22_branch_motor import BranchMotor, LATERAL_FRACTION, MOTORS, MotorCue, OutcomeScorer
from .tt22_match import step_recorded
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_learn import JitteredOpening, _replay
from .tt22_rally_trial import CONFIG, OUTPUT, alternating

ARCHIVES = (('paired-course.json.gz', 'frozen'), ('tt22_opponent-learned.json.gz', 'fixed'))
PONG = OUTPUT/'duration-pong-player.json'
POINT_PONG = OUTPUT.parent/'tt3'/'discovered-numeric-source.json'
HELD = 1913093713
SHOTS_PER_DONOR = 5
BRANCH_HORIZON = 39
JOINT_RATE = RATE[:5]
ABLATION = {'neutral': (0., 0.), 'geometric': (1., 1.)}


class BranchPractice:
    def __init__(self, coefficients):
        self.arena = JitteredOpening(**CONFIG)
        self.snapshot = mujoco.MjData(self.arena.model)
        self.cue = MotorCue(coefficients)
        self.action = np.zeros(5)
        self.arena._aim = lambda *args: self.action

    def step(self, command, rival):
        self.action = np.asarray(rival)
        self.arena._rival_attempt = True
        _, reward, done, timeout, _ = self.arena.step(command, render=False)
        return reward, done or timeout

    def branches(self, tick, proposal, row):
        arena = self.arena
        mujoco.mj_copyData(self.snapshot, arena.model, arena.data)
        context = {name: deepcopy(getattr(arena, name)) for name in PHASE if hasattr(arena, name)}
        before = tuple(arena.events)
        controls = np.asarray(row['rival_actions'][tick:tick+BRANCH_HORIZON])
        before_legal, before_contact = arena.legal_returns, arena.contacts
        features, indices, actions, sizes, _, _ = proposal
        results, frame_count = [], 0
        for motor, tape, length in zip(indices, actions, sizes, strict=True):
            self.restore(context, before)
            target = self.cue.target.copy()
            for i in range(BRANCH_HORIZON):
                rival = controls[i] if i < len(controls) else np.zeros(5)
                if i < length:
                    command = np.r_[tape[i], 0.]
                    target += tape[i]*JOINT_RATE
                else:
                    drive = np.clip((self.cue.neutral-target)/JOINT_RATE, -1., 1.)
                    command = np.r_[drive, 0.]
                    target += drive*JOINT_RATE
                _, end = self.step(command, rival)
                frame_count += 1
                if arena.legal_returns > before_legal or end:
                    break
            landing = next((list(event['location'][:2]) for event in arena.events[len(before):]
                            if event['kind'] == 'legal_return' and event['side'] == 1), None)
            results.append((int(motor), bool(arena.legal_returns > before_legal),
                            bool(arena.contacts > before_contact), landing))
        self.restore(context, before)
        return dict(seed=row['seed'], tick=tick, leg=self.cue.previous, features=features.tolist(),
                    outcomes=results, physical_frames=frame_count)

    def restore(self, context, events):
        mujoco.mj_copyData(self.arena.data, self.arena.model, self.snapshot)
        for key, value in context.items():
            setattr(self.arena, key, deepcopy(value))
        self.arena.events = list(events)

    def donor(self, row, *, student=False):
        self.cue.reset()
        arena = self.arena
        arena.reset(seed=row['seed'], server=-1, render=False)
        observations = dict(row['optical'])
        decisions = {shot['tick']: shot['leg'] for shot in row['shots']} if student else None
        corpus = []
        for tick, (learner, rival) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
            estimate = self.cue.observe(tick, observations.get(tick))
            if student and tick in decisions:
                if estimate is None or decisions[tick] != self.cue.previous:
                    raise ValueError(f"student optical cue diverged at {row['seed']}:{tick}")
                if self.cue.previous > 0 and len(corpus) < SHOTS_PER_DONOR and arena._last_hitter == -1:
                    proposal = self.cue.proposals(*estimate)
                    if proposal is None:
                        raise ValueError(f"student motor menu changed at {row['seed']}:{tick}")
                    corpus.append(self.branches(tick, proposal, row))
                self.cue.consume()
            if not student and estimate is not None and len(corpus) < SHOTS_PER_DONOR and arena._last_hitter == -1:
                proposal = self.cue.proposals(*estimate)
                if proposal is not None:
                    corpus.append(self.branches(tick, proposal, row))
                    self.cue.consume()
            reward, ended = self.step(learner, rival)
            self.cue.advance(learner)
            if ended:
                break
        if (tick+1 != len(row['actions']) or arena.point_result() != row['outcome'] or reward != row['reward']
                or json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True)):
            raise ValueError(f"donor physics replay changed for {row['seed']}")
        return corpus

    def close(self):
        self.arena.close()


def train(output, limit):
    started = perf_counter()
    archives = [(OUTPUT/name, group) for name, group in ARCHIVES]
    source = _read(PONG)['coefficients'][0][1:]
    saved = dict(schema='tt22-branch-practice-1', source=source, source_hash=_hash(PONG),
                 archive={path.name: {'sha256': _hash(path), 'group': group} for path, group in archives},
                 motors=MOTORS, contexts=[], donors=[], profile=_read(archives[0][0])['profile'])
    practice = BranchPractice(source)
    try:
        donors = [(path.name, row) for path, group in archives for row in _read(path)['rows'][group]]
        for archive, row in donors[:limit]:
            examples = practice.donor(row)
            saved['donors'].append([archive, row['seed']])
            saved['contexts'].extend(examples)
            _write(output, saved)
            print(row['seed'], [(e['leg'], len(e['outcomes']), sum(t[1] for t in e['outcomes']))
                                for e in examples], round(perf_counter()-started, 1), flush=True)
    finally:
        practice.close()
    saved['seconds'] = perf_counter()-started
    _write(output, saved)


class NativePoint:
    def __init__(self, weights):
        self.arena = JitteredOpening(**CONFIG)
        self.rival = install_rally_opponent(self.arena, target_profile=weights['profile'])
        self.optics = NaturalViewTransport('fixed', lens_fov=self.arena.lens_intrinsics(),
                                           law=FlightLaw(), flight=False, associate=True)

    def play(self, seed, learner):
        self.arena.reset(seed=seed, server=-1, render=False)
        learner.reset()
        tracker = self.optics.tracker()
        actions, rival_actions, optical = [], [], []
        for tick in range(self.arena.max_steps):
            tracker.update(self.arena.render_cameras(), tick=tick)
            ball = tracker.ball
            position = None
            if ball is not None and ball.observed and tracker.sources.get('ball_range') == 'stereo':
                position = np.asarray(ball.position)
                optical.append((tick, position.tolist()))
            command = learner.command(tick, position)
            actions.append(command.tolist())
            reward, done, capped, rival = step_recorded(self.arena, command)
            rival_actions.append(rival)
            if done or capped:
                break
        events = list(self.arena.events)
        return dict(seed=seed, role=-1, reward=reward, actions=actions, rival_actions=rival_actions,
                    optical=optical, events=events, longest=alternating(events), shots=learner.shots,
                    outcome=self.arena.point_result(), rgb=tracker.digest.hexdigest())

    def close(self):
        self.arena.close()


def held(checkpoint, output, count, schedule, conditions, lateral_fraction=LATERAL_FRACTION):
    started = perf_counter()
    weights = _read(checkpoint)
    if weights['motors'] != [list(motor) for motor in MOTORS]:
        raise ValueError('practice motor library and live library disagree')
    seeds = np.random.default_rng(schedule).integers(2**31, size=count).tolist()
    if set(seeds) & {row[1] for row in weights['donors']}:
        raise ValueError('held seeds intersect branch practice')
    coefficients = dict(ABLATION, displacement=_read(POINT_PONG)['coefficients'][0][1:])
    coefficients = {name: coefficients.get(name, weights['source']) for name in conditions}
    experiment = dict(schema='tt22-branch-held-1', weights=checkpoint.name, weights_hash=_hash(checkpoint),
                      seeds=seeds, schedule=schedule, rows={name: [] for name in conditions}, coefficients=coefficients,
                      profile=weights['profile'], lateral_fraction=lateral_fraction)
    if output.exists():
        import json
        prior = _read(output)
        if any(prior[key] != json.loads(json.dumps(experiment[key])) for key in experiment if key != "rows"):
            raise ValueError("cannot resume different treatments or source")
        experiment = prior
    policies = {name: BranchMotor(coefficients[name],
                                 OutcomeScorer(weights['contexts'], contextual=name != 'marginal'), lateral_fraction)
                for name in conditions}
    point = NativePoint(weights)
    try:
        for index, seed in enumerate(seeds):
            names = list(policies)
            for name in names if index % 2 == 0 else reversed(names):
                if index < len(experiment["rows"][name]):
                    if experiment["rows"][name][index]["seed"] != seed:
                        raise ValueError("invalid prior seed")
                    continue
                row = point.play(seed, policies[name])
                experiment['rows'][name].append(row)
                _write(output, experiment)
                print(name, seed, row['longest'], row['outcome']['legal_returns'], row['outcome']['reason'],
                      round(perf_counter()-started, 1), flush=True)
    finally:
        point.close()
    experiment['verified'] = sum(_replay(rows) for rows in experiment['rows'].values())
    experiment['seconds'] = perf_counter()-started
    _write(output, experiment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=OUTPUT/'branch-motor-practice.json.gz')
    parser.add_argument('--output', type=Path, default=OUTPUT/'branch-motor-held.json.gz')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--schedule', type=int, default=HELD)
    parser.add_argument('--conditions', choices=('contextual', 'marginal', *ABLATION, 'displacement'), nargs='+',
                        default=('contextual', 'marginal'))
    parser.add_argument('--lateral-fraction', type=float, default=LATERAL_FRACTION)
    args = parser.parse_args()
    if not 0. <= args.lateral_fraction <= 1.:
        parser.error('lateral fraction must be between zero and one')
    if args.train:
        train(args.checkpoint, args.limit)
    else:
        held(args.checkpoint, args.output, args.count, args.schedule, args.conditions, args.lateral_fraction)


if __name__ == '__main__':
    main()
