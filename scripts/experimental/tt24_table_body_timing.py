"""Transfer when to exert effort from earlier live wrist-only table tennis."""

import argparse
from collections import deque
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, SequenceArena
from .tt24_bounce import BounceArena
from .tt24_bounce_trial import seeds, verify_replay
from .tt24_motor import MotorEffects, MotorPolicy, RewardSearch, optical_sightings
from .tt24_table_body import DIR, store, summary


DATA = DIR / 'table-body-optical-source.npz'
RESULT = DIR / 'table-body-timing.json.gz'
CONFIRM = DIR / 'table-body-timing-confirm.json.gz'
SOURCE = DIR / 'wrist-match-held.json.gz'


class Optics:
    """Use a short shared retinal history with no coordinate or body model."""

    def __init__(self):
        self.recent = deque(maxlen=4)

    def read(self, rgb):
        ball, blade = optical_sightings(rgb)
        if ball is None or blade is None:
            self.recent.clear()
            return None
        span = max(.065, blade.half_span)
        point = np.r_[(ball.center - blade.center) / span, np.log(max(.03, ball.radius / span))]
        self.recent.append(point)
        past = self.recent[0]
        diff = (point - past) / max(1, len(self.recent) - 1)
        range_ = np.linalg.norm(point[:2])
        closing = -float(point[:2] @ diff[:2]) / max(range_, .18)
        return np.r_[np.log1p(range_), point[2], closing, diff[2], np.linalg.norm(diff[:2]),
                     range_ * np.exp(-point[2])]


class Effort:
    """A smooth source fit from optical state to the magnitude of motor drive."""

    def __init__(self, features, drive, *, constant=False, permute=None):
        self.constant = constant
        if self.constant:
            return
        if permute is not None:
            drive = [np.roll(row, permute.integers(8, len(row) - 7)) for row in drive]
        x = np.vstack(features)
        y = np.concatenate(drive).astype(float)
        valid = np.isfinite(x).all(axis=1)
        self.center = np.mean(x[valid], axis=0)
        self.scale = np.maximum(np.std(x[valid], axis=0), .10)
        x = self.forms(x[valid])
        self.default = float(y[valid].mean())
        self.weight = np.linalg.solve(x.T @ x + 5.5 * np.eye(x.shape[1]), x.T @ y[valid])

    def forms(self, features):
        x = np.clip((features - self.center) / self.scale, -3., 3.)
        il, ir = np.triu_indices(x.shape[1])
        return np.c_[np.ones(len(x)), x, .45 * x[:, il] * x[:, ir]]

    def predict(self, features):
        if self.constant:
            return 1.
        if features is None:
            return .50
        raw = float((self.forms(features[None]) @ self.weight)[0])
        return float(np.clip(.50 + .50 * raw / max(self.default, .02), .35, 1.9))


def collect(path):
    with gzip.open(SOURCE, 'rt') as stream:
        source = json.load(stream)
    if source['verified_tapes'] != sum(len(row['results']) for row in source['cases']):
        raise ValueError('source games must be physically verified')
    env = SequenceArena(**dict(CONFIG, width=320, height=240))
    env.streams = ('wrist',)
    payload = {}
    try:
        for case in source['cases']:
            row = case['results']['wrist']
            env.lane = row['lane']
            env.reset(seed=row['seed'], server=-1, render=False)
            sensed = Optics()
            features = []
            commands = np.asarray(row['actions'])[:, :5]
            for near, far in zip(row['actions'], row['rival_actions'], strict=True):
                retina = sensed.read(env.render_cameras()['wrist'])
                features.append(np.full(6, np.nan) if retina is None else retina)
                opponent = np.asarray(far)
                env._aim = env._command_to = lambda *args, opponent=opponent: opponent
                _, reward, done, cap, _ = env.step(near, render=False)
                if done or cap:
                    break
            if len(features) != len(commands) or reward != row['reward'] or env.point_result() != row['outcome']:
                raise ValueError('source replay diverged in duration or reward')
            if digest_json(env.events) != digest_json(row['events']):
                raise ValueError('source replay diverged in physical contact events')
            power = np.linalg.norm(commands, axis=1)
            features = np.asarray(features)
            payload[f'x_{row["seed"]}'] = features
            payload[f'y_{row["seed"]}'] = power
            print('table timing source', row['seed'], len(power), np.isfinite(features[:, 0]).sum(), flush=True)
        np.savez_compressed(path, **payload)
    finally:
        env.close()


class TimedPolicy(MotorPolicy):
    """Apply the same joint readout with a learned visual effort multiplier."""

    def __init__(self, weights, effects, gate):
        super().__init__(weights, effects, readout='rate', retinal='relative')
        self.gate, self.scene = gate, Optics()
        self.factor = 1.

    def command(self, obs, *, plastic=False, blind=False, sampled=True):
        if sampled:
            self.factor = self.gate.predict(self.scene.read(None if blind else obs['rgb']))
        return super().command(obs, plastic=plastic, blind=blind, sampled=sampled)

    def _drive(self, displacement, velocity):
        return self.factor * super()._drive(displacement, velocity)


def game(arena, start, weights, gate, effects, *, split='practice', control=None):
    obs = arena.reset(seed=start, split=split, render=control is None)
    policy = TimedPolicy(weights, effects, gate)
    actions, factors = [], []
    done = False
    while not done:
        tick = len(actions)
        command = (np.zeros(5) if control == 'hold' else
                   policy.command(obs, sampled=tick % 2 == 0, blind=control == 'blind', plastic=split == 'practice'))
        obs, _, done = arena.step(command, render=control is None and (tick + 1) % 2 == 0)
        actions.append(command.tolist())
        factors.append(policy.factor)
    result = arena.judge_metrics()
    return result | dict(score=result['reward'], frames=len(actions), actions=actions, factors=factors)


def validate(x, y):
    fits = []
    for i, (optics, score) in enumerate(zip(x, y, strict=True)):
        source_x = [v for j, v in enumerate(x) if i != j]
        source_y = [v for j, v in enumerate(y) if i != j]
        source = Effort(source_x, source_y)
        shuffle = Effort(source_x, source_y, permute=np.random.default_rng(349905))
        good = np.isfinite(optics).all(axis=1)
        errors = []
        for head in (source, shuffle):
            pred = head.forms(optics[good]) @ head.weight
            errors.append(float(np.mean((pred - score[good])**2)) if np.any(good) else None)
        fits.append(dict(frames=int(good.sum()), trained=errors[0], shuffled=errors[1]))
    return fits


def optical_archive(data):
    with np.load(data, allow_pickle=False) as a:
        keys = [key[2:] for key in a.files if key.startswith('x_')]
        x, y = ([a[f'{prefix}_{key}'] for key in keys] for prefix in ('x', 'y'))
    gates = dict(table=Effort(x, y), shuffled=Effort(x, y, permute=np.random.default_rng(349905)),
                 cold=Effort(x, y, constant=True))
    return keys, x, y, gates


def run(data, output):
    keys, x, y, gates = optical_archive(data)
    validation = validate(x, y)
    print('table timing source validation', validation, flush=True)
    practice = seeds(261111, 2) * 10
    held = seeds(261112, 12)
    result = dict(schema='tt24-table-body-timing-1', source_tapes=keys, source_validation=validation,
                  source_frames=sum(map(len, y)), visible_source_frames=sum(int(np.isfinite(row[:, 0]).sum()) for row in x),
                  practice=practice, target_demonstrations=0, held_starts=held, training={}, held={}, summary={})
    env = BounceArena(width=96, height=72, apex_reward_weight=.24)
    try:
        for name, gate in gates.items():
            search = RewardSearch(5, 261113, radius=.42)
            effects = MotorEffects(5)
            training = []
            for i in range(0, len(practice), 2):
                weights = search.propose()
                episodes = [game(env, start, weights, gate, effects) for start in practice[i:i + 2]]
                score = float(np.mean([row['score'] + .25 * min(row['frames'] / 120, 1.) for row in episodes]))
                search.update(weights, score)
                training.append(dict(i=i // 2, evidence=[{key: row[key] for key in ('seed', 'score', 'rebounds', 'frames')}
                                                        for row in episodes], selected=float(search.elite[0][0])))
            result['training'][name] = dict(attempts=training, weights=search.best.tolist(),
                                            effects=effects.readout.tolist())
            conditions = {name: None, 'hold': 'hold', 'blind': 'blind'} if name == 'table' else {name: None}
            for condition, stimulus in conditions.items():
                rows = [game(env, start, search.best, gate, deepcopy(effects), split='held', control=stimulus)
                        for start in held]
                for row in rows:
                    verify_replay(env, row)
                result['held'][condition] = rows
                result['summary'][condition] = summary(rows)
                print('table timing held', condition, result['summary'][condition], flush=True)
            store(output, result)
    finally:
        env.close()


def confirm(data, trained, output):
    _, _, _, gates = optical_archive(data)
    with gzip.open(trained, 'rt') as stream:
        artifact = json.load(stream)
    starts = seeds(261114, 24)
    if len(set(starts + artifact['held_starts'])) != len(starts) + len(artifact['held_starts']):
        raise ValueError('confirmation and initial evaluations must not overlap')
    modes = dict(table=('table', 'table', None), shuffled=('shuffled', 'shuffled', None),
                 cold=('cold', 'cold', None), table_constant=('table', 'cold', None),
                 hold=('table', 'table', 'hold'), blind=('table', 'table', 'blind'))
    result = dict(schema='tt24-table-body-timing-confirm-1', loaded=str(trained), starts=starts,
                  extra_target_practice=0, held={}, summary={})
    env = BounceArena(width=96, height=72, apex_reward_weight=.24)
    try:
        for name, (motor, gate, control) in modes.items():
            policy = artifact['training'][motor]
            effects = MotorEffects(5)
            effects.readout[:] = policy['effects']
            rows = [game(env, seed, np.asarray(policy['weights']), gates[gate], deepcopy(effects),
                         split='held', control=control) for seed in starts]
            for row in rows:
                verify_replay(env, row)
            result['held'][name] = rows
            result['summary'][name] = summary(rows)
            print('table timing confirm', name, result['summary'][name], flush=True)
            store(output, result)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--trained', type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.data.exists():
        collect(args.data)
    if args.confirm:
        confirm(args.data, args.trained, args.output or CONFIRM)
    else:
        run(args.data, args.output or RESULT)


if __name__ == '__main__':
    main()
