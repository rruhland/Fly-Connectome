"""Seed only the 3D world policy's tangential position block from autonomous RGB Pong."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

from .tt3_pong import CAMERA_SECONDS, NATIVE_HZ, _native_rallies, cases_from_seeds, summarize_pong
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_source import CHECKPOINT as NATIVE_WEIGHTS, VisualConsequence
from .tt24_source_native import CAP, HELD as NATIVE_QUALIFICATION, OPPONENT_SPEED, Collector, batches
from .tt24_world_prior import LIMITS, WEIGHTS
from .tt24_world_prior_train import save


OUTPUT = NATIVE_WEIGHTS.parent
RAW = OUTPUT / 'pong-world-native.json.gz'
CUT = OUTPUT / 'pong-world-initial.json'
SCHEDULE = {'practice': 893119545, 'held': 1804618423}
SIZE = {'practice': 112, 'held': 52}
SHUFFLE_SEED = 414080275
HORIZON = (4, 8)
WORLD_RADIUS = .076
WORLD_NORMAL_SPEED = 2.1


class Autonomous(Collector):
    """The qualified native policy remains responsible for every command."""

    def __init__(self, seed, model, plant, horizon):
        super().__init__(seed, model, plant, horizon)
        self.bound = None

    def choose(self):
        if self.bound is None and self.motor.effects is not None:
            self.bound = len(self.actions)
        return super().choose()

    def traces(self):
        if self.bound is None:
            return []
        blade, mover = self.path()
        axis = self.motor.axis
        across = np.array((-axis[1], axis[0]))
        scale = self.scene.extended[self.motor.identity][1]
        if not np.isfinite(blade[0]).all():
            return []
        times = CAMERA_SECONDS * (np.arange(5) - 2.)
        labels = []
        for tick in range(max(6, self.bound), len(blade) - HORIZON[1], 2):
            before, located = mover[tick - 4:tick + 1], blade[tick]
            future = blade[tick + HORIZON[0]:tick + HORIZON[1]]
            if not np.isfinite(np.r_[before.ravel(), located]).all():
                continue
            valid = np.isfinite(future).all(axis=1)
            if valid.sum() < 2:
                continue
            offset = before[-1] - located
            depth = offset @ across
            if not (.42 < abs(depth) / scale < 4.5 and np.linalg.norm(offset) < 5.2 * scale):
                continue
            velocity = times @ before / (times @ times)
            displacement = (located - blade[0]) @ axis
            later = (future[valid].mean(axis=0) - blade[0]) @ axis
            labels.append(dict(tick=tick, tangent_gap=float(offset @ axis), distance=float(abs(depth)),
                               tangent_speed=float(velocity @ axis), closing_speed=float(np.sign(depth) * velocity @ across),
                               current=float(displacement), later=float(later), apparent_span=float(scale),
                               signed_alias=self.motor.effects.index(self.actions[tick]) - 1))
        return labels


def collect():
    started = perf_counter()
    fitted, source_receipt = _read(NATIVE_WEIGHTS), _read(NATIVE_QUALIFICATION)
    if source_receipt['source_sha256'] != _hash(NATIVE_WEIGHTS):
        raise ValueError('need the exact native weights qualified for active Pong play')
    mode = VisualConsequence.load('pong')
    cases = {name: cases_from_seeds(np.random.default_rng(SCHEDULE[name]).choice(2147483646, n, replace=False) + 1)
             for name, n in SIZE.items()}
    known = {row['case']['serve'] for runs in source_receipt['rows'].values() for row in runs}
    known.update(row['serve'] for rows in _read(OUTPUT / 'source-practice.json.gz')['roster'].values() for row in rows)
    launches = [case.serve for group in cases.values() for case in group]
    if len(launches) != len(set(launches)) or set(launches) & known:
        raise ValueError('the new demonstrations overlap already recorded native launches')
    metadata = dict(schema='tt24-pong-world-native-1', native_policy_sha256=_hash(NATIVE_WEIGHTS),
                    native_qualification_sha256=_hash(NATIVE_QUALIFICATION), schedule=SCHEDULE,
                    cases={name: [asdict(case) for case in rows] for name, rows in cases.items()},
                    camera_seconds=CAMERA_SECONDS, horizon_camera_frames=list(HORIZON))
    data = _read(RAW) if RAW.exists() else dict(metadata=metadata, rows={})
    if data['metadata'] != metadata:
        raise ValueError('cached Pong trajectories have a different source or schedule')
    for name, group in cases.items():
        if name in data['rows']:
            continue
        rows = []
        for group in batches(group, 38):
            agents = [Autonomous(case.exploration, mode, fitted['plant'], source_receipt['horizon']) for case in group]
            measured = _native_rallies(group, agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
            for game, agent in zip(measured, agents, strict=True):
                signs = agent.motor.effects
                rows.append(dict(game, visual_effects=None if signs is None else list(signs),
                                 relation_executed=bool(agent.forecasts), signs=agent.decision_counts.tolist(),
                                 track_sha256=agent.scene.signature.hexdigest(), cues=agent.traces()))
        data['rows'][name] = dict(episodes=rows, native=summarize_pong(rows),
                                  observations=sum(len(row['cues']) for row in rows))
        data['collection_seconds'] = perf_counter() - started
        _write(RAW, data)
        print('autonomous Pong', name, len(rows), data['rows'][name]['observations'], flush=True)
    return data


def flatten(rows):
    names = ('tangent_gap', 'distance', 'tangent_speed', 'closing_speed', 'current', 'later', 'apparent_span')
    values = [(episode['case']['serve'], *[entry[k] for k in names])
              for episode in rows['episodes'] for entry in episode['cues']]
    array = np.array(values)
    return array[:, 0].astype(np.int64), array[:, 1:]


def aligned(data, speed):
    tangent, depth, side_speed, normal_speed, current, later, span = data.T
    positions = WORLD_RADIUS / span
    velocities = WORLD_NORMAL_SPEED / speed
    gap = np.clip(tangent * positions / .074, -1., 1.)
    lateral = np.clip(side_speed * velocities / .42, -1., 1.)
    residual = np.clip(current * positions / .071, -1., 1.)
    cue = np.column_stack((np.ones(len(data)), np.clip(depth * positions / .19, 0., 1.),
                           np.clip(normal_speed * velocities / 2.1, -1., 1.), gap**2))
    vector = np.column_stack((gap, lateral, residual))
    raw = (vector[..., None] * cue[:, None, :]).reshape(len(data), -1)
    # Bound each observed native target in the receiving head's available range.
    scaled = LIMITS[0] * .72 * np.tanh(later / (1.30 * span))
    now = LIMITS[0] * .72 * np.tanh(current / (1.30 * span))
    return raw, scaled, now


def nonlinear(raw):
    return LIMITS[0] * raw / (1. + np.abs(raw))


def invert(y):
    return y / np.maximum(.001, LIMITS[0] - np.abs(y))


def fit(data, *, raw_path=RAW, prefix='pong-world'):
    _, train = flatten(data['rows']['practice'])
    held_episodes, held = flatten(data['rows']['held'])
    speed = float(np.median(np.abs(train[:, 3])))
    x, goal, now = aligned(train, speed)
    q, reference, present = aligned(held, speed)
    movement = goal - now
    # Within visual-motion strata, retain the attainable change and current
    # actuator location while destroying the observed command/outcome relation.
    clip = .025
    shuffled = goal.copy()
    bins = np.column_stack((np.sign(train[:, 3]), np.round(now / .012), np.minimum(3, (train[:, 1] / train[:, 6]).astype(int))))
    rng = np.random.default_rng(SHUFFLE_SEED)
    for bucket in np.unique(bins, axis=0):
        idx = np.flatnonzero((bins == bucket).all(axis=1))
        shuffled[idx] = np.clip(now[idx] + rng.permutation(movement[idx]), -.72 * LIMITS[0], .72 * LIMITS[0])
    # Weight visible current-to-future motion above near-stationary frames.
    importance = 1. + 2. * np.minimum(1., np.abs(movement) / clip)
    design = x.T @ (x * importance[:, None]) + 2.1 * np.eye(x.shape[1])
    true = np.linalg.solve(design, x.T @ (invert(goal) * importance))
    sham = np.linalg.solve(design, x.T @ (invert(shuffled) * importance))
    sham *= np.linalg.norm(true) / max(1.e-8, np.linalg.norm(sham))
    cold_matched = rng.normal(size=true.shape)
    cold_matched *= np.linalg.norm(true) / np.linalg.norm(cold_matched)
    applied = {'pong': true, 'shuffled': sham, 'matched_cold': cold_matched, 'cold': np.zeros_like(true)}
    states, audit = {}, {}
    dynamic = np.abs(reference - present) > .003
    for name, coef in applied.items():
        state = np.zeros(WEIGHTS)
        state[10:22] = coef
        states[name] = state.tolist()
        predicted = nonlinear(q @ coef)
        audit[name] = dict(rmse=float(np.mean((predicted - reference)**2)**.5),
                           rms_moving=float(np.mean((predicted[dynamic] - reference[dynamic])**2)**.5),
                           correction_sign=float(np.mean(np.sign(predicted[dynamic]-present[dynamic])
                                                         == np.sign(reference[dynamic] - present[dynamic]))))
        save(OUTPUT / f'{prefix}-initial-{name}.npz', state)
    audit['persistence'] = dict(rmse=float(np.mean((present - reference)**2)**.5),
                                rms_moving=float(np.mean((present[dynamic] - reference[dynamic])**2)**.5))
    return dict(schema='tt24-pong-world-pose-init-1', native_sha256=_hash(raw_path), initial=states,
                held=audit, held_launches=int(len(np.unique(held_episodes))), held_moving_cues=int(dynamic.sum()),
                speed_mapping_m_per_s=speed, horizontal_block_start=10, initialized_coefficients=12,
                native_target_conversion=dict(radius_m=WORLD_RADIUS, time_normal_mps=WORLD_NORMAL_SPEED,
                                              target_limit_share=.72, target_span_scale=1.30), shuffle=SHUFFLE_SEED)


def main():
    torch.set_num_threads(1)
    result = fit(collect())
    _write(CUT, result)
    print('held native optical pose', result['held'], flush=True)


if __name__ == '__main__':
    main()
