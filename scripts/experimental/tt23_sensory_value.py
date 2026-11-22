"""Read remaining-duration value from actual short visual transitions in both games."""

import gzip
import json
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_model_rollout import ORIGIN
from .tt20_curriculum import model_arena
from .tt21_body_state import CommandTwin


OUTPUT = ROOT/'artifacts/tt23'
NATIVE = OUTPUT.with_name('tt22')/'value-source.json.gz'
RECEIPT = OUTPUT/'sensory-value.json.gz'
SHUFFLE = 1985657604
OFFSETS = np.array((-6, -3, 0))
LANDMARKS = (24, 30)
OPTIONS = ('left', 'center', 'right')
MODES = ('sensory', 'shuffled', 'clock')


def features(actor, mover, width):
    """Pairwise geometry of measured motion and relative position, in any dimension."""
    vectors = np.concatenate((mover - actor, np.diff(mover, axis=0), np.diff(actor, axis=0))) / width
    lengths = np.linalg.norm(vectors, axis=1)
    direction = vectors / np.maximum(lengths[:, None], .003)
    angles = (direction @ direction.T)[np.triu_indices(len(vectors), 1)]
    return np.r_[lengths, angles]


def design_clock(ticks):
    age = np.asarray(ticks) / 300.
    knots = np.array((.2, .4, .6, .8))
    return np.c_[np.ones(len(age)), age, np.maximum(age[:, None] - knots, 0.)]


def ridge(design, target, weights, penalty):
    return np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                           design.T @ (weights * target))


def matrix(rows):
    return np.array([row['features'] for row in rows])


def weights(rows):
    seeds = Counter(row['seed'] for row in rows)
    return np.array([1. / (len(seeds) * seeds[row['seed']]) for row in rows])


def source_windows(episodes):
    rows, possible = [], 0
    for episode in episodes:
        trace = np.array([[value if value is not None else (np.nan, np.nan) for value in pair]
                          for pair in episode['trace']])
        last = min(len(trace) - 2, (episode['ticks'] - 1) // 4)
        for tick in range(6, last + 1, 4):
            possible += 1
            window = trace[tick + OFFSETS]
            if not np.isfinite(window).all():
                continue
            remaining = (episode['ticks'] - 4 * tick) / 120.
            rows.append(dict(seed=episode['case']['serve'], option=episode['policy'], tick=tick,
                             remaining=remaining, features=features(*window.transpose(1, 0, 2), 1.).tolist()))
    return rows, possible


class SensorValue:
    def __init__(self, rows):
        x = matrix(rows)
        w = weights(rows)
        self.center = np.average(x, weights=w, axis=0)
        self.scale = np.maximum(np.sqrt(np.average(np.square(x - self.center), weights=w, axis=0)), .03)
        elapsed = np.array([row['tick'] for row in rows])
        clock = design_clock(elapsed)
        y = -np.expm1(-np.array([row['remaining'] for row in rows]) / 2.)
        self.clock = ridge(clock, y, w, 1e-5 * np.eye(clock.shape[1]))
        residual = y - clock @ self.clock
        permutation = np.arange(len(rows))
        rng = np.random.default_rng(SHUFFLE)
        age_bin = elapsed // 15
        for age in np.unique(age_bin):
            group = np.flatnonzero(age_bin == age)
            permutation[group] = rng.permutation(group)
        basis = self.basis(x)
        penalty = .04 * np.eye(basis.shape[1])
        self.coef = dict(sensory=ridge(basis, residual, w, penalty),
                         shuffled=ridge(basis, residual[permutation], w, penalty))
        self.reference = (x - self.center) / self.scale
        distance = cdist(self.reference, self.reference, metric='sqeuclidean') / x.shape[1]
        seeds = np.array([row['seed'] for row in rows])
        distance[seeds[:, None] == seeds] = np.inf
        self.support95 = float(np.sqrt(np.quantile(distance.min(axis=1), .95)))

    def basis(self, x):
        z = np.clip((x - self.center) / self.scale, -4., 4.)
        return np.c_[z, np.square(z)]

    def predict(self, rows):
        x = matrix(rows)
        basis = self.basis(x)
        clock = design_clock([row['tick'] for row in rows]) @ self.clock
        supported = cdist((x - self.center) / self.scale, self.reference, metric='sqeuclidean').min(axis=1)
        values = dict(clock=clock, **{name: clock + basis @ coefficient for name, coefficient in self.coef.items()})
        for index, row in enumerate(rows):
            row['value'] = {name: float(scores[index]) for name, scores in values.items()}
            row['support_distance'] = float(np.sqrt(supported[index] / x.shape[1]))

    def state(self):
        return dict(clock=self.clock.tolist(), center=self.center.tolist(), scale=self.scale.tolist(),
                    coefficient={key: value.tolist() for key, value in self.coef.items()}, support95=self.support95)


def compare(groups, label):
    """Average pair order per original seed, retaining only outcome-discordant pairs."""
    results = defaultdict(list)
    cases = set()
    for group in groups:
        for left, right in combinations(group, 2):
            actual = left[label] - right[label]
            if not actual:
                continue
            seed = left['seed']
            cases.add(seed)
            for mode in MODES:
                predicted = left['value'][mode] - right['value'][mode]
                concordant = .5 if abs(predicted) < 1e-10 else float(predicted * actual > 0.)
                results[mode, seed].append(concordant)
    return dict(seeds=len(cases), pairs=sum(len(results['clock', seed]) for seed in cases),
                concordance={name: float(np.mean([np.mean(results[name, seed]) for seed in cases]))
                             if cases else None for name in MODES})


def source_score(held, model):
    model.predict(held)
    observed = -np.expm1(-np.array([row['remaining'] for row in held]) / 2.)
    groups = defaultdict(list)
    for row in held:
        groups[row['seed'], row['tick']].append(row)
    return dict(rows=len(held), supported=sum(row['support_distance'] <= model.support95 for row in held),
                mean_seed_mae={mode: float(weights(held) @ abs(observed - [row['value'][mode] for row in held]))
                               for mode in MODES}, paired=compare(groups.values(), 'remaining'))


def arm_windows(cases, twin):
    landmarks = {tick: [] for tick in LANDMARKS}
    samples = {int(frame) for tick in LANDMARKS for frame in tick + OFFSETS}
    coverage = {tick: Counter(all_visible=0) for tick in LANDMARKS}
    for case in cases:
        identities, rows = set(), {tick: [] for tick in LANDMARKS}
        for option in OPTIONS:
            point = case['results'][option]
            action = np.asarray(point['actions'], dtype='<f8')
            far = np.asarray(point['rival_actions'], dtype='<f8')
            identity = sha256(action.tobytes() + far.tobytes()).hexdigest()
            if identity in identities:
                continue
            identities.add(identity)
            ball = dict(point['optical'])
            paddle = {}
            twin.reset()
            for tick, command in enumerate(action[:max(LANDMARKS) + 1]):
                if tick in samples:
                    mujoco.mj_kinematics(twin.model, twin.data)
                    paddle[tick] = twin.data.geom('learner_racket').xpos.copy() - ORIGIN
                twin.advance(command)
            for tick in LANDMARKS:
                clock = tick + OFFSETS
                for frame in clock:
                    coverage[tick][f'sight_{frame}'] += frame in paddle and frame in ball
                if not all(frame in paddle and frame in ball for frame in clock):
                    continue
                actor = np.array([paddle[frame] for frame in clock])
                mover = np.array([ball[frame] for frame in clock])
                x = features(actor, mover, .848)
                rows[tick].append(dict(seed=case['seed'], stream=case['stream'], option=option, tick=tick,
                                       tape=identity, features=x.tolist(), learner_returns=point['outcome']['legal_returns'],
                                       longest=point['longest'], ten=int(point['longest'] >= 10)))
        for tick in LANDMARKS:
            coverage[tick]['verified_starts'] += 1
            coverage[tick]['distinct_options'] += len(identities)
            coverage[tick]['visible_options'] += len(rows[tick])
            if len(rows[tick]) == len(identities):
                coverage[tick]['all_visible'] += 1
                landmarks[tick].append(rows[tick])
    return landmarks, coverage


def archive():
    cases, manifest = [], {}
    for path in sorted(OUTPUT.glob('jog-bank-*.json.gz')):
        content = path.read_bytes()
        source = json.loads(gzip.decompress(content))
        stream = source['metadata']['stream']
        verified = [case for case in source['cases'] if case.get('verified') == 3]
        if len(verified) * 3 != source['verified'] or source['metadata']['handover'] != LANDMARKS[0] + 1:
            raise ValueError(f'archive verification or handover changed for {path.name}')
        manifest[path.name] = dict(sha256=sha256(content).hexdigest(), verified_starts=len(verified),
                                   complete=source['complete'])
        cases.extend(dict(case, stream=stream) for case in verified)
    return cases, manifest


def run():
    source = _read(NATIVE)
    train, train_possible = source_windows(source['train'])
    held, held_possible = source_windows(source['held'])
    if {row['seed'] for row in train} & {row['seed'] for row in held}:
        raise ValueError('native fit and held launch seeds must be disjoint')
    model = SensorValue(train)
    source_metrics = source_score(held, model)
    source_metrics.update(train_windows=len(train), train_possible=train_possible, held_possible=held_possible)
    cases, manifest = archive()
    nominal = model_arena()
    try:
        windows, coverage = arm_windows(cases, CommandTwin(nominal))
    finally:
        nominal.close()
    target_metrics = {}
    for tick, groups in windows.items():
        rows = [row for group in groups for row in group]
        if rows:
            model.predict(rows)
        target_metrics[tick] = dict(coverage[tick], supported=sum(row['support_distance'] <= model.support95 for row in rows),
                                    tested_windows=len(rows), **{key: compare(groups, key)
                                    for key in ('learner_returns', 'longest', 'ten')})
    mae = source_metrics['mean_seed_mae']
    source_pair = source_metrics['paired']['concordance']
    target_pair = target_metrics[LANDMARKS[0]]['learner_returns']
    checks = dict(source_mae=mae['sensory'] < min(mae['shuffled'], mae['clock']),
                  source_pair=source_pair['sensory'] >= .55 and source_pair['sensory'] >= source_pair['shuffled'] + .05,
                  complete_target=len(manifest) == 4 and all(row['complete'] for row in manifest.values()),
                  target_coverage=target_pair['seeds'] >= 10,
                  target_pair=(target_pair['seeds'] > 0 and target_pair['concordance']['sensory'] >= .65 and
                               target_pair['concordance']['sensory'] >= target_pair['concordance']['shuffled'] + .10))
    result = dict(schema='tt23-sensory-value-1', source_sha256=_hash(NATIVE), manifest=manifest, model=model.state(),
                  source=source_metrics, arm=target_metrics, gates=checks, qualifies=all(checks.values()),
                  train=train, held=held, arm_examples=windows)
    _write(RECEIPT, result)
    print('source', source_metrics, flush=True)
    print('arm', target_metrics, flush=True)
    print('gates', checks, flush=True)


if __name__ == '__main__':
    run()
