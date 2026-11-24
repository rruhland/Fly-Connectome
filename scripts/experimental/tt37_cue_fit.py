"""Fit visual forecasts without backpropagation and score disjoint opponent seeds."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write

SCALE = np.array([.16, .075, .48, .48, .44])
MODES = ('uniform', 'ball', 'motion', 'cue')
STRENGTH = (.015, .06, .24, .96, 3.84)


def vector(row, mode):
    if mode == 'uniform':
        return []
    if mode == 'ball':
        return row['ball']
    n = 6 if mode == 'motion' else 12
    return np.r_[row['ball'], row['paddle'][:n]]


def matrix(rows, mode):
    x = np.asarray([vector(r, mode) for r in rows])
    y = np.array([r['arriving'] for r in rows])
    groups = np.array([r['release'] for r in rows])
    if not len(rows):
        raise ValueError('No paired ball-and-paddle observations')
    return x, y, groups


def weights(groups):
    _, where, count = np.unique(groups, return_inverse=True, return_counts=True)
    inv = 1. / count[where]
    return inv / inv.sum()


def solve(x, y, group, penalty):
    w = weights(group)
    mean = w @ x
    spread = np.sqrt(w @ np.square(x - mean)).clip(1e-6)
    ys = w @ y
    s = (x - mean) / spread
    prior = penalty * np.eye(x.shape[1])
    beta = np.linalg.solve(s.T @ (w[:, None] * s) + prior,
                           s.T @ (w[:, None] * (y - ys)))
    return dict(mean=mean.tolist(), spread=spread.tolist(), ys=ys.tolist(), beta=beta.tolist(),
                penalty=penalty)


def predict(model, value):
    mean, spread, ys, beta = (np.asarray(model[k]) for k in ('mean', 'spread', 'ys', 'beta'))
    if not len(mean):
        return np.broadcast_to(ys, np.shape(value)[:-1] + ys.shape)
    return ((value - mean) / spread).clip(-4., 4.) @ beta + ys


def choose(x, y, group, mode):
    if mode == 'uniform':
        return solve(x, y, group, 0.), None
    ids = np.random.default_rng(377901).permutation(np.unique(group))
    if len(ids) < 4:
        raise ValueError('Too few independent practice releases for regularization selection')
    costs = defaultdict(list)
    for fold in np.array_split(ids, 4):
        test = np.isin(group, fold)
        w = weights(group[test])
        for strength in STRENGTH:
            model = solve(x[~test], y[~test], group[~test], strength)
            error = (predict(model, x[test]) - y[test]) / SCALE
            # X and Vx are slightly more useful than Z and Vy to early near-arm preparation.
            loss = np.square(error) @ np.array([2., 1., 1.5, 1., 1.])
            costs[strength].append(float(w @ loss))
    selected = min(costs, key=lambda p: (np.mean(costs[p]), -p))
    return solve(x, y, group, selected), float(np.mean(costs[selected]))


def group_rows(docs, phase, later=None):
    rows = []
    seen = set()
    for d in docs:
        for p in d['rows']:
            current = p['causal'] if phase == 'causal' else p['bins'][phase]
            for row in current:
                if later is not None and later != (row['leg'] > 0):
                    continue
                card = p['card']
                identity = (card['id'], card['seed'], card['reply_seed'], row['leg'])
                if identity in seen:
                    raise ValueError('Forecast sample counted twice')
                seen.add(identity)
                if (row['contact_s'] not in p['far_contacts_s']
                        or not row['tick_s'] < row['contact_s'] < row['label_capture_s']):
                    raise ValueError('Forecast marker crosses or does not match an audited far contact')
                rows.append(dict(row, release=card['id'], seed=card['seed'], reply_seed=card['reply_seed']))
    return rows


def metrics(errors, group):
    if not len(errors):
        return None
    w = weights(group)
    mae = w @ abs(errors)
    return dict(mae=mae.tolist(), rmse=np.sqrt(w @ np.square(errors)).tolist(),
                position_mae_m=float(w @ np.linalg.norm(errors[:, :2], axis=1)),
                velocity_mae_mps=float(w @ np.linalg.norm(errors[:, 2:], axis=1)))


def paired_interval(cue, ball, ids, part):
    groups = [
        np.mean(np.linalg.norm(cue[ids == i, part], axis=1) - np.linalg.norm(ball[ids == i, part], axis=1))
        for i in np.unique(ids)
    ]
    rng = np.random.default_rng(371080)
    samples = rng.choice(groups, size=(2000, len(groups))).mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def assess(models, rows):
    usable = [r for r in rows if r['paddle'] is not None]
    if not usable:
        return dict(total=len(rows), paired=0)
    raw = {}
    errors = {}
    for mode, model in models.items():
        x, y, groups = matrix(usable, mode)
        errors[mode] = predict(model, x) - y
        raw[mode] = metrics(errors[mode], groups)
    leads = np.array([r['lead_s'] for r in usable])
    intervals = {f'{mode}_minus_ball_{unit}': paired_interval(errors[mode], errors['ball'], groups, part)
                 for mode in ('cue', 'motion') for unit, part in (('position_m', slice(None, 2)),
                                                                  ('velocity_mps', slice(2, None)))}
    return dict(total=len(rows), paired=len(usable), releases=len(np.unique(groups)),
                lead_ms=np.quantile(1000. * leads, [.1, .5, .9]).tolist(), results=raw,
                bootstrap_intervals_95=intervals)


def call(practice, held, output, checkpoint, deploy_phase='causal', line=1.):
    pd, hd = [_read(path) for path in practice], [_read(path) for path in held]
    if any(d['pool'] != 'practice' for d in pd) or any(d['pool'] == 'practice' for d in hd):
        raise ValueError('Practice and held sets must retain their original roles')
    cards = lambda ds: [p['card'] for d in ds for p in d['rows']]
    for key in ('id', 'seed', 'reply_seed'):
        if {c[key] for c in cards(pd)} & {c[key] for c in cards(hd)}:
            raise ValueError(f'Forecast evaluation overlaps practice {key} values')
    if not all(d.get('complete') for d in pd + hd):
        raise ValueError('A forecast data collection is incomplete')
    phases = ('causal', '100', '167', '234')
    report = dict(schema='tt37-cue-forecast-1', fit={p.name: _hash(p) for p in practice},
                  held={p.name: _hash(p) for p in held},
                  input_feature_names=['x', 'y', 'z', 'vx', 'vy', 'vz'],
                  target_plane_y=-.40, target=['x', 'z', 'vx', 'vy', 'vz'], phases={})
    for phase in phases:
        all_train = group_rows(pd, phase)
        train = [r for r in all_train if r['paddle'] is not None]
        models, scores = {}, {}
        for mode in MODES:
            models[mode], scores[mode] = choose(*matrix(train, mode), mode)
        if phase == deploy_phase:
            # The engineering choice is fixed entirely by practice cross-validation.
            best = min(('motion', 'cue'), key=lambda k: scores[k])
            meta = {} if phase == 'causal' and line == 1. else dict(training_phase=phase, line=line)
            _write(checkpoint, dict(models=models, chosen=best, xval=scores, fit=report['fit'],
                                    practice_releases=sorted({r['release'] for r in all_train}),
                                    practice_seeds=sorted({r['seed'] for r in all_train}),
                                    practice_reply_seeds=sorted({r['reply_seed'] for r in all_train}), **meta))
            if line != 1.:
                if any(d.get('line', 1.) != line for d in hd):
                    raise ValueError('Causal assessment used the wrong inbound gate')
                report['deployed_gate'] = dict(y_m=line, training_phase=phase, chosen=best,
                    held={d['pool']: dict(all=assess(models, group_rows([d], 'causal')),
                                          opening=assess(models, group_rows([d], 'causal', later=False)),
                                          later=assess(models, group_rows([d], 'causal', later=True)))
                          for d in hd})
        report['phases'][phase] = dict(practice=dict(total=len(all_train), paired=len(train),
                                                     releases=len({r['release'] for r in train}), xval=scores),
                                       held={d['pool']: dict(
                                           legally_delivered_contacts=sum(p['far'] for p in d['rows']),
                                           physical_far_contacts=sum(len(p['far_contacts_s']) for p in d['rows']),
                                           emitted_gates=sum(p['gate_total'] for p in d['rows']),
                                           all=assess(models, group_rows([d], phase)),
                                           opening=assess(models, group_rows([d], phase, later=False)),
                                           later=assess(models, group_rows([d], phase, later=True)))
                                             for d in hd})
    _write(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--practice', nargs='+', type=Path, required=True)
    parser.add_argument('--held', nargs='*', type=Path, default=())
    parser.add_argument('--deploy-phase', choices=('causal', '100', '167', '234'), default='causal')
    parser.add_argument('--line', type=float, default=1.)
    args = parser.parse_args()
    call(args.practice, args.held, args.output, args.checkpoint, args.deploy_phase, args.line)
