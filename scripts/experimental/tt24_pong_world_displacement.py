"""Isolate the observed active Pong displacement from persistence of its current pose."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_pong_world import OUTPUT, RAW, aligned, flatten, invert, nonlinear
from .tt24_pong_world_train import register, run
from .tt24_world_prior import WEIGHTS
from .tt24_world_prior_train import save


PREFIX = 'pong-world-displacement'
PLAN = OUTPUT / f'{PREFIX}-plan.json'
CUT = OUTPUT / f'{PREFIX}-initial.json'
RECEIPT = OUTPUT / f'{PREFIX}-learning.json.gz'
SCHEDULE = 6460858100
SHUFFLE = 414316664


def selected(rows):
    seed, values = flatten(rows)
    active = np.array([cue['signed_alias'] != 0 for episode in rows['episodes'] for cue in episode['cues']])
    return seed[active], values[active], values


def motion_fit():
    raw = _read(RAW)['rows']
    _, train, sample = selected(raw['practice'])
    ids, held, _ = selected(raw['held'])
    speed = float(np.median(np.abs(sample[:, 3])))
    x, later, now = aligned(train, speed)
    q, wanted, current = aligned(held, speed)
    y = later - now
    change = wanted - current
    rng = np.random.default_rng(SHUFFLE)
    bins = np.column_stack((np.sign(train[:, 3]), np.round(now / .024),
                            np.minimum(2, (train[:, 1] / train[:, 6]).astype(int))))
    scrambled = y.copy()
    for bucket in np.unique(bins, axis=0):
        points = np.flatnonzero((bins == bucket).all(axis=1))
        scrambled[points] = rng.permutation(y[points])
    matrix = x.T @ x + 2.1 * np.eye(x.shape[1])
    estimates = {name: np.linalg.solve(matrix, x.T @ invert(label))
                 for name, label in (('pong', y), ('shuffled', scrambled))}
    estimates['matched_cold'] = rng.normal(size=x.shape[1])
    for name in ('shuffled', 'matched_cold'):
        estimates[name] *= np.linalg.norm(estimates['pong']) / max(1.e-8, np.linalg.norm(estimates[name]))
    estimates['cold'] = np.zeros(x.shape[1])
    audit, initial = {}, {}
    for name, coef in estimates.items():
        predicted = nonlinear(q @ coef)
        definite = np.abs(change) > .003
        audit[name] = dict(rmse=float(np.mean((predicted - change)**2)**.5),
                           rms_moving=float(np.mean((predicted[definite] - change[definite])**2)**.5),
                           correction_sign=float(np.mean(np.sign(predicted[definite]) == np.sign(change[definite]))))
        weights = np.zeros(WEIGHTS)
        weights[10:22] = coef
        initial[name] = weights.tolist()
        save(OUTPUT / f'{PREFIX}-initial-{name}.npz', weights)
    result = dict(schema='tt24-pong-world-displacement-init-1', native_sha256=_hash(RAW), initial=initial,
                  held=audit, native_train_active=len(train), native_held_active=len(held),
                  held_launches=int(len(set(ids))), held_moving_cues=int((np.abs(change) > .003).sum()),
                  speed_mapping_m_per_s=speed, initialized_coefficients=12, shuffle=SHUFFLE,
                  anchored='world begins with zero tangential blade displacement')
    _write(CUT, result)
    print('held native selected motion', audit, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    register(PLAN, SCHEDULE)
    result = motion_fit()
    if result['held']['pong']['rms_moving'] < result['held']['shuffled']['rms_moving']:
        run(args.workers, plan_path=PLAN, init_path=CUT, output=RECEIPT, schedule=SCHEDULE, prefix=PREFIX)


if __name__ == '__main__':
    main()
