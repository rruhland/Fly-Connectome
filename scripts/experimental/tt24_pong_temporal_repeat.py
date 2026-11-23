"""Repeat the fixed temporal-gain experiment across independent source fits."""

import argparse

import numpy as np
import torch

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM
from .tt24_pong_temporal_source import GRID, LOCK, PLAN, fit, points, summary
from .tt24_pong_temporal_wrist import DIAGNOSTIC, HELD, effects, run, starts, variant


REPEAT_PLAN = ROOT / 'pong-temporal-repeat-plan.json'
REPEAT_SOURCE = ROOT / 'pong-temporal-repeat-source.json.gz'
REPEAT_RESULT = ROOT / 'pong-temporal-repeat-summary.json'
STREAMS = 4
MASTER = 927034665
MODES = ('cold', 'always_1', 'always_2', 'pong', 'shuffled')


def register():
    root = np.random.default_rng(MASTER)
    counts = dict(native_practice=80, native_held=32, wrist=6)
    source_parent = _read(PLAN)
    native_used = {start for seeds in source_parent['rosters'].values() for start in seeds}
    wrist_used = set(_read(DIAGNOSTIC)['seeds']) | set(_read(HELD)['seeds'])
    streams = []
    for _ in range(STREAMS):
        row = {name: starts(int(root.integers(1, 2**31)), count) for name, count in counts.items()}
        row['shuffle_seed'] = int(root.integers(1, 2**31))
        for name, used in (('native_practice', native_used), ('native_held', native_used), ('wrist', wrist_used)):
            if used.intersection(row[name]):
                raise ValueError('replication starts intersect another roster')
            used.update(row[name])
        streams.append(row)
    plan = dict(schema='tt24-pong-temporal-repeat-1', parent_plan_sha256=_hash(PLAN), parent_fit_sha256=_hash(LOCK),
                repetitions=streams, counts=counts, modes=list(MODES), same_gain_comparison='exact deterministic alias',
                change_to_forecast='none', changed_source_or_target_physics=False, wrist_gain_practice=0)
    if REPEAT_PLAN.exists() and _read(REPEAT_PLAN) != plan:
        raise ValueError('replication design differs from the recorded plan')
    _write(REPEAT_PLAN, plan)
    return plan


def learn(plan):
    torch.set_num_threads(1)
    weights = np.asarray(_read(WARM)['weights'])
    receipt = dict(plan_sha256=_hash(REPEAT_PLAN), streams=[])
    if REPEAT_SOURCE.exists():
        receipt = _read(REPEAT_SOURCE)
        if receipt['plan_sha256'] != _hash(REPEAT_PLAN):
            raise ValueError('already fitted source cohorts differ from the replication plan')
    for i, roster in enumerate(plan['repetitions'][len(receipt['streams']):], start=len(receipt['streams'])):
        benefit, practice_tapes = points(weights, GRID, roster['native_practice'], True)
        scores = benefit[..., 0]
        rng = np.random.default_rng(roster['shuffle_seed'])
        ind = np.column_stack([rng.permutation(len(GRID)) for _ in roster['native_practice']])
        fits = dict(pong=fit(scores), shuffled=fit(np.take_along_axis(scores, ind, axis=0)))
        gains = dict(cold=0., always_1=1., always_2=2., pong=fits['pong']['gain'], shuffled=fits['shuffled']['gain'])
        held, held_tapes = points(weights, list(gains.values()), roster['native_held'], True)
        stats = {name: summary(row) for name, row in zip(gains, held, strict=True)}
        receipt['streams'].append(dict(stream=i, fits=fits, gains=gains, outcomes=stats,
                                       native_practice=benefit.tolist(), native_action_sha256=practice_tapes.tolist(),
                                       native_held=held.tolist(), held_action_sha256=held_tapes.tolist(),
                                       practice_points=int(np.prod(benefit.shape[:2])),
                                       practice_frames=int(benefit[..., 2].sum())))
        _write(REPEAT_SOURCE, receipt)
        print('repeat source', i, 'gains', gains, 'held net', {k: v['net'] for k, v in stats.items()}, flush=True)
    return receipt


def deploy(plan, learned):
    pooled = {name: [] for name in MODES}
    receipt = dict(plan_sha256=_hash(REPEAT_PLAN), source_sha256=_hash(REPEAT_SOURCE), streams=[])
    for i, (roster, source) in enumerate(zip(plan['repetitions'], learned['streams'], strict=True)):
        gains = source['gains']
        numerical = {}
        aliases = {mode: numerical.setdefault(gains[mode], mode) for mode in MODES}
        modes = {name: variant(gain) for gain, name in numerical.items()}
        path = ROOT / f'pong-temporal-repeat-{i}.json.gz'
        outcomes = run(path, roster['wrist'], modes,
                       dict(source_sha256=_hash(REPEAT_SOURCE), stream=i, aliases=aliases,
                            wrist_gain_practice=0, camera_capture_time_public=True))
        scores = {name: outcomes['summary'][actual] for name, actual in aliases.items()}
        data = {name: outcomes['conditions'][actual] for name, actual in aliases.items()}
        for name in MODES:
            pooled[name].extend(data[name])
        receipt['streams'].append(dict(stream=i, gains=gains, aliases=aliases, scores=scores,
                                       differences={name: effects(data['pong'], data[name])
                                                    for name in MODES if name != 'pong'},
                                       unique_replays=sum(len(row) for row in outcomes['conditions'].values())))
        receipt['pooled_totals'] = {name: sum(row['run'] for row in pooled[name]) for name in MODES}
        receipt['pooled_effects'] = {name: effects(pooled['pong'], pooled[name]) for name in MODES if name != 'pong'}
        _write(REPEAT_RESULT, receipt)
        print('repeat wrist', i, {name: row['total'] for name, row in scores.items()}, receipt['pooled_effects'],
              flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    protocol = register()
    bank = learn(protocol)
    if not args.source_only:
        deploy(protocol, bank)
