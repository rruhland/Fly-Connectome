"""Paired practice curves for two independent sets of real robot serves."""

from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write
from .tt22_joint_practice import point
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer_trial import MotorArena


def summarize(record, *, wide=False):
    names = ('pong', 'cold', 'unselected')
    held = record['optics' if wide else 'held_optics']
    ids = [row['seed'] for row in held]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate held serve')
    matrix = {}
    for name in names:
        assembled = []
        for rep in record['replicates']:
            stages = rep[name] if wide else rep['eval'][name]
            if [s['budget'] for s in stages] != [0, 8, 16, 24]:
                raise ValueError('incomplete practice curve')
            curve = []
            for stage in stages:
                rows = record['scored'][stage['index']]['rows'] if wide else stage['rows']
                if [row['seed'] for row in rows] != ids:
                    raise ValueError('mismatched evaluation starts')
                curve.append([row['legal'] > 0 for row in rows])
            assembled.append(curve)
        matrix[name] = np.array(assembled, dtype=float)
    boot = np.random.default_rng(508713)
    n, _, m = matrix['pong'].shape
    streams = boot.integers(n, size=(10000, n, 1))
    cases = boot.integers(m, size=(10000, 1, m))

    def interval(per_start):
        samples = per_start[streams, cases].mean(axis=(1, 2))
        return dict(delta=per_start.mean(), interval=np.quantile(samples, [.025, .975]).tolist(),
                    per_training_stream=per_start.mean(axis=1).tolist())

    contrasts = {}
    for name in names[1:]:
        delta = matrix['pong'] - matrix[name]
        contrasts[name] = dict(early=interval(delta[:, :3, :].mean(axis=1)), final=interval(delta[:, 3, :]))
    return dict(rates={k: a.mean(axis=(0, 2)).tolist() for k, a in matrix.items()},
                streams={k: a.mean(axis=2).tolist() for k, a in matrix.items()},
                observed=sum(r['mover'] is not None for r in held), serves=m,
                contrasts=contrasts)


def verify(source, wide):
    """Re-render each policy-independent prefix and physically replay the two starting policies."""
    arena = MotorArena()
    template = np.asarray(source['template'])
    verified = []
    try:
        for data, shifted in ((source, False), (wide, True)):
            optics = data['optics' if shifted else 'held_optics']
            rep = data['replicates'][0]
            arms = {}
            for arm in ('pong', 'cold'):
                key = rep[arm][0] if shifted else rep['eval'][arm][0]
                checkpoint = data['scored'][key['index']] if shifted else key
                arms[arm] = checkpoint
            for j, expected in enumerate(optics):
                obs = prefix(arena, expected['seed'], expected['lane'])
                if obs != expected:
                    raise ValueError(f"optical evidence changed at {expected['seed']}")
                for key, checkpoint in arms.items():
                    if point(arena, template, checkpoint['params'], obs) != checkpoint['rows'][j]:
                        raise ValueError(f"motor replay changed at {expected['seed']} {key}")
            verified.append(dict(rendered_prefixes=len(optics), physically_replayed=2*len(optics)))
    finally:
        arena.close()
    return verified


if __name__ == '__main__':
    directory = ROOT / 'artifacts/tt22'
    central = _read(directory / 'anonymous-joint-practice.json.gz')
    wide = _read(directory / 'anonymous-wide-hold.json.gz')
    evidence = dict(central=summarize(central), wide=summarize(wide, wide=True),
                    verification=verify(central, wide))
    _write(directory / 'practice-report.json.gz', evidence)
    for k in ('central', 'wide'):
        print(k, evidence[k]['contrasts'])
