"""Replayable hidden motor-interface changes in scored, first-person receiving points."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, Wiring, _counts, _hash, _metrics, _point, _write
from .tt5_checkpoint import load, setup
from .tt6_recall import InterfaceMemory, RecallPlayer


def course(receipt, checkpoint, seed, stream, schedules, modes, horizon, installation=0, cached=False):
    if not schedules or min(schedules) < 0:
        raise ValueError('expected a nonempty sequence of nonnegative wiring labels')
    _, _, base = setup(receipt, installation)
    try:
        if cached:
            from .tt6_render import cache_hall_stereo
            cache_hall_stereo(base.arena)
        labels = np.array(schedules)
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        new_ids = rng.integers(0, 2**31, size=labels.max())
        starts = rng.integers(0, 2**31, size=len(labels)).tolist()
        interfaces = [base] + [Wiring(base.arena, int(i)) for i in new_ids]
        runs = {}
        for mode in modes:
            original = load(checkpoint)
            fit = original.adapter
            player = (original if mode == 'frozen' else
                      RecallPlayer(original, InterfaceMemory(fit, remember=mode == 'carry', horizon=min(horizon, 12) if mode == 'short' else horizon,
                                                           early=mode != 'short')))
            rows = []
            for label, start in zip(labels, starts, strict=True):
                wiring = interfaces[label]
                row = _point(wiring, player, start)
                # Recover the physical wiring for diagnostics after a point has finished.
                true_order = np.argsort(wiring.order)[base.order[fit.scripts]]
                true_sign = (np.sign(fit.gains) * base.signs[fit.scripts] * wiring.signs[true_order]).astype(int)
                fitted_order = row.get('inferred_order', fit.scripts)
                fitted_sign = row.get('inferred_sign', np.sign(fit.gains))
                correct = (true_order == fitted_order) & (true_sign == fitted_sign)
                rows.append(dict(**row, **_counts(row), axes_correct=int(correct.sum()), context=int(label)))
            runs[mode] = dict(points=rows, summary=_metrics(rows))
            print(stream, mode, [(r['context'], r.get('probe_commands',0), r['axes_correct'],r['player_returns']) for r in rows], flush=True)
        return dict(stream=stream, point_seeds=starts, interface_seeds=new_ids.tolist(), runs=runs)
    finally:
        base.arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--contexts',type=int,nargs='+',default=[0,1,2,1,0,2])
    parser.add_argument('--modes',nargs='+',choices=['carry','refit','short','frozen'],default=['carry','refit','frozen'])
    parser.add_argument('--streams',type=int,default=1)
    parser.add_argument('--cached-render',action='store_true')
    parser.add_argument('--installation',type=int,default=0)
    parser.add_argument('--horizon',type=int,default=24)
    parser.add_argument('--receipt',type=Path,default=ROOT/'artifacts/tt5/first-person-source.json.gz')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/tt5/first-person-mini-to-hall.npz')
    a=parser.parse_args()
    result=dict(seed=a.seed,contexts=a.contexts,checkpoint_sha256=_hash(a.checkpoint),receipt_sha256=_hash(a.receipt),
                cached_render=a.cached_render, installation=a.installation, horizon=a.horizon, conditions=a.modes, streams=[])
    for i in range(a.streams):
        result['streams'].append(course(a.receipt,a.checkpoint,a.seed,i,a.contexts,a.modes,a.horizon,a.installation,a.cached_render))
        _write(a.output,result)
