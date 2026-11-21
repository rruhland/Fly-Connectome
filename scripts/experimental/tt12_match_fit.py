"""Refine the arm model from visual motion during one completed table-tennis game."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt6_memory import _serializable_body, load
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt10_series_trial import _restore_tracker


def samples(model, requested, poses):
    # The same command-space step response as the player's predictive servo.
    requested = np.vstack([np.zeros(model.action_size),requested])
    eff = np.cumsum(requested,axis=0)
    for i,kernel in enumerate(model.kernels):
        eff[:,i] += lfilter(kernel-1.,[1.],requested[:,i])
    poses = np.asarray(poses)
    ticks = np.flatnonzero(np.isfinite(poses).all(axis=1))[::2]
    if not len(ticks):
        return np.empty((0,model.action_size+1)),np.empty((0,3))
    # Estimate the unobserved delay from RGB and command history alone.
    lags = range(5)
    states = [eff[np.maximum(0,ticks-d)] for d in lags]
    error = [np.linalg.norm(model.pose(q)[:,:3]-poses[ticks],axis=1) for q in states]
    delay = np.argmin([np.median(np.minimum(e,.14)) for e in error])
    q = states[delay]
    return np.c_[np.ones(len(q)),q], poses[ticks]-model.pose(q)[:,:3]


def refit(model, rows):
    x,y = (np.concatenate(t) for t in zip(*rows,strict=True))
    delta = np.zeros((x.shape[1],3))
    for _ in range(3):
        radius = np.linalg.norm(y-x@delta,axis=1)
        weights = np.minimum(1.,.035/np.maximum(radius,1e-6))
        normal = x.T @ (weights[:,None]*x)
        delta = np.linalg.solve(normal+np.diag([90.,*([30.]*model.action_size)]),x.T@(weights[:,None]*y))
    return delta


def cost(rows, delta):
    return float(np.mean([np.mean(np.minimum(np.linalg.norm(y-x@delta,axis=1),.18))
                          for x,y in rows if len(x)]))


def run(source, game_file, output, report):
    saved = _read(source)
    entry = saved['series']
    model = load(source)[0].models[entry['model']]
    receipt = _read(game_file)
    game = receipt['games'][-1]
    arena = TT8ServiceArena(**entry['arena'])
    wiring = TT6Wiring(arena,entry['arena']['seed'])
    rows = []
    try:
        cache_hall_stereo(arena,reflections=True)
        factory = _restore_tracker(arena.cameras,entry['optics'])
        for point in game['points']:
            # Phase, server and seed reconstruct the recorded environment only.
            interface = PointInterface(wiring,prepare_point(arena,point['phase']),server=point['server'],seed=game['seed'])
            tracker = factory()
            frames = interface.reset(seed=point['seed'])
            positions = []
            for i in range(len(point['actions'])+1):
                tracker.update(frames)
                actor = tracker.actor
                positions.append(actor.position if actor is not None and actor.observed else [np.nan]*3)
                if i < len(point['actions']):
                    frames,*_ = interface.step(point['actions'][i])
            if arena.point_result()!=point['evaluator'] or interface.applied!=point['applied']:
                raise ValueError('practice point did not replay')
            rows.append(samples(model,point['actions'],positions))
    finally:
        arena.close()
    n = max(1,len(rows)*2//3)
    delta = refit(model,rows[:n])
    steps = [0.,.5,1.]
    held = [cost(rows[n:],step*delta) for step in steps]
    step = steps[np.argmin(held)]
    coeff = model.coefficients.copy()
    correction = step*refit(model,rows)
    coeff[:model.action_size+1,:3] += correction
    saved['models'][entry['model']] = _serializable_body(replace(model,coefficients=coeff))
    _write(output,saved)
    _write(report,dict(source=_hash(source),practice_game=_hash(game_file),checkpoint=_hash(output),
                       samples=[len(x) for x,_ in rows],practice_points=n,held_optical_errors_m=held,
                       steps=steps,applied=step,correction=correction.tolist(),implementation=_hash(Path(__file__))))
    print(n,held,step,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','game','output','report'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    run(a.source,a.game,a.output,a.report)
