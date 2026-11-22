"""Fit a ball-to-arm registration from the observed consequences of real motor strokes."""

import argparse
from dataclasses import replace
from itertools import product

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import observed_ball
from .tt22_flight_inverse import FlightInverse
from .tt22_ordinary_game import DIR
from .tt22_value_player import TrajectoryDrySimulator


OUT = ROOT / 'artifacts/tt23'
PRACTICE = OUT / 'registration-practice.json.gz'
OFFSETS = np.array(list(product([0., -.006, .006], [0., -.010, .010], [0., -.004, .004, -.008, .008])))
DONORS = ('rally-anchor-612423769', 'rally-anchor-220381117', 'rally-anchor-8351422',
          'rally-anchor-1505337903-zero-guard', 'paired-course', 'ordinary-game')


def donor_rows(name):
    d = _read(DIR / (name + '.json.gz'))
    if 'branches' in d:
        return d['branches']['frozen']
    if isinstance(d['rows'], dict):
        return d['rows']['frozen']
    return d['rows']


def shots(row, engine, inverse, law):
    """Fit using observed paths on real commands, including the strokes that missed."""
    motor = np.asarray(row['actions'])
    optical = {t: np.array(v) for t,v in row['optical']}
    decisions = {s['tick']:s for s in row['shots']}
    engine.twin.reset()
    history = []
    for tick,action in enumerate(motor):
        history.append((tick, optical.get(tick)))
        if tick in decisions:
            shot = decisions[tick]
            horizon = min(len(motor)-tick, 21, int(np.ceil(shot['eta']))+9)
            seen = [i for i in range(1,horizon+1) if tick+i in optical]
            if len(seen)>=6 and any(i>=shot['eta']+3 for i in seen):
                state = optical_start(history, law.gravity)
                obs = observed_ball(history, state, law)
                inferred = inverse.fit(history)
                if inferred is not None:
                    obs = replace(obs, velocity=inferred[1], spin=inferred[2])
                # Vector observations broadcast different optical registrations over the same motor tape.
                variants = replace(obs, center=obs.center + OFFSETS)
                tape = np.broadcast_to(motor[tick:tick+horizon,:5],(len(OFFSETS),horizon,5))
                model_paths = engine.paths(tape, np.full(len(OFFSETS), horizon), variants, horizon)
                targets = np.array([optical[tick+i] for i in seen])
                # Observed and predicted trajectory are compared in camera coordinates.
                err = np.linalg.norm(model_paths[:,seen] - OFFSETS[:,None,:] - targets, axis=2)
                after = np.asarray(seen)>=shot['eta']+2
                yield dict(seed=row['seed'], tick=tick, after_bounce=bool(obs.bounced),
                           z_speed=float(obs.velocity[2]), optical_frames=len(seen),
                           errors_m=np.mean(np.minimum(.30,err[:,after]),axis=1).tolist())
        engine.twin.advance(action)


def select(rows):
    return int(np.argmin(np.mean([r['errors_m'] for r in rows],axis=0)))


def fit():
    law = FlightLaw()
    engine = TrajectoryDrySimulator(law)
    inverse = FlightInverse()
    rows, visited = [], set()
    try:
        for source in DONORS:
            for row in donor_rows(source):
                if row['seed'] in visited:
                    continue
                visited.add(row['seed'])
                rows.extend(dict(s,source=source) for s in shots(row,engine,inverse,law))
            print(source, len(rows), flush=True)
    finally:
        engine.close()
    seeds = np.array([r['seed'] for r in rows])
    hold = []
    for seed in sorted(set(seeds)):
        best = select([r for r in rows if r['seed']!=seed])
        for r in rows:
            if r['seed']==seed:
                hold.append([r['errors_m'][0],r['errors_m'][best]])
    best = select(rows)
    d = dict(metric='mean capped Euclidean camera residual after nominal impact', source=list(DONORS),
             offsets=OFFSETS.tolist(), best=best, offset_m=OFFSETS[best].tolist(),
             rows=rows, held_by_entire_seed=np.mean(hold,axis=0).tolist(), held_median=np.median(hold,axis=0).tolist())
    _write(PRACTICE, d)
    print(len(rows),d['offset_m'],d['held_by_entire_seed'],d['held_median'],flush=True)


if __name__=='__main__':
    fit()
