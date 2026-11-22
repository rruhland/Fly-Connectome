"""Factorial evaluator-only substitutions locate the error in a camera-driven motor forecast."""

from dataclasses import replace
from itertools import product

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import ORIGIN, observed_ball
from .tt22_flight_inverse import FlightInverse
from .tt22_opponent import RallyArena
from .tt22_ordinary_game import DIR
from .tt22_rally_trial import CONFIG
from .tt22_value_player import TrajectoryDrySimulator
from .tt23_optical_contact import OUT

FACTORS = list(product([False, True], repeat=3))


def point(row, arena, engine, inverse, law):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    engine.twin.reset()
    commands = np.asarray(row['actions'])
    far = np.asarray(row['rival_actions'])
    sight = dict(row['optical'])
    hist, true_xyz, trials = [], [], []
    shots = {s['tick']:s for s in row['shots']}
    orig_aim,orig_direct=arena._aim,arena._command_to
    try:
        for t,a in enumerate(commands):
            p=arena.ball.copy()-ORIGIN
            true_xyz.append(p)
            hist.append((t,None if t not in sight else np.array(sight[t])))
            if t in shots:
                s=shots[t]
                obs=observed_ball(hist,optical_start(hist,law.gravity),law)
                f=inverse.fit(hist)
                if f is not None:
                    obs=replace(obs,velocity=f[1],spin=f[2])
                horizon=min(len(commands)-t,int(np.ceil(s['eta']))+6)
                variants=replace(obs,center=np.array([p if z[0] else obs.center for z in FACTORS]),
                                 velocity=np.array([arena.velocity if z[1] else obs.velocity for z in FACTORS]),
                                 spin=np.array([arena.spin if z[2] else obs.spin for z in FACTORS]))
                tape=np.broadcast_to(commands[t:t+horizon,:5],(len(FACTORS),horizon,5))
                preds=engine.paths(tape,np.full(len(FACTORS),horizon),variants,horizon)
                trials.append((dict(tick=t, seed=row['seed'], reference_tick=t+horizon, fit=(f is not None),
                                    eta=s['eta'],center_error=(obs.center-p).tolist(),
                                    velocity_error=(obs.velocity-arena.velocity).tolist(),
                                    spin_error=(obs.spin-arena.spin).tolist(),observed_spin=obs.spin.tolist(),
                                    ref_spin=arena.spin.tolist(),ref_velocity=arena.velocity.copy().tolist()),
                               preds[:,-1]))
            arena._aim=arena._command_to=lambda *args:far[t]
            _,r,done,cap,_=arena.step(a,render=False)
            if bool(done or cap)!=(t+1==len(commands)):
                raise ValueError('original rally ended on a different frame')
            engine.twin.advance(a)
        true_xyz.append(arena.ball.copy()-ORIGIN)
        if arena.point_result()!=row['outcome'] or r!=row['reward']:
            raise ValueError('recorded point outcome failed to replay')
    finally:
        arena._aim,arena._command_to=orig_aim,orig_direct
    return [dict(item,errors=np.linalg.norm(paths-true_xyz[item['reference_tick']],axis=1).tolist())
            for item,paths in trials]


def run():
    arena=RallyArena(**CONFIG)
    law=FlightLaw()
    engine=TrajectoryDrySimulator(law)
    inverse=FlightInverse()
    rows=[]
    try:
        for r in _read(DIR/'ordinary-game.json.gz')['branches']['frozen']:
            rows.extend(point(r,arena,engine,inverse,law))
    finally:
        arena.close()
        engine.close()
    errors=np.array([r['errors'] for r in rows])
    result=dict(rows=rows, factors=FACTORS, n=len(rows), metric='distance from true position six frames after forecast intercept, capped by point end',
                median=np.median(errors,axis=0).tolist(),mean=np.mean(errors,axis=0).tolist())
    print(result['n'],list(zip(FACTORS,result['median'],result['mean'])),flush=True)
    _write(OUT/'forecast-factorial-diagnostic.json.gz',result)


if __name__=='__main__':
    run()
