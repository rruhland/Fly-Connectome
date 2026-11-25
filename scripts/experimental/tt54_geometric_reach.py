"""Separate the stock proposal support from a liberal geometric motor bound on archived offers."""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import tt38_joint_mpc as mpc
from .tt29_rally_physics import BladeIK, BallFlight, NOMINAL_NO_LOAD, desired_shot
from .tt4_transfer_trial import _read, _write, _hash
from .tt19_view_full_game import digest_json
from .tt54_motor_funnel import PLAN, ROOT, STOCK


def scan(geometry,path,encoder):
    # A conservative claim about a deliberately liberal fit. A shaft speed
    # box permits impossible instantaneous acceleration and does not certify
    # inertia, clearance, ball contact, arm rebound energy or robustness.
    records=[]
    warm={}
    for frame in path[::4]:
        t=float(frame[0]); p, v=frame[1:4],frame[4:7]
        if not -.84 > p[1] > -1.82:
            continue
        n_goal=desired_shot(p,v,1,[0.,.67],4.0)
        offset=.025
        bounds=(np.maximum(geometry.low,encoder-t*NOMINAL_NO_LOAD),
                np.minimum(geometry.high,encoder+t*NOMINAL_NO_LOAD))
        if np.any(bounds[0]+2e-7>=bounds[1]):
            continue
        for axis_weight in (.03,.10,.20):
            def residual(q):
                b, n=geometry.pose(q)
                return np.r_[b+offset*n-p, axis_weight*(n-n_goal),.003*(q-encoder)]

            def jac(q):
                _, _, point, normal=geometry.pose(q,jacobian=True)
                return np.r_[point+offset*normal,axis_weight*normal,.003*np.eye(5)]

            q0=np.clip(warm.get(axis_weight,encoder),bounds[0]+1e-7,bounds[1]-1e-7)
            q=least_squares(residual,q0,jac=jac,bounds=bounds,max_nfev=23).x
            warm[axis_weight]=q
            center,n=geometry.pose(q)
            d=p-center
            plane=float(n@d)
            radius=float(np.linalg.norm(d-plane*n))
            records.append(dict(time=t,p=p.tolist(),q=q.tolist(),center=center.tolist(),
                                normal=n.tolist(),normal_cos=float(n@n_goal),radial_m=radius,
                                plane_err_m=abs(plane-offset),shaft_frac=float(np.max(abs(q-encoder)/(t*NOMINAL_NO_LOAD)))))
    return records


def run(out=ROOT/'geometric-reach.json.gz'):
    plan=_read(PLAN)
    if plan['source_sha256']!=_hash(STOCK):
        raise ValueError('Changed precise physical offer source')
    baseline=_read(STOCK)
    result=dict(funnel_sha256=_hash(PLAN),known_physics_evaluator_only=True,rows=[],
                note='Uses archived exact ball state and a no-load kinematic box only to diagnose pose availability. A fit is not a guaranteed trajectory or a scored return. Stock camera and encoders remain the only actor input.')
    with mpc.centered(baseline['ready']):
        a=mpc.arena(width=64,height=48)
        try:
            geometry=BladeIK(a)
            flight=BallFlight(a)
            for r in plan['rows']:
                i=r['index']; old=baseline['rows'][i]; c=baseline['cards'][i]
                a.launch(old['shot'],card=c);a.partner.reset(seed=c['seed'])
                checks={t:s for s in r['scopes'] for t in (s['ticks'][0],s['ticks'][-1])}
                probes=[]
                for t,u in enumerate(old['actions'][:max(checks)+1]):
                    if t in checks:
                        previous=[e for e in old['events'] if e['time']<=a.data.time+1e-8]
                        if digest_json(previous)!=digest_json(a.events):
                            raise ValueError('Exact physical reach prefix diverged')
                        p=a.ball.copy();v=a.velocity.copy();spin=a.data.qvel[a._ball_dof+3:a._ball_dof+6].copy()
                        track=flight.path(p,v,spin,1,a._received_bounce,horizon=.35)
                        reach=scan(geometry,track,a.joints[:5].copy())
                        probes.append(dict(tick=t,scope=checks[t],p=p.tolist(),v=v.tolist(),
                                           encoder=a.joints.tolist(),path=track.tolist(),rows=reach))
                    a.exchange(u,mpc.FAR_COMMAND)
                result['rows'].append(dict(id=r['id'],probes=probes))
                _write(out,result)
                print(r['id'],[(p['tick'],len(p['rows']),
                               sum(y['radial_m']<.055 and y['plane_err_m']<.009 and y['normal_cos']>.84 for y in p['rows'])) for p in probes],flush=True)
        finally:
            a.close()


if __name__=='__main__':
    run()
