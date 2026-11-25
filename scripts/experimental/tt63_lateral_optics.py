"""Independently check actual stereo and support on the translated clamp."""

from collections import Counter
from pathlib import Path

import numpy as np

from . import tt63_lateral_clamp as fixed
from . import tt63_fixture_qualification as q
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt27_mount_position import paddle_origin
from .tt37_opponent_audit import guarded, verify_clearance


OUT=fixed.ROOT/'optical-calibration.json'
BASE=fixed.BASE


def survey():
    d=fixed.check(); geometry=_read(fixed.GEOMETRY)
    offsets=fixed.decision(geometry); g={r['dx']:r for r in geometry['rows']}; data={}
    if not geometry['complete'] or len(offsets)!=2:
        raise ValueError('No mechanically accepted surveyed offset')
    results=dict(protocol=_hash(fixed.PLAN),geometry=_hash(fixed.GEOMETRY),known_encoder_poses=[],rows=[],complete=False)
    for dx in offsets:
        # Identical *encoder* reference for static calibration, not the
        # separately commissioned game ready. The translation of the pad's
        # red RGB silhouette is available without exact base or ball state.
        with fixed.slid(dx),BASE.centered(d['ready']):
            a=BASE.arena(detail='cad')
            try:
                a.streams=('left','right'); guard=q.GripClearance(a.model,stock=True)
                a.reset(seed=501,render=False)
                near=list(guard.distances(a.data,horizon=.025)); s=fixed.fixture.support_check(a,'stock')
                est=paddle_origin(a)
                results['known_encoder_poses'].append(dict(dx=dx,q=a.joints.tolist(),silhouette_xyz=est.tolist(),
                       support=s,conservative_closest_m=min((f[2] for f in near),default=.025)))
            finally:a.close()
    reference=results['known_encoder_poses'][0]['silhouette_xyz']
    for r in results['known_encoder_poses']:
        est=np.array(r['silhouette_xyz'])-reference
        r['xyz_delta_m']=est.tolist(); r['dx_error_m']=float(est[0]-r['dx'])
    with fixed.fixture.school.enabled():
        for spec in d['development']:
            p=Path(spec['source'])
            old=data[p] if p in data else data.setdefault(p,_read(p)); c,row,expect=q.get(old,spec)
            pivot=expect['offer']['near_bounce']['time']; open_=expect['offer']['far_contact']['step']
            # Freeze an early physical packet independent of candidate output.
            stop=max(open_+9,int(np.floor((pivot+.093)/BASE.DT)))
            rec=dict(id=c['id'],tick_start=open_,tick_stop=stop,arms=[])
            for dx in offsets:
                ready=g[dx]['ready']; events=[]; obs=[]
                with fixed.slid(dx),BASE.centered(ready):
                    a=BASE.arena(); cad=BASE.arena(width=64,height=48,detail='cad'); camera,rig,*_=BASE.resources(a)
                    try:
                        q.tape.setup(a,c,row,expect); guarded(a,q.GripClearance(cad.model,stock=True))
                        eye=BASE.Vision(rig,a.control_dt,raw=True)
                        eye.eye=BASE.LearnedOptics(eye.eye,fixed.fixture.school.course.OPTIC)
                        seen=Counter()
                        for tick in range(stop+1):
                            frames=camera.capture() if tick%2==0 and tick>=open_-8 else None
                            x=BASE.observed(eye,tick,frames,a.joints)
                            if frames is not None and tick>=open_:
                                seen.update(eye.eye.tracker.last_frame_views)
                            if tick>=open_ and x is not None and x[4]<-.90:
                                truth=np.r_[a.ball,a.velocity]
                                obs.append(dict(tick=tick,x=x.tolist(),error=(x-truth).tolist()))
                            if tick==stop:
                                break
                            _,_,done,cap,_=a.exchange(np.zeros(6),BASE.FAR_COMMAND)
                            if done or cap:break
                        verify_clearance(a); events=a.events.copy(); clearance=a.clearance_result()
                    finally:camera.close();cad.close();a.close()
                rec['arms'].append(dict(dx=dx,tracked=len(obs),captures=dict(seen),obs=obs,clearance=clearance,
                                        physical_prefix_sha256=digest_json(events)))
            rec['same_prefix']=rec['arms'][0]['physical_prefix_sha256']==rec['arms'][1]['physical_prefix_sha256']
            results['rows'].append(rec);_write(OUT,results)
            print(c['id'],rec['same_prefix'],[(r['dx'],r['tracked'],r['captures'],r['clearance']['penetration_samples']) for r in rec['arms']],flush=True)
    result={}
    for dx in offsets:
        values=[a for r in results['rows'] for a in r['arms'] if a['dx']==dx]
        delta=np.array([o['error'] for a in values for o in a['obs']])
        result[str(dx)]=dict(samples=len(delta),median_position_m=float(np.median(np.linalg.norm(delta[:,:3],axis=1))),
                    median_velocity_mps=float(np.median(np.linalg.norm(delta[:,3:6],axis=1))),
                    stereo_frames={k:sum(a['captures'].get(k,0) for a in values) for k in ('left','right')},
                    full_cad_samples=sum(a['clearance']['samples'] for a in values),
                    penetrations=sum(a['clearance']['penetration_samples'] for a in values))
    results.update(complete=True,summary=result,boundary='Common encoder diagnostic locates the physical mount by red RGB only. Dynamic evaluation uses two fixed public RGB streams at the ordinary cadence and encoders. Native ball state is read after the optical estimate, for comparison only. No mounting or optic calibration is inferred from candidate rewards.')
    _write(OUT,results);print('optics',result,flush=True)


if __name__=='__main__':
    survey()
