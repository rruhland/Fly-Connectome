"""Isolate tangential patch velocity at an already attained native receive, for diagnosis."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt95_patch_lessons as labels
from . import tt88_mount_transfer as tilt
from . import tt90_installation_learning as tt90
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=labels.ROOT
PLAN=ROOT/'attained.protocol.json'
OUT=ROOT/'attained.json.gz'
H=9


def setup():
    source=_read(tt90.system.OUT)
    candidates=[r for r in source['rows'] if r['body']==1 and r['case']['role']=='low' and r['broad']]
    if not source['complete'] or len(candidates)<2:
        raise ValueError('The certified mounted body has not finished its independent audit')
    selected=candidates[:2]
    plan=dict(schema='tt95-known-attained-velocity-1',assessment=_hash(tt90.system.OUT),
              trained=_hash(labels.MODEL),motor=_hash(tilt.MODEL),ticks=H,
              cases=[r['case']['id'] for r in selected],
              rule='Training-excluded TT84 physical success receipts are used only as diagnostic pose clocks. '
                   'Hold identical real incoming and preceding stock/mounted commands. From preceding measured joints '
                   'and refs apply either full or normal-only attained endpoint from same learned inverse. Actual real '
                   'near landed, far answered and CAD, not a policy score or a fit to these held targets.')
    if not PLAN.exists():
        _write(PLAN,plan)
    if _read(PLAN)!=plan:
        raise ValueError('Changed known attainable mechanism study')
    return selected


def harness():
    anchors=setup()
    m,ready,_=tt90.material()
    q,tape=tt90.system.q,tt90.system.q.tape
    result=_read(OUT) if OUT.exists() else dict(protocol=_hash(PLAN),rows=[],complete=False)
    if result['protocol']!=_hash(PLAN):
        raise ValueError('Changed native attainable calibration')
    if result['complete']:
        return
    with tt90.school.enabled(cameras=False),tt90.system.installed(m['v'],m['mount']),tt90.BASE.centered(ready):
        a=tt90.BASE.arena(width=64,height=48)
        cad=tt90.BASE.arena(width=64,height=48,detail='cad')
        clear=q.GripClearance(cad.model,stock=True)
        try:
            body=labels.inv.Body(a)
            model=labels.inv.Inverse(body,a,tilt.MODEL)
            for entry in anchors[len(result['rows']):]:
                case=entry['case']
                ref=next(r for r in entry['trial']['physical'] if q.received(r,True))
                c,old,expect=q.get(_read(Path(case['source'])),case)
                when=ref['contact']['time_s']
                hit_tick=int(np.floor(when/labels.DT - 1e-8))
                cut=hit_tick-H
                tape.setup(a,c,old,expect)
                enc,times=[],[]
                start=None
                for j,u in enumerate(ref['commands'][:hit_tick]):
                    enc.append(a.joints.copy())
                    times.append(float(a.data.time))
                    if j==cut:
                        position,rate=labels.encoder_at(times,enc,times[-1])
                        state=np.r_[position,rate,a.data.ctrl[a._arm_ids[1][2]][:5]]
                        start=(state.copy(),a.events.copy())
                        body.read(a)
                    a.exchange(u,tt90.BASE.FAR_COMMAND)
                expected=[e for e in ref['events'] if e['time']<=a.data.time+1e-9]
                if digest_json(expected)!=digest_json(a.events) or start is None:
                    raise ValueError('An unchanged attained movement diverged before contact')
                pose=labels.inv.tt72.blade(a)
                p,R,T=np.asarray(pose['p']),np.asarray(pose['R']),np.asarray(pose['twist'])
                n=R[:,2]
                loc=np.asarray(ref['placements']['delivered'][0]['near_contact']['location'])
                radius=labels.tangent(n,loc-p)
                uv=labels.coords(n,radius)
                full=T[3:]+np.cross(T[:3],radius)
                norm=full-labels.tangent(n,full)
                paths=[]
                for method,v in [('normal',norm),('full',full)]:
                    target=np.r_[p+radius,n,v]
                    rates,_=model.solve(start[0],target,uv,H)
                    out=body.eval(rates[None])[0]
                    goal=labels.inv.world_goal(out,uv)
                    u=np.c_[rates,np.zeros(H)]
                    suffix=[x.tolist() for x in u]
                    suffix.extend((u[-1][None,:]*np.array([.38,.25,.12,.06])[:,None]).tolist())
                    r=tape.matched_motion(a,c,old,expect,ref['commands'][:cut]+suffix+[[0.]*6]*155,clear)
                    if not labels.inv.tt72.first_ball(r,ref['placements']['delivered'][0]) or not r['same']:
                        raise ValueError('The mechanism trial changed the physical receiving ball')
                    if q.received(r,True):
                        re=tape.matched_motion(a,c,old,expect,r['commands'],clear)
                        if not q.received(re,True) or digest_json(r['events'])!=digest_json(re['events']):
                            raise ValueError('A regained full face did not repeat')
                        r['repeat_sha256']=digest_json(re['events'])
                    r.update(method=method,target=target.tolist(),achieved_body_no_ball=goal.tolist())
                    paths.append(r)
                    print('TT95 attained',c['id'],method,r['landed'],r['answered'],r['contact'],flush=True)
                result['rows'].append(dict(case=case,cut=cut,clock=hit_tick,tangent=labels.tangent(n,full).tolist(),
                                           baseline_contact=ref['contact'],baseline_departure=ref['depart'],paths=paths))
                result['complete']=len(result['rows'])==len(anchors)
                _write(OUT,result)
        finally:
            a.close();cad.close()


if __name__=='__main__':
    harness()
