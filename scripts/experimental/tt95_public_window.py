"""Prospective public contact-window proposal from earned mounted observations."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import tt95_patch_play as point
from . import tt95_motion_window as tube
from . import tt95_patch_lessons as lab
from . import tt88_public_inverse as public
from . import tt90_installation_learning as tt90
from .tt37_optical_clinic import OpticalClinic
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=lab.ROOT
PLAN=ROOT/'window-public.protocol.json'
OUT=ROOT/'window-public.json.gz'
H=point.H


class Proposal:
    """Reconstruct a brief target window from camera forecasts and scored motions."""
    def __init__(self,twin,mode,exclude):
        self.target=point.PatchRoots(twin,'normal',exclude)
        self.motor=self.target.inverse
        self.mode=mode
        B=np.load(lab.MODEL)
        try:
            clipped=~np.isin(B['id'],list(exclude))
            original=np.flatnonzero(clipped)
            d=_read(lab.MOUNTED)
            entries=[x for r in d['rows'] for x in r['labels']]
            stock=len(B['id'])-len(entries)
            practice=_read(lab.PRACTICE)
            traces={}
            for c,r in zip(practice['cards'],practice['rows'],strict=True):
                for block in r['optical_earned']:
                    traces[c['id'],block['tag'][-1]-1]=block['frames']
            self.omega={}
            for i in original:
                if i<stock:
                    continue
                x=entries[i-stock]; rows=traces[x['id'],x['leg']]
                cut=x['score_time']
                qv=lab.encoder_at([r[0]*lab.DT for r in rows],[r[2] for r in rows],cut)
                if qv is None:
                    continue
                q,v=qv
                _,_,_,W=self.motor.ik.frame(q,True)
                self.omega[int(np.flatnonzero(original==i)[0])]=W@v
        finally:
            B.close()

    def roots(self, engine, ref):
        if self.mode=='terminal':
            p,meta=self.target.paths(engine,ref)
            return p,[dict(x,method='terminal') for x in meta]
        state=np.r_[engine.previous[:5],engine.rate[:5],ref[:5]]
        out,notes=[],[]
        for cost,h,x,uv,i in self.target.target_options(engine):
            v=x[6:]-lab.tangent(x[3:6],x[6:])
            omega=self.omega.get(i,np.zeros(3))
            dt=np.arange(-2,2)*lab.DT
            normal=Rotation.from_rotvec(omega[None,:]*dt[:,None]).apply(x[3:6][None,:].repeat(4,axis=0))
            goal=np.c_[x[:3]+dt[:,None]*v,normal,v[None,:].repeat(4,axis=0)]
            cmd,meta=tube.optimize(self.motor,None,state,goal,uv,rounds=2,ticks=h+1)
            form,targets=ref.copy(),[]
            for u in cmd:
                m=self.motor.motor
                form[:5]=np.clip(form[:5]+m.stride*u,m.lo,m.hi)
                targets.append(form.copy())
            for j in range(H-len(targets)):
                form[:5]=np.clip(form[:5]+m.stride*cmd[-1]*.38/(1+j),m.lo,m.hi)
                targets.append(form.copy())
            out.append(np.asarray(targets)-ref)
            notes.append(dict(method='window',clock=h,lesson_id=int(self.target.data['id'][i]),
                              same_body=bool(self.target.data['body'][i]),omega=omega.tolist(),goal=goal.tolist(),
                              loss=meta['final']))
        return np.asarray(out),notes


def run():
    native=_read(tt90.system.OUT)
    front=[r for r in native['rows'] if r['body']==1 and r['case']['role']=='low']
    # Two low known failure routes, the first a calibration revisit, the
    # second not exposed to target-clock inference or motion-window tuning.
    ordered=[next(r for r in front if r['case']['id']==k) for k in (14429,15541)]
    exclude={r['case']['id'] for r in native['rows']}
    plan=dict(schema='tt95-window-public-1',native=_hash(tt90.system.OUT),lessons=_hash(lab.MODEL),
              measured_pivot=_hash(tube.OUT),cases=[r['case']['id'] for r in ordered],
              note='Public cameras and encoders only. Compare normal-only endpoint against same patch and '
                   'normal-only motion maintained over four forecast clocks. The 14429 setup is known; '
                   '15541 was outside the motion-window pilot. All installation exams are excluded from lesson retrieval. ')
    if not PLAN.exists():
        _write(PLAN,plan)
    if _read(PLAN)!=plan:
        raise ValueError('Changed public motion-window contrast')
    out=_read(OUT) if OUT.exists() else dict(protocol=_hash(PLAN),rows=[],complete=False)
    if out['protocol']!=_hash(PLAN):
        raise ValueError('Changed public motion-window receipt')
    if out['complete']:
        return
    m,ready,_=tt90.material()
    q,tape=tt90.system.q,tt90.system.q.tape
    with tt90.school.enabled(),tt90.system.installed(m['v'],m['mount']),tt90.BASE.centered(ready):
        a=tt90.BASE.arena()
        cad=tt90.BASE.arena(width=64,height=48,detail='cad')
        clear=q.GripClearance(cad.model,stock=True)
        cam,rig,*_=tt90.BASE.resources(a)
        try:
            for r in ordered[len(out['rows']):]:
                spec=r['case']
                c,row,anchor=q.get(_read(Path(spec['source'])),spec)
                raw=r['trial']['stock_commands']
                expected=raw['placements']['delivered'][0]
                values=[]
                for mode in ('terminal','window'):
                    engine=OpticalClinic(body=True)
                    try:
                        proposer=Proposal(engine.twin,mode,exclude)
                        trial=public.run_case(a,clear,cam,rig,engine,proposer,c,row,anchor,raw['commands'],mode)
                    finally:
                        engine.twin.close()
                    if not lab.inv.tt72.first_ball(trial,expected):
                        raise ValueError('Window changed matched far strike or near bounce')
                    if not trial['issued'] and digest_json(trial['events'])!=digest_json(raw['events']):
                        raise ValueError('A veto changed the original outgoing tape')
                    if q.received(trial):
                        re=tape.matched_motion(a,c,row,anchor,trial['commands'],clear)
                        if not q.received(re) or digest_json(re['events'])!=digest_json(trial['events']):
                            raise ValueError('Public window result did not reproduce')
                        trial['repeat_sha256']=digest_json(re['events'])
                    values.append(trial)
                    print('TT95 public window',c['id'],mode,trial['issued'],trial['landed'],trial['answered'],
                          sum(x['gate'].get('nominal_legal',0) for x in trial['notes']),flush=True)
                out['rows'].append(dict(case=spec,reference_landed=raw['landed'],results=values))
                out['complete']=len(out['rows'])==len(ordered)
                _write(OUT,out)
        finally:
            cam.close();cad.close();a.close()


if __name__=='__main__':
    run()
