"""Hold the patch motion across the collision clock instead of one free-body endpoint."""

from pathlib import Path

import numpy as np

from . import tt95_attained_velocity as prior
from . import tt95_patch_lessons as lab
from . import tt88_mount_transfer as tilt
from . import tt90_installation_learning as tt90
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=lab.ROOT
OUT=ROOT/'window-matched.json.gz'
PLAN=ROOT/'window.protocol.json'
N=10
WINDOW=np.array([N-3,N-2,N-1,N])
SCALES=np.repeat([.014,.12,.32],3)
WEIGHTS=np.array([.45,.68,1.0,.88])[:,None]


def body_source(block,uv):
    p,n,w,v=np.split(block[10:],[3,6,9])
    offset=lab.inv.coordinates(n,uv)
    return np.r_[p+offset,n,v+np.cross(w,offset)]


def rates_for(motor,state,ref):
    q=state[10:].copy()
    all=[]
    for g in ref:
        u=np.clip((np.clip(g,motor.lo,motor.hi)-q)/motor.stride,-.8,.8)
        q=np.clip(q+motor.stride*u,motor.lo,motor.hi)
        all.append(u)
    return np.asarray(all)


def optimize(model,body,state,wanted,uv,*,native=False,rounds=4,ticks=N):
    """Terminal patch velocity is tracked over multiple legal contact times."""
    fit=model.attainable(state,wanted[-1],uv,ticks)[0]
    base=model.nominal(state,*fit[1:],ticks)
    clock=np.arange(1,ticks+1)/ticks
    window=np.arange(ticks-3,ticks+1)
    scale=np.c_[np.minimum(1.,1.55*clock),1.35*(clock**3-.25*clock),
                4.*clock*(1.-clock),np.clip((clock-.67)*4,-.0,1.)]
    eye=np.eye(20)*.041
    plant=model.motor

    def objectives(parts):
        all_ref=base+np.einsum('tb,nbj->ntj',scale,parts.reshape(-1,4,5))
        outcomes=[]
        if native:
            cmds=np.asarray([rates_for(plant,state,x) for x in all_ref])
            features=np.stack([np.asarray([body_source(z,uv) for z in body.eval(cmds[:,:t])]) for t in window],axis=1)
        else:
            moved=np.broadcast_to(state,(len(all_ref),15)).copy()
            cmds=np.empty_like(all_ref)
            for i,r in enumerate(all_ref.transpose(1,0,2),1):
                u=np.clip((np.clip(r,plant.lo,plant.hi)-moved[:,10:])/plant.stride,-.8,.8)
                cmds[:,i-1,:]=u
                moved=plant.next(moved,u)
                if i in window:
                    outcomes.append([model.features(z[:10],uv) for z in moved])
            features=np.asarray(outcomes).transpose(1,0,2)
        loss=((features-wanted)/SCALES)*WEIGHTS
        return loss.reshape(len(features),-1),cmds,features

    theta=np.zeros(20)
    incumbent,program,last=objectives(theta[None])
    initial=float(np.linalg.norm(incumbent[0]))
    for _ in range(rounds):
        diff,_,_=objectives(theta[None]+eye)
        J=(diff-incumbent)/.041
        direction=-np.linalg.solve(J@J.T+.36*np.eye(20),J@incumbent[0])
        trial=np.clip(theta+np.array([.55,1.,1.45])[:,None]*np.clip(direction,-.26,.26),-.64,.64)
        value,dest,shape=objectives(trial)
        t=int(np.argmin(np.linalg.norm(value,axis=1)))
        if np.linalg.norm(value[t])>=np.linalg.norm(incumbent[0])-.0001:
            break
        theta,incumbent,program,last=trial[t],value[t:t+1],dest[t:t+1],shape[t:t+1]
    return program[0],dict(initial=initial,final=float(np.linalg.norm(incumbent[0])),
                           modeled=last[0].tolist(),native_learning_critic=native)


def setup():
    prior_result=_read(prior.OUT)
    candidates=prior.setup()
    p=dict(schema='tt95-window-contact-1',source=_hash(prior.OUT),cases=[r['case']['id'] for r in candidates],
           control_ticks=N,window_frames=WINDOW.tolist(),
           rule='Evaluator-only unchanged hard-ball pose references from deliberately independent paid TT84 '
                'receives. Use a short measured path of patch position, full blade normal and patch velocity '
                'from the unchanged action program. Do not supply that action program to the inverse. '
                'Compare normal-only and full path with the same learned motor, then a single full-path '
                'local correction from actual stock actuator dynamics without a live ball. Grade '
                'exact matched real delivery, physical legal landing and actual answer.')
    if not prior_result['complete']:
        raise ValueError('Complete terminal controls before examining duration')
    if not PLAN.exists():
        _write(PLAN,p)
    if _read(PLAN)!=p:
        raise ValueError('The fixed two-room contact study changed')
    return candidates


def run():
    anchors=setup()
    m,ready,_=tt90.material()
    q,tape=tt90.system.q,tt90.system.q.tape
    result=_read(OUT) if OUT.exists() else dict(protocol=_hash(PLAN),rows=[],complete=False)
    if result['protocol']!=_hash(PLAN):
        raise ValueError('Changed native phase-window study')
    if result['complete']:
        return
    with tt90.school.enabled(cameras=False),tt90.system.installed(m['v'],m['mount']),tt90.BASE.centered(ready):
        a=tt90.BASE.arena(width=64,height=48)
        cad=tt90.BASE.arena(width=64,height=48,detail='cad')
        clear=q.GripClearance(cad.model,stock=True)
        try:
            b=lab.inv.Body(a)
            model=lab.inv.Inverse(b,a,tilt.MODEL)
            for p in anchors[len(result['rows']):]:
                spec=p['case']
                ref=next(x for x in p['trial']['physical'] if q.received(x,True))
                c,old,expected=q.get(_read(Path(spec['source'])),spec)
                hit=ref['contact']['time_s']
                # Track through the first control boundary after real contact,
                # using the isolated body's continuation to avoid ball impulses.
                stop=int(np.ceil(hit/lab.DT))
                cut=stop-N
                tape.setup(a,c,old,expected)
                enc=[];times=[]
                for u in ref['commands'][:cut]:
                    times.append(float(a.data.time))
                    enc.append(a.joints.copy())
                    a.exchange(u,tt90.BASE.FAR_COMMAND)
                b.read(a)
                times.append(float(a.data.time))
                enc.append(a.joints.copy())
                q0,v0=lab.encoder_at(times,enc,times[-1])
                state=np.r_[q0,v0,b.state[10:]]
                witness=np.asarray(ref['commands'][cut:stop])[:,:5]
                teacher=np.stack([b.eval(witness[None,:t])[0] for t in WINDOW])
                actual, raw=ref['placements']['delivered'][0], teacher[-2]
                n=raw[13:16]
                uv=lab.coords(n,lab.tangent(n,np.asarray(actual['near_contact']['location'])-raw[10:13]))
                full=np.asarray([body_source(row,uv) for row in teacher])
                normal=full.copy()
                for u in normal:
                    u[6:]-=lab.tangent(u[3:6],u[6:])
                samples=[]
                for kind,target,native in (('normal',normal,False),('full',full,False),('full_refined',full,True)):
                    commands,meta=optimize(model,b,state,target,uv,native=native)
                    reached=np.asarray([body_source(b.eval(commands[None,:t])[0],uv) for t in WINDOW])
                    extension=np.c_[commands,np.zeros(len(commands))]
                    extension=np.vstack([extension,extension[-1][None,:]*np.array([.40,.20,.08])[:,None]])
                    response=tape.matched_motion(a,c,old,expected,ref['commands'][:cut]+extension.tolist()+[[0.]*6]*155,clear)
                    if not response['same'] or not lab.inv.tt72.first_ball(response,actual):
                        raise ValueError('An independent motion program changed the initial incoming')
                    if q.received(response):
                        verify=tape.matched_motion(a,c,old,expected,response['commands'],clear)
                        if not q.received(verify) or digest_json(verify['events'])!=digest_json(response['events']):
                            raise ValueError('Recovered movement did not replay exactly')
                        response['repeat_sha256']=digest_json(verify['events'])
                    response.update(condition=kind,body_monitor=reached.tolist(),target=target.tolist(),
                                    optimizer=meta,body_error_rms=np.sqrt(np.mean((reached-target)**2,axis=0)).tolist())
                    samples.append(response)
                    print('TT95 tube',c['id'],kind,response['landed'],response['answered'],
                          None if response['contact'] is None else response['contact']['radial_m'],response['depart'],flush=True)
                result['rows'].append(dict(case=spec,cut=cut,stop=stop,patch=uv.tolist(),variants=samples))
                result['complete']=len(result['rows'])==len(anchors)
                _write(OUT,result)
        finally:
            a.close();cad.close()


if __name__=='__main__':
    run()
