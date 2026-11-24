"""Local clearance-constrained repairs of lower long-preparation coach strokes."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_motor_clinic import landing_error
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt38_motorcoach import AIM, answerable, schedules
from .tt40_long_low import OUT, physical_finish
from .tt40_low_feasibility import Faces, SOURCE

NOISE=np.r_[[.018,.023,.025,.025,.040],[.36,.32,.34,.36,.41],[.009,.032,.039]]


def reopen(a,c,known,upto,clear=None):
    a.launch(known['shot'],card=c)
    a.partner.reset(seed=c['seed'])
    if clear is not None:
        guarded(a,clear)
    else:
        a.sample_clearance(False)
    for x in known['actions'][:upto]:
        a.exchange(x,FAR_COMMAND)
    expected=[e for e in known['events'] if e['time']<=float(a.data.time)+1e-8]
    if digest_json(a.events)!=digest_json(expected):
        raise ValueError('Original long-preparation command prefix changed')
    return list(known['actions'][:upto])


def search(a,engine,known,batch,rounds,margin,seed):
    w=known['chosen']; goal=w['variant']
    base=np.asarray(w['parameters'])
    q=np.asarray(goal['q']); dq=np.asarray(goal['dq']); eta=goal['eta']
    ik=Faces(a)
    _,_,linear,normal=ik.pose(np.clip(q+base[:5],ik.low,ik.high),jacobian=True)
    J=np.r_[linear,.075*normal,.012*np.eye(5)]
    lift=np.linalg.lstsq(J,np.r_[[0,0,1.], np.zeros(8)],rcond=None)[0]
    rng=np.random.default_rng(seed)
    v=(2*qmc.Sobol(len(NOISE),scramble=True,seed=int(rng.integers(1,2**31))).random_base2(
        int(np.ceil(np.log2(batch))))[:batch]-1)*NOISE
    bump=np.resize(np.array([.0015,.0035,.0055,.009,.013,.020]),batch)
    v[:18]=0.
    v[:,:5]+=bump[:,None]*lift
    v[:18,10]=np.tile([-.006,0,.006],6)
    scale=a.control_dt
    n=round((eta+1.16)/scale)
    archive=[]; found=[]; trace=[]
    try:
        a.clearance_margin=margin # Only read by isolated candidate rollouts.
        for k in range(rounds):
            params=base+v
            targets=schedules(np.repeat(q[None],batch,axis=0),np.repeat(dq[None],batch,axis=0),
                            np.full(batch,eta),params,scale,n)
            f=np.zeros((batch,9))
            rate,y=engine.trajectories(targets,AIM,max(.225,eta+.13),feedback=f)
            legal=np.isfinite(y[:,0])
            score=np.where(legal,answerable(y)-12.,np.where(f[:,1]>.5,landing_error(f)+2*f[:,8],18.+36*f[:,0]))
            for j in np.flatnonzero(legal):
                end=min(rate.shape[1],max(1,int(np.ceil((y[j,1]+.12)/scale))))
                found.append(dict(tape=rate[j,:end].tolist(),cost=float(answerable(y[j])),
                    outcome=y[j].tolist(),feedback=f[j].tolist(),variant=goal,parameters=params[j].tolist()))
            trace.append(dict(round=k,tested=batch,legal=int(legal.sum()),touched=int(f[:,1].sum()),margin=margin))
            if len(found)>=4:
                break
            archive.extend(zip(score,v.copy(),strict=True))
            top=sorted(archive,key=lambda p:p[0])[:20]
            chosen=rng.integers(len(top),size=batch)
            v=np.asarray([top[r][1] for r in chosen])+rng.normal(size=v.shape)*NOISE*.57
            v=np.clip(v,-3.5*NOISE,3.5*NOISE)
            v[:len(top)]=np.array([x[1] for x in top])
    finally:
        a.clearance_margin=0.
    distinct=[]
    for r in sorted(found,key=lambda z:z['cost']):
        if any(abs(r['outcome'][1]-x['outcome'][1])<.012 and
               np.linalg.norm(np.asarray(r['feedback'][2:5])-x['feedback'][2:5])<.14 for x in distinct):
            continue
        distinct.append(r)
        if len(distinct)>=5:
            break
    return distinct,trace


def run(path,output,batch,rounds):
    d=_read(path); source=_read(SOURCE)
    if d['source_sha256']!=_hash(SOURCE):
        raise ValueError('Use the fixed matched development source')
    out=dict(schema='tt40-long-low-clean-1', source_sha256=_hash(SOURCE), bank_sha256=_hash(path),
              ready=d['ready'],rows=[],role='privileged full-CAD follow-up on the same preparation')
    with centered(d['ready']):
        a=arena(protocol='random')
        cad=arena(protocol='random',detail='cad')
        clear=CourtClearance(cad.model,articulated=True)
        engine=Rollout(a,1)
        try:
            for case in d['rows']:
                clock=process_time()
                c=case['card']; old=source['rows'][case['index']]; late=case['late_tick']
                known=case['wins'][0]
                item={k:case[k] for k in ('id','leg','height','card','fork_tick','late_tick','preparation_s')}
                item.update(search=[],trials=[],wins=[])
                out['rows'].append(item)
                for margin in (.0015,.003):
                    reopen(a,c,known,late)
                    chosen,log=search(a,engine,known,batch,rounds,margin,[400951,c['seed'],round(margin*1e6)])
                    item['search'].extend(log)
                    for plan in chosen[:3]:
                        native=[]
                        with contact_observer(a,native):
                            prefix=reopen(a,c,known,late,clear)
                            answer=physical_finish(a,old,c,case['leg'],prefix,plan)
                        verify_clearance(a)
                        shape=a.clearance_result()
                        item['trials'].append(dict(legal=answer['legal'],answered=answer['answered'],
                                                   margin=margin,clearance=shape))
                        if answer['answered'] and shape['penetration_samples']==0:
                            answer['fork_tick']=case['fork_tick']
                            item['wins'].append(answer)
                            break
                    if item['wins']:
                        break
                item['cpu_s']=process_time()-clock
                _write(output,out)
                print(case['id'],item['search'],item['trials'],bool(item['wins']),flush=True)
        finally:
            a.close();cad.close()
    out['complete']=True
    _write(output,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=OUT/'long-low-lower.json.gz')
    p.add_argument('--output',type=Path,default=OUT/'long-low-clean.json.gz')
    p.add_argument('--batch',type=int,default=160)
    p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args()
    run(a.source,a.output,a.batch,a.rounds)
