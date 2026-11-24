"""Learn a shallow rising stroke bank from scored physical trials, then freeze it."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash,_read,_write
from .tt29_endurance import audit
from .tt29_rally_physics import Rollout, TABLE_HEIGHT
from .tt25_reference_receiver import ReferenceReceiver
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt38_motorcoach import AIM,SIZES,answerable
from .tt41_shallow_model import Chart,contact_preference,goals


ROOT=Path(__file__).resolve().parents[2]/'artifacts'
OUT=ROOT/'tt41'
WITNESS=ROOT/'tt40/long-low.json.gz'
COURSE=ROOT/'tt37/opponent-plan.json'


def make_plan(out):
    sampler=qmc.Sobol(8,scramble=True,seed=413081).random_base2(5)
    lows=np.array([-.27,-.73,-1.15,-4.9,-2.75,.185,-29.,-24.])
    highs=np.array([.27,-.34,1.15,-3.15,-1.95,.265,29.,24.])
    practice=[]
    for i,(x,y,vx,vy,vz,delay,wx,wz) in enumerate(qmc.scale(sampler,lows,highs)):
        # Physics is never assigned mid-point. Initial position is a released
        # free ball with one intended near bounce. Native drag and bounce decide
        # the actual arrival, including misses and incomplete deliveries.
        pos=np.array([x-vx*delay,y-vy*delay,.783-vz*delay-4.905*delay**2])
        vel=np.array([vx,vy,vz+9.81*delay]); spin=np.array([wx,0.,wz])
        shot=dict(id=410000+i,**{'pos_'+a:float(v) for a,v in zip('xyz',pos,strict=True)},
                  **{'vel_'+a:float(v) for a,v in zip('xyz',vel,strict=True)},
                  **{'w_vel_'+a:float(v) for a,v in zip('xyz',spin,strict=True)})
        shot['pos_z']-=TABLE_HEIGHT
        practice.append(dict(shot=shot,nominal_bounce=[float(x),float(y)],train=i<24))
    d=_read(COURSE)
    roster=np.random.default_rng(418711).choice(len(d['held_openings']),size=6,replace=False)
    cards=[d['held_openings'][int(i)]|dict(reply_seed=4158409+103*j,key=f'shallow-held-{j}') for j,i in enumerate(roster)]
    _write(out,dict(practice=practice,held=cards,design_sha256=_hash(COURSE),
                    recipe='Fixed Sobol continuous incoming release design; initial geometry and mode chosen before practice outcomes. Six measured independent opponent openings preassigned before training.'))


def atlas(out):
    old=_read(WITNESS); home=np.asarray(old['ready'])
    r=next(r for r in old['rows'] if r['id']==14843)
    seed=np.asarray(next(w for w in r['wins'] if w['answered'])['chosen']['variant']['q'])
    with centered(home):
        a=arena(detail='cad')
        try:
            chart=Chart(a,home,seed)
            result=chart.assess()
        finally:
            a.close()
    result.update(source_sha256=_hash(WITNESS),use_of_witness='One static search seed only, no target or time sequence copied')
    _write(out,result)
    print('geometry',result['signs'],'hub',result['hubs'][0],flush=True)


def launch(a,shot):
    # Deliberately designated teacher release, not an assessment game opener.
    ReferenceReceiver.launch(a,shot)
    a.card=dict(id=shot['id'],index=shot['id'],seed=416027+shot['id'],reply_seed=419010+315*shot['id'],
                mode='full',opening='random',profile='reach',key=f'shallow-practice-{shot["id"]}')
    a.crossings={};a.before_step=None; a._last_hitter=-1; a._received_bounce=False
    a.launches+=1
    a.partner.reset(seed=a.card['seed'])


def touch_band(event):
    x,y,z=event['location']
    return -1.43<y<-1.045 and .818<z<1.10 and event['incoming'][2]>.06


def attempts(a,engine,chart,shot,bank,*,learn):
    launch(a,shot)
    start=process_time()
    path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=False,horizon=.78)
    bases=chart.windows(path,a.joints[:5])
    found=[]; history=[]; rollout_notes=[]
    if bases:
        rng=np.random.default_rng([410971,int(shot['id'])])
        if learn:
            count=96
            gaussian=qmc.Sobol(13,scramble=True,seed=int(rng.integers(1,2**31))).random_base2(7)[:count]
            key=np.arange(count)%len(bases)
            params=(2*gaussian-1)*SIZES*.72
            params[:len(bases)]=0.
        else:
            from .tt41_shallow_model import FEATURE
            key=[]; params=[]
            for i,b in enumerate(bases):
                x=np.r_[b['p'],b['v']]
                pool=[s for s in bank if s['sign']==b['sign']]
                pool=sorted(pool,key=lambda z:float(np.linalg.norm((x-z['x'])/FEATURE)))[:2]
                for row in pool:
                    key.append(i); params.append(row['parameters'])
            key=np.asarray(key,dtype=int); params=np.asarray(params)
        for layer in range(3 if learn else 1):
            if not len(key):break
            moves,cutoff=goals(bases,key,params,a.control_dt)
            f=np.zeros((len(moves),9)); rates,res=engine.trajectories(moves,AIM,cutoff,feedback=f)
            rank,good,touch=contact_preference(res,f)
            by=answerable(res)
            for j in np.flatnonzero(good):
                b=bases[key[j]]
                if res[j,5]<.017 or f[j,5:8][1]<-1.45 or f[j,5:8][1]>-1.018:
                    continue
                lead=round((res[j,1]+.09)/a.control_dt)
                found.append((float(by[j]),np.array(rates[j,:lead]),b,np.array(params[j]),f[j],res[j]))
            rollout_notes.append(dict(layer=layer,trials=len(moves),physical_model_touches=int(touch.sum()),
                model_legal=int(good.sum()),acceptable=len(found),closest_m=float(f[:,0].min())))
            if not learn or (len(found)>=4 and min(row[0] for row in found)<3.3):break
            history.extend(zip(rank,key,params,strict=True))
            best=sorted(history,key=lambda x:-x[0])[:15]
            count=64
            seeds=rng.integers(len(best),size=count)
            key=np.asarray([best[i][1] for i in seeds])
            params=np.array([best[i][2] for i in seeds])+rng.normal(size=(count,13))*SIZES*(.53 if layer==0 else .36)
            params=np.clip(params,-2.2*SIZES,2.2*SIZES)
            for j,(_,idx,w) in enumerate(best[:5]):
                key[j]=idx;params[j]=w
    selected=min(found,key=lambda row:row[0]) if found else None
    commands=[]
    if selected is not None:
        plan=selected[1]
    else:
        plan=np.zeros((0,6))
    for tick in range(205):
        move=plan[tick] if tick<len(plan) else np.zeros(6)
        a.exchange(move,FAR_COMMAND)
        commands.append(move.tolist())
        if a.terminated or a.truncated or any(e['kind']=='legal_return' and e.get('side')==-1
            for e in a.events[4:] if e['time']>.78):break
    score=audit(a.events,opening_side=-1)
    contacts=[c for c in a.events if c['kind']=='contact' and c['side']==1]
    own=score['learner']>0
    first_bounce=next((r for r in a.events if r['kind']=='legal_return' and r.get('side')==-1),None)
    label=dict(delivered=first_bounce is not None,legal=own,answered=score['opponent']>0,
               shallow=bool(own and contacts and touch_band(contacts[0])),actual_contact=contacts[0] if contacts else None,
               records=rollout_notes,geometry=len(bases),events=a.events,actions=commands,motor=a.motor_audit(),cpu=process_time()-start)
    if selected is not None:
        _,_,base,change,output,grade=selected
        label['candidate']=dict(x=np.r_[base['p'],base['v']].tolist(),sign=base['sign'],pace=base['pace'],
                                pitch=base['pitch'],parameters=change.tolist(),twin_departure=output[2:5].tolist(),
                                twin_landing=grade[3:6].tolist(),contact_eta=float(grade[1]))
    return label


def worker(spec):
    calibration,rows,bank,learn,hub_index=spec
    d=_read(calibration)
    hub=np.r_[d['hubs'][hub_index]['q'],d['home'][-1]]
    with centered(hub):
        a=arena(width=192,height=144); cad=arena(detail='cad')
        try:
            engine=Rollout(a,1); chart=Chart(cad,hub,d['seeds'][0],atlas=d)
            out=[]
            for row in rows:
                r=attempts(a,engine,chart,row['shot'],bank,learn=learn)
                print(row['shot']['id'],r['delivered'],r['legal'],r['shallow'],r['answered'],r['geometry'],flush=True)
                out.append(row|r)
            return out
        finally:
            a.close();cad.close()


def train(plan,calibration,output,workers,*,frozen=None,hub_index=0):
    d=_read(plan)
    cases=[r for r in d['practice'] if r['train']==(frozen is None)]
    bank=[] if frozen is None else _read(frozen)['bank']
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as cpu:
        result=list(cpu.map(worker,[(calibration,cases[i::workers],bank,frozen is None,hub_index) for i in range(workers)]))
    rows=sorted((r for batch in result for r in batch),key=lambda r:r['shot']['id'])
    if frozen is None:
        for r in rows:
            if r['shallow'] and 'candidate' in r:
                bank.append(r['candidate']|dict(practice_release=r['shot']['id'],physical_answered=r['answered'],hub_index=hub_index))
    _write(output,dict(plan_sha256=_hash(plan),atlas_sha256=_hash(calibration),mode='physical_practice' if frozen is None else 'frozen_transfer',
                       workers=workers,hub_index=hub_index,bank=bank,rows=rows))


def combine(paths,output):
    docs=[_read(p) for p in paths]
    atlas={d['atlas_sha256'] for d in docs};plan={d['plan_sha256'] for d in docs}
    if len(atlas)!=1 or len(plan)!=1 or any(d['mode']!='physical_practice' for d in docs):
        raise ValueError('Only same-protocol rewarded practice can initialize the live skill')
    rows=[]
    for d in docs:
        hub=d.get('hub_index',0)
        rows.extend(r|dict(hub_index=hub) for r in d['bank'])
    _write(output,dict(atlas_sha256=next(iter(atlas)),plan_sha256=next(iter(plan)),bank=rows,
        sources={p.name:_hash(p) for p in paths},mode='combined_earned_shallow'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','atlas','train','probe','merge')); p.add_argument('out',type=Path)
    p.add_argument('--plan',type=Path,default=OUT/'shallow-plan.json'); p.add_argument('--atlas',type=Path,default=OUT/'shallow-atlas.json.gz')
    p.add_argument('--bank',type=Path,default=OUT/'shallow-practice.json.gz');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--hub',type=int,default=0);p.add_argument('--sources',type=Path,nargs='+')
    a=p.parse_args()
    if a.mode=='plan':make_plan(a.out)
    elif a.mode=='atlas':atlas(a.out)
    elif a.mode=='merge':combine(a.sources,a.out)
    else:train(a.plan,a.atlas,a.out,a.workers,frozen=a.bank if a.mode=='probe' else None,hub_index=a.hub)


if __name__=='__main__':main()
