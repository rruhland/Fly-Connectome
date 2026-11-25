"""Native body-mechanism qualification followed, only if earned, by RGB games."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from . import tt40_low_feasibility as low
from . import tt57_handle_rotation as tape
from . import tt63_fixtures as fixtures
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_opponent_audit import guarded, verify_clearance
from .tt38_motorcoach import AIM, SIZES, schedules
from .tt39_carry_joints import Carried
from .tt41_shallow_grip import GripClearance


mpc=fixtures.mpc
school=fixtures.school
season=fixtures.season
ROOT=fixtures.ROOT
DEVELOPMENT=ROOT/'development.json.gz'
ASSESSMENT=ROOT/'assessment.json.gz'
GO=ROOT/'selection.json'
GAMES=ROOT/'games.json.gz'
STAGE_TRIALS=144
ROUNDS=3
INTERIOR=.058


def ordered(path):
    g=_read(path)
    if (g['protocol']!=_hash(fixtures.PLAN) or g['geometry_code']!=_hash(Path(fixtures.__file__)) or not g['complete']):
        raise ValueError('Complete frozen commissioning before scoring a fixture')
    return g


def search(coach, seed, bases):
    """Symmetric equal-budget signed strokes with a physically interior preference."""
    if not bases:
        return [],[]
    rng=np.random.default_rng(seed); n=STAGE_TRIALS
    groups=[np.array([i for i,b in enumerate(bases) if b['face']==s],dtype=int) for s in (-1,1)]
    groups=[g for g in groups if len(g)]
    # Preserve distinct clocks and faces on the first pass.
    ix=np.empty(n,dtype=int)
    for k,g in enumerate(groups):
        ix[k::len(groups)]=np.resize(g,len(ix[k::len(groups)]))
    grid=qmc.Sobol(len(SIZES),scramble=True,seed=int(rng.integers(1,2**30))).random_base2(int(np.ceil(np.log2(n))))
    j=np.unique(ix,return_index=True)[1]
    pars=(2*grid[:n]-1)*SIZES*.85
    pars[j]=0.
    q=np.array([z['q'] for z in bases]); dq=np.array([z['dq'] for z in bases]); t=np.array([z['time'] for z in bases]); dt=coach.a.control_dt
    candidates=[]; notes=[]; edge=.043
    for k in range(ROUNDS):
        cutoff=float(t[ix].max()+.115)
        goal=schedules(q[ix],dq[ix],t[ix],pars,dt,int(np.ceil((cutoff+.99)/dt)))
        feedback=np.zeros((n,9)); commands,y=coach.core.trajectories(goal,AIM,cutoff,feedback=feedback)
        loss,legal=mpc.cost(y,feedback)
        contact=feedback[:,1]>.5
        margin=np.where(legal,y[:,12],feedback[:,8])
        loss+= np.where(contact,44.*np.maximum(margin-edge,0.),0.)
        # A separated legal stroke should beat even a very near miss. Within
        # either family prioritize the richer .058-m face portion.
        rank=np.lexsort((loss,~(legal&(margin<=INTERIOR)),~legal))
        for j in rank[:14]:
            if not legal[j]:
                continue
            active=int(np.ceil((y[j,1]+.10)/dt))
            info=dict(cost=float(loss[j]),radius=float(y[j,12]),feedback=feedback[j].tolist(),
                      face=bases[ix[j]]['face'],depth=float(bases[ix[j]]['depth']),round=k)
            candidates.append((commands[j,:active].tolist(),info))
        notes.append(dict(trials=n,legal=int(legal.sum()),touch=int(contact.sum()),interior=int(np.sum(legal&(margin<=INTERIOR))),
                          least_legal_radius=None if not legal.any() else float(min(margin[legal]))))
        best=rank[:24]; old=ix[best].copy(); moves=pars[best].copy()
        selected=rng.integers(len(best),size=n)
        ix=old[selected]; pars=moves[selected]+rng.normal(0.,1.,size=pars.shape)*SIZES*(.56-.12*k)
        pars=np.clip(pars,-2.7*SIZES,2.7*SIZES)
        ix[:len(old)],pars[:len(old)]=old,moves
    result=[]
    for p in sorted(candidates,key=lambda z:(z[1]['radius']>INTERIOR,z[1]['cost'])):
        cmd=np.asarray(p[0]); keep=True
        for r in result:
            k=min(len(cmd),len(r[0])); variation=np.sqrt(np.mean((cmd[:k]-np.asarray(r[0])[:k])**2))
            if variation<.035:keep=False;break
        if keep:result.append(p)
        if len(result)==6:break
    return result,notes


def get(d, spec):
    c,row=d['cards'][spec['i']],d['rows'][spec['i']]
    offer=row['placements']['delivered'][0]
    if (digest_json([offer['far_contact'],offer['near_bounce'],offer['receive_line']])!=spec['delivery']
            or c['id']!=spec['id']):
        raise ValueError('Changed frozen physical first offer')
    return c,row,dict(offer=offer,seed=c['seed'])


def received(x, interior=False):
    return (x['same'] and x['landed'] and x['answered'] and x['clearance']['penetration_samples']==0
            and x['contact'] is not None and (not interior or
                (x['contact']['radial_m'] <= INTERIOR and abs(x['contact']['face_alignment'])>=.75)))


def score(a,clear,doc,spec,config,controls):
    c,old,anchor=get(doc,spec)
    control=tape.matched_motion(a,c,old,anchor,old['actions']+[[0.]*6]*128,clear)
    if config=='stock' and not control['same']:
        raise ValueError('Original real incoming delivery no longer reproduces')
    # Both installations begin coaching after identical physical far
    # separation from a commissioned neutral body, before any near search.
    far=next(e for e in old['events'] if e['kind']=='separation' and e.get('collider')=='rival_racket')
    fork=int(np.ceil(far['time']/mpc.DT+1e-7))+1
    tape.setup(a,c,old,anchor)
    prefix=[[0.]*6]*fork
    for cmd in prefix:
        _,_,done,cap,_=a.exchange(cmd,mpc.FAR_COMMAND)
        if done or cap:break
    before=[e for e in old['events'] if e['time']<=a.data.time+1e-8]
    if a.steps!=fork or digest_json(a.events)!=digest_json(before):
        return dict(stock_commands=control,matched_prefix=False,physical=[],fork=fork,
                    prefix_events=a.events)
    coach=low.Probe(a); basis=coach.basis()
    pose=tape.read_geometry(a,coach,basis) if basis else []
    try:
        a.clearance_margin=.001
        candidates,notes=search(coach,[630662,c['seed']],basis)
    finally:a.clearance_margin=0.
    results=[]
    for motion,meta in candidates[:controls]:
        actual=tape.matched_motion(a,c,old,anchor,prefix+motion+[[0.]*6]*155,clear)
        actual['predicted']=meta
        results.append(actual)
        if received(actual,interior=True):
            check=tape.matched_motion(a,c,old,anchor,actual['commands'],clear)
            if not received(check,interior=True) or digest_json(actual['events'])!=digest_json(check['events']):
                raise ValueError('Potential interior winner did not reproduce')
            actual['repeat_sha256']=digest_json(check['events'])
            break
    return dict(stock_commands=control,matched_prefix=True,fork=fork,basis=pose,native=notes,physical=results)


def docket(mode, n):
    p=fixtures.check(); record=ordered(fixtures.GEOMETRY)
    index={r['name']:r for r in record['rows']}
    if mode=='development':
        cohort=p['development']; out=DEVELOPMENT
        possible=[r for r in record['rows'] if r.get('ordinary_covered',0)>=.85*index['stock']['ordinary_covered']]
        # Run stock and the single best mechanical alternative per family.
        def best(family):
            members=[r for r in possible if r['name'].startswith(family)]
            if not members:return []
            return [max(members,key=lambda r:(bool(r['live_eligible']),sum(r['low_improvements_m']),r['ordinary_covered']))['name']]
        variants=['stock']+best('neck')+best('low-shelf')
    else:
        selected=_read(GO)
        if selected['development_sha256']!=_hash(DEVELOPMENT) or selected['selected'] is None:
            raise ValueError('No stock or altered geometry won development prequalification')
        variants=['stock',selected['selected']]; cohort=p['held']; out=ASSESSMENT
    contract=dict(phase=mode,protocol=_hash(fixtures.PLAN),geometry=_hash(fixtures.GEOMETRY),variants=variants,
                  operator=_hash(Path(__file__)),complete=False,rows=[])
    if out.exists():
        prior=_read(out)
        if any(prior[k]!=v for k,v in contract.items() if k not in ('rows','complete')):
            raise ValueError('Changed forensic physical fixture docket')
        contract=prior
    roster=[(spec,name) for spec in cohort for name in variants]
    data={}
    with school.enabled(cameras=False):
        for s,name in roster[len(contract['rows']):n]:
            src=s['source']; doc=data[src] if src in data else data.setdefault(src,_read(Path(src)))
            geom=index[name]; ready=(geom.get('commission') or {}).get('ready',p['ready'])
            with fixtures.mounted(name),mpc.centered(ready):
                a=mpc.arena(width=64,height=48); cad=mpc.arena(width=64,height=48,detail='cad')
                try:
                    guard=GripClearance(cad.model,stock='pitch' not in fixtures.CONFIGS[name]);
                    r=score(a,guard,doc,s,name,p['physical_budget'])
                finally:a.close();cad.close()
            contract['rows'].append(dict(case=s,name=name,ready=ready,result=r))
            contract['complete']=len(contract['rows'])==len(roster)
            _write(out,contract)
            print(mode,len(contract['rows']),s['id'],name,'control',r['stock_commands']['same'],r['stock_commands']['landed'],
                  'model',[(z['touch'],z['legal'],z['interior']) for z in r.get('native',[])],
                  'actual',[(t['landed'],t['answered'],received(t,interior=True),t['clearance']['penetration_samples']) for t in r['physical']],flush=True)


def selection():
    d=_read(DEVELOPMENT); g=ordered(fixtures.GEOMETRY)
    if not d['complete'] or d['geometry']!=_hash(fixtures.GEOMETRY):
        raise ValueError('Complete all prespecified practice variants before opening assessment')
    rows={}
    for r in g['rows']:
        name=r['name']
        if name not in d['variants']:continue
        rr=[v for v in d['rows'] if v['name']==name]
        rows[name]=dict(sites=r['ordinary_covered'],geometry_eligible=r.get('live_eligible',False),
                        matched=sum(v['result']['matched_prefix'] for v in rr),
                        real=sum(any(received(z) for z in v['result']['physical']) for v in rr),
                        interior=sum(any(received(z,True) for z in v['result']['physical']) for v in rr))
    pool=[(r['interior'],r['real'],r['sites'],name) for name,r in rows.items() if name!='stock' and
          (r['geometry_eligible'] or (r['interior']>=2 and r['interior']>rows['stock']['interior']))]
    chosen=max(pool)[3] if pool else None
    _write(GO,dict(development_sha256=_hash(DEVELOPMENT),geometry=_hash(fixtures.GEOMETRY),rows=rows,selected=chosen))
    print('prequalify',rows,chosen,flush=True)


def games(n):
    p=fixtures.check(); g=ordered(fixtures.GEOMETRY); pick=_read(GO)['selected']; held=_read(ASSESSMENT)
    if not held['complete']:
        raise ValueError('Complete fixed actual incoming assessment prior to full games')
    tests=[v for v in held['rows'] if v['name']==pick]
    interior=sum(v['case']['role']=='low' and any(received(z,True) for z in v['result']['physical']) for v in tests)
    ordinary=sum(v['case']['role']=='routine' and (received(v['result']['stock_commands']) or any(received(z) for z in v['result']['physical'])) for v in tests)
    if interior<2 or ordinary<sum(v['case']['role']=='routine' for v in tests):
        raise ValueError('Mechanical qualification did not preserve routines and produce two held interior answers')
    geom={r['name']:r for r in g['rows']}; src=_read(fixtures.RESERVED)
    roster=[(i,name) for i in p['game_indexes'] for name in ('stock',pick)]
    rng=np.random.default_rng(630925); reply=rng.integers(1,2**30,size=len(p['game_indexes'])).tolist()
    cases={i:dict(src['cards'][i],reply_seed=s) for i,s in zip(p['game_indexes'],reply,strict=True)}
    contract=dict(protocol=_hash(fixtures.PLAN),choice=_hash(GO),assessment=_hash(ASSESSMENT),cap=p['cap'],cards=list(cases.values()),rows=[],complete=False)
    if GAMES.exists():
        prior=_read(GAMES)
        if any(prior[k]!=v for k,v in contract.items() if k not in ('rows','complete')):
            raise ValueError('Changed ordinary RGB game comparison')
        contract=prior
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK)); ids={v['id'] for v in cases.values()}
    with np.load(mpc.BANK) as f:raw=mpc.DeepField(f,'local')
    with school.enabled():
        for i,name in roster[len(contract['rows']):n]:
            card=cases[i]; ready=(geom[name].get('commission') or {}).get('ready',p['ready'])
            with fixtures.mounted(name),mpc.centered(ready):
                a=mpc.arena(); cad=mpc.arena(width=64,height=48,detail='cad')
                camera,rig,stride,lo,hi=mpc.resources(a)
                with patch.object(mpc,'Fragments',lambda b,e:Carried(b,e,practice=[school.course.PRACTICE],held=ids)):
                    actor=season.Player(mpc.BANK,raw,variant=src['variant'],budget=src['budget'],coverage=rank)
                try:
                    school.course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),_read(season.CARRY))
                    guarded(a,GripClearance(cad.model,stock='pitch' not in fixtures.CONFIGS[name]))
                    r=mpc.play(a,camera,rig,stride,lo,hi,np.array(ready),card,src['rows'][i]['shot'],raw,actor,
                               school.course.OPTIC,'frozen',p['cap'])
                    verify_clearance(a)
                    if r['memory']['before']!=r['memory']['after'] or a.launches!=1 or any(v['target']!=list(draw(card,j,'random',None)[0]) for j,v in enumerate(r['placements']['requested'])):
                        raise ValueError('Changed physical opponent law or training boundary')
                    r['clearance']=a.clearance_result()
                finally:actor.close();camera.close();cad.close();a.close()
            contract['rows'].append(dict(index=i,fixture=name,id=card['id'],result=r)); contract['complete']=len(contract['rows'])==len(roster)
            _write(GAMES,contract)
            print('game',i,name,r['score']['learner'],r['clearance']['penetration_samples'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('development','select','assessment','games'))
    ap.add_argument('--n',default=64,type=int)
    par=ap.parse_args()
    if par.mode=='select':selection()
    elif par.mode=='games':games(par.n)
    else:docket(par.mode,par.n)
