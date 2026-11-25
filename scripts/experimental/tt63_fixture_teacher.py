"""Does an altered training body teach useful movements after restoring the original?"""

import argparse
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import tt63_fixtures as fixture
from . import tt63_fixture_qualification as qualifier
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import desired_shot
from .tt41_shallow_clamp import pitched


BASE=fixture.mpc
ROOT=fixture.ROOT/'teacher'
SOURCES=ROOT/'lesson-tapes.json.gz'
PLAN=ROOT/'protocol.json'
BANK=ROOT/'lessons.json'
OUT=ROOT/'original-body.json.gz'
OLD=Path('artifacts/tt41')
CAMPAIGNS=('stock','alternate')
CASE_IDS=(11667,8595,8831,6903)
DEPTHS=(-1.06,-1.14,-1.24,-1.33,-1.43,-1.53,-1.63,-1.72)


def freeze():
    ROOT.mkdir(exist_ok=True,parents=True); source={}; by={}
    old=fixture.check(); cases=[r for r in old['development'] if r['id'] in CASE_IDS]
    for label in CAMPAIGNS:
        p=OLD/f'shallow-clamp-frozen-{label}.json.gz'; d=_read(p)
        if d['adapter']['degrees']!=(0. if label=='stock' else 6.):
            raise ValueError('Wrong fixed TT41 alternate fixture')
        options={}
        for c,r in zip(d['cards'],d['rows']):
            if r['clearance']['penetration_samples'] or c['id'] in CASE_IDS:
                continue
            selected=[]
            for j,o in enumerate(r['placements']['delivered'][:-1]):
                if not o['near_landed'] or not o['near_contact'] or not o['receive_line']:
                    continue
                h=fixture.projected_height(o['receive_line'])
                if .979<=h<=1.060:
                    selected.append((j,h))
            if selected:
                options[c['id']]=(c,r,selected)
        by[label]=options
        source[label]=dict(path=str(p),sha256=_hash(p),ready=d['ready'],degrees=d['adapter']['degrees'],rows=[])
    common=sorted(set(by['stock'])&set(by['alternate']))
    for ident in common:
        k=min(4,*[len(by[v][ident][2]) for v in CAMPAIGNS])
        for label in CAMPAIGNS:
            c,r,options=by[label][ident]
            chosen=sorted(options,key=lambda x:(abs(x[1]-1.022),x[0]))[:k]
            source[label]['rows'].append(dict(card=c,shot=r['shot'],commands=r['actions'],events=r['events'],
                                              lessons=[dict(leg=j,height=h,offer=r['placements']['delivered'][j]) for j,h in chosen]))
    if len(common)<2 or sum(len(r['lessons']) for r in source['stock']['rows'])<5:
        raise ValueError('Not enough release-paired source-body training examples')
    _write(SOURCES,source)
    _write(PLAN,dict(schema='tt63-other-body-as-teacher-1',fixture_protocol=_hash(fixture.PLAN),lesson_tapes_sha256=_hash(SOURCES),
                     cases=cases,parent={str(p):old['checks'][str(p)] for p in (fixture.DEVELOPMENT,)},
                     domains=common,depths=DEPTHS,candidates=qualifier.STAGE_TRIALS,rounds=qualifier.ROUNDS,
                     role='Restored original stock bench on every trial. Alternative teachers supply equal counts of physically landed CAD-clean ordinarily answered prior strokes from identical release IDs. Training-only transfer uses relative attained blade alignment, placement, and motion, not the altered body or its exact joint commands. Candidate trials face the unchanged real incoming ball and natural opponent. This is a training geometry intervention, not an RGB learner activation.' ))
    print('paired source lessons',common,[(n,sum(len(r['lessons']) for r in d['rows'])) for n,d in source.items()],flush=True)


def fit():
    d=_read(PLAN); sources=_read(SOURCES)
    if d['lesson_tapes_sha256']!=_hash(SOURCES):
        raise ValueError('Changed physically paid source strokes')
    out=dict(protocol=_hash(PLAN),bank={})
    for label,src in sources.items():
        row=[]
        with BASE.centered(src['ready']),pitched(src['degrees']):
            a=BASE.arena(width=64,height=48)
            try:
                g=a.model.geom('learner_racket').id
                for x in src['rows']:
                    c=x['card']; a.launch(x['shot'],card=c); a.partner.reset(seed=c['seed'])
                    lookup={round(r['offer']['near_contact']['time']*480):r for r in x['lessons']}
                    original=a._racket_contact
                    def note(side,point,vel):
                        k=round(a.data.time*480)
                        if side==1 and k in lookup:
                            l=lookup.pop(k); p=a.ball.copy(); f=a.data.geom_xmat[g].reshape(3,3)[:,2].copy(); dist=p-a.data.geom_xpos[g]
                            if f@dist<0:f=-f
                            twist=np.zeros(6);mujoco.mj_objectVelocity(a.model,a.data,mujoco.mjtObj.mjOBJ_GEOM,g,twist,0)
                            row.append(dict(id=c['id'],leg=l['leg'],height=l['height'],p=p.tolist(),v=np.asarray(vel).tolist(),
                                            normal=f.tolist(),offset=dist.tolist(),blade_velocity=twist[3:].tolist(),q=a.joints[:5].tolist(),
                                            geometry_time_s=float(a.data.time)))
                        return original(side,point,vel)
                    with patch.object(a,'_racket_contact',note):
                        for u in x['commands']:a.exchange(u,BASE.FAR_COMMAND)
                    if lookup or digest_json(a.events)!=digest_json(x['events']):
                        raise ValueError('Rerun of physically paid source lesson differs')
            finally:a.close()
        out['bank'][label]=row
    if len(out['bank']['stock'])!=len(out['bank']['alternate']):
        raise ValueError('Inconsistent source teacher dose')
    _write(BANK,out);print('physically reproduced source teachers',[(n,len(r)) for n,r in out['bank'].items()],flush=True)


def transform(from_,to):
    axis=np.cross(from_,to); s=float(np.linalg.norm(axis)); angle=np.arctan2(s,np.clip(from_@to,-1.,1.))
    return Rotation.from_rotvec(axis*angle/max(s,1e-8))


def offers(engine, lessons):
    a=engine.a; ik=engine.ik; ready=a.joints[:5]; result=[]
    arc=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=.88)
    if not len(arc):return []
    for depth in DEPTHS:
        if not arc[:,2].min()-.01<depth<arc[:,2].max()+.01:continue
        time,p,v=np.split(arc[np.argmin(abs(arc[:,2]-depth)),:7],[1,4]); eta=float(time[0]);
        if not .11<eta<.64 or p[2]<.81:continue
        near=sorted(lessons,key=lambda l:np.linalg.norm((p-np.array(l['p']))/[.18,.32,.08])+
                    .55*np.linalg.norm((v-np.array(l['v']))/[.62,.88,1.05]))[:3]
        want=desired_shot(p,v,1,[0.,.70],4.2)
        for l in near:
            old=desired_shot(np.asarray(l['p']),np.asarray(l['v']),1,[0.,.70],4.2)
            R=transform(old,want); n=R.apply(l['normal']); d=R.apply(l['offset']); velocity=R.apply(l['blade_velocity']);
            for shrink in (.58,.86,1.0):
                c=p-((d@n)*n+shrink*(d-(d@n)*n))
                for face in (-1,1):
                    q,(b,m,j,jn)=ik.fit(c,face*n,[*l['q']]); m*=face
                    if (np.linalg.norm(b-c)>.047 or m@n<.79 or np.max(abs(q-ready))>4.15*eta+.15):continue
                    offset=p-b; radius=np.linalg.norm(offset-(offset@m)*m)
                    if radius>.078 or not .002<offset@m<.052:continue
                    if any(k['face']==face and abs(k['depth']-depth)<.01 and np.linalg.norm(k['q']-q)<.11 for k in result):continue
                    dq=np.linalg.lstsq(np.r_[j,.069*jn,.023*np.eye(5)],np.r_[velocity,np.zeros(8)],rcond=None)[0].clip(-5.2,5.2)
                    result.append(dict(face=face,depth=depth,time=eta,q=q,dq=dq,radius=float(radius),donor=l['id']))
    return sorted(result,key=lambda r:r['radius'])[:32]


def run(n):
    p=_read(PLAN); bank=_read(BANK)
    if bank['protocol']!=_hash(PLAN) or any(_hash(Path(p0))!=v for p0,v in p['parent'].items()):
        raise ValueError('Changed other-body curriculum')
    old=_read(fixture.DEVELOPMENT); plan=fixture.check()
    receipt=dict(schema='tt63-restored-body-1',protocol=_hash(PLAN),bank=_hash(BANK),operator=_hash(Path(__file__)),rows=[],complete=False)
    if OUT.exists():
        r=_read(OUT)
        if any(r[k]!=v for k,v in receipt.items() if k not in ('rows','complete')):raise ValueError('Changed curriculum experiment')
        receipt=r
    guard=None
    with fixture.school.enabled(cameras=False),BASE.centered(plan['ready']):
        a=BASE.arena(width=64,height=48); cad=BASE.arena(width=64,height=48,detail='cad')
        try:
            guard=qualifier.GripClearance(cad.model,stock=True)
            for spec in p['cases'][len(receipt['rows']):n]:
                c,row,expected=qualifier.get(old,spec); out={}
                far=next(e for e in row['events'] if e['kind']=='separation' and e.get('collider')=='rival_racket')
                k=int(np.ceil(far['time']/BASE.DT+1e-7))+1; prefix=[[0.]*6]*k
                for label,data in bank['bank'].items():
                    qualifier.tape.setup(a,c,row,expected)
                    for u in prefix:a.exchange(u,BASE.FAR_COMMAND)
                    if digest_json(a.events)!=digest_json([e for e in row['events'] if e['time']<=a.data.time+1e-8]):
                        raise ValueError('Restored original fixture changed the physical starting prefix')
                    engine=qualifier.low.Probe(a); bases=offers(engine,data)
                    choices,meta=qualifier.search(engine,[636614,c['seed']],bases)
                    seen=[]
                    for u,v in choices[:3]:
                        r=qualifier.tape.matched_motion(a,c,row,expected,prefix+u+[[0.]*6]*140,guard);r['forecast']=v;seen.append(r)
                        if qualifier.received(r,True):break
                    out[label]=dict(bases=len(bases),source_ids=sorted({r['donor'] for r in bases}),model=meta,physical=seen)
                receipt['rows'].append(dict(case=spec,teacher=out));receipt['complete']=len(receipt['rows'])==len(p['cases']);_write(OUT,receipt)
                print(spec['id'],[(n,x['bases'],[v['legal'] for v in x['model']],sum(qualifier.received(v) for v in x['physical'])) for n,x in out.items()],flush=True)
        finally:a.close();cad.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','fit','run'));a.add_argument('--n',type=int,default=4)
    p=a.parse_args()
    if p.mode=='plan':freeze()
    elif p.mode=='fit':fit()
    else:run(p.n)
