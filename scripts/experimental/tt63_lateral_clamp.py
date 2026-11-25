"""Slide the complete regulation end clamp to test the world-centre low hole."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt63_fixtures as fixture
from . import tt63_fixture_qualification as coach
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent_audit import guarded, verify_clearance
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt41_shallow_model import Chart


ROOT=fixture.ROOT/'lateral'
PLAN=ROOT/'protocol.json'
GEOMETRY=ROOT/'geometry.json'
DEVELOPMENT=ROOT/'development.json.gz'
HOLD=ROOT/'assessment.json.gz'
GAMES=ROOT/'games.json.gz'
OFFSETS=(0.,-.12,.12,-.18,.18)
BASE=fixture.mpc
X=(-.22,-.12,-.06,0.,.06,.12,.22)
Y=(-1.28,-1.40,-1.52,-1.60)
Z=(.973,.994,1.016)


def freeze():
    old=fixture.check(); ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema='tt63-lateral-end-clamp-1',main_protocol=_hash(fixture.PLAN),offsets_m=OFFSETS,
                     geometry=dict(x=X,y=Y,z=Z,interior_radius_m=.058,minimum_regular_fraction=.85),
                     development=old['development'],held=old['held'],ready=old['ready'],game_indexes=old['game_indexes'],
                     cap=old['cap'],source=old['checks'],game_reply_seed=636221,
                     rule='Rigid x-only shift of the complete existing bolted clamp and body. Same table, six joints, paddle, limits, launch, opponent and referee. Select one alternate from both directions by explicit central-low interior and regular atlas coverage, then apply the identical finite diagnostic coach. Require 2 new clean interior responses in development and >=85% normal sites to open fixed matched holds. Full games need two independent held interior responses and all four routine positives. Actual full games receive only public observations.'))


def check():
    d=_read(PLAN)
    if d['main_protocol']!=_hash(fixture.PLAN) or any(_hash(Path(p))!=v for p,v in d['source'].items()):
        raise ValueError('Changed mature TT63 or historical data')
    return d


@contextmanager
def slid(dx):
    if dx not in OFFSETS:
        raise ValueError('Unregistered physical lateral placement')
    if dx==0.:
        yield
        return
    old=fixture.TT20Arena._model_document
    def document(self):
        d=old(self)
        if self.fixture!=fixture.TABLE_EDGE or abs(dx)+.085>=self.plant.table_half_width:
            raise ValueError('Whole original clamp must fit along the regulation end')
        for label in ('learner_base','tt20_table_end_clamp'):
            b=d.find(f"worldbody/body[@name='{label}']")
            p=np.fromstring(b.get('pos','0 0 0'),sep=' ')
            b.set('pos',fixture._vector(p+[dx,0.,0.]))
        return d
    with patch.object(fixture.TT20Arena,'_model_document',document):
        yield


def low_atlas(a, q):
    chart=Chart(a,q,q[:5]); result=[]
    if not chart.guard.clear(q[:5]):
        raise ValueError('Unsafe body-relative joint ready')
    for x in X:
        for y in Y:
            for z in Z:
                p=np.array([x,y,z]); v=np.array([-.18*x,-3.95,.47])
                if y<=-1.52:
                    v[2]=-.44
                got=[]
                for pace,pitch in ((4.25,.07),(3.7,.18),(4.63,-.03)):
                    for a0 in chart.poses_at(p,v,pace,pitch,q[:5],quick=False):
                        if .011<=a0['distance']<=.037 and chart.guard.route(q[:5],a0['q'],steps=7):
                            got.append(a0)
                r=min(got,key=lambda b:b['tangent'],default=None)
                result.append(dict(p=p.tolist(),radius_m=None if r is None else float(r['tangent']),
                              travel_rad=None if r is None else r['travel'],signed_face=None if r is None else r['sign']))
    return result


def geometry():
    p=check(); ready=np.array(p['ready']); source=_read(fixture.grip.PREPARED)['rows']; base=None; goal=None
    output=dict(protocol=_hash(PLAN),operator=_hash(Path(__file__)),rows=[],complete=False)
    for dx in OFFSETS:
        with slid(dx),BASE.centered(ready):
            a=BASE.arena(width=64,height=48,detail='cad')
            try:
                ident=fixture.identity(a.model,'stock')
                if base is None:
                    base=ident; goal=np.r_[*fixture.Faces(a).pose(ready[:5])]
                if any(not np.array_equal(v,ident[k]) for k,v in base.items()):
                    raise ValueError('Clamp translation unexpectedly changes non-fixture properties')
                s=fixture.support_check(a,'stock')
                entry=dict(dx=dx,support=s)
                if not s['safe']:
                    raise ValueError('Translated clamp no longer seats on actual regulation table')
                chosen=None if dx==0. else next((r for alpha in (0.,.40,.78)
                    if (r:=fixture.calibrate(a,ready,goal+np.array([dx*alpha,0.,0.,0.,0.,0.]))) is not None),None)
                q=np.array(chosen['ready']) if chosen else ready
                original=np.r_[*fixture.Faces(a).pose(q[:5])]
                work=fixture.grip.scan_arena(a,q,[dict(r,prepared=q[:5].tolist()) for r in source] if dx else source)
                atlas=low_atlas(a,q)
                n=lambda pred:sum(r['radius_m'] is not None and r['radius_m']<=coach.INTERIOR and pred(r) for r in atlas)
                entry.update(ready=q.tolist(),commission=chosen,actual_center=original[:3].tolist(),actual_axis=original[3:].tolist(),
                             central=n(lambda r:abs(r['p'][0])<=.061),all_interior=n(lambda r:True),atlas=atlas,**work)
                output['rows'].append(entry); _write(GEOMETRY,output)
                print(dx,entry['ordinary_covered'],entry['central'],entry['all_interior'],entry['actual_center'],flush=True)
            finally:a.close()
    output['complete']=True;_write(GEOMETRY,output)


def decision(data):
    stock=data['rows'][0]
    minimum=stock['ordinary_covered']*.85
    valid=[g for g in data['rows'][1:] if g['ordinary_covered']>=minimum]
    variants=[0.]
    if valid:
        chosen=max(valid,key=lambda g:(g['central'],g['all_interior'],g['ordinary_covered'],-abs(g['dx'])))
        # A changed mechanism needs a genuine new spatial interior domain.
        if chosen['central']>stock['central']:
            variants.append(chosen['dx'])
    return variants


def paths(mode,n):
    p=check(); g=_read(GEOMETRY)
    if (not g['complete'] or g['protocol']!=_hash(PLAN) or g['operator']!=_hash(Path(__file__))):
        raise ValueError('Complete and freeze lateral geometry before physical forks')
    geometry={v['dx']:v for v in g['rows']}; variants=decision(g)
    out=DEVELOPMENT if mode=='development' else HOLD
    if len(variants)<2:
        _write(out,dict(schema='tt63-lateral-forks-1',status='no alternate with both improved center atlas and regular coverage',
                        geometry=_hash(GEOMETRY),protocol=_hash(PLAN),complete=True,rows=[]))
        return
    if mode=='assessment':
        d=_read(DEVELOPMENT)
        if not d['complete'] or d['geometry']!=_hash(GEOMETRY) or sum(r['dx']!=0 and r['interior'] for r in d['rows'])<2:
            raise ValueError('No developed lateral installation with two actual clean interior returns')
    key='development' if mode=='development' else 'held'
    roster=[(r,dx) for r in p[key] for dx in variants]; cache={}
    log=dict(schema='tt63-lateral-forks-1',operator=_hash(Path(__file__)),geometry=_hash(GEOMETRY),protocol=_hash(PLAN),
             variants=variants,mode=mode,complete=False,rows=[])
    if out.exists():
        previous=_read(out)
        if any(previous[k]!=v for k,v in log.items() if k not in ('rows','complete')):
            raise ValueError('Changed physical lateral docket')
        log=previous
    with fixture.school.enabled(cameras=False):
        for spec,dx in roster[len(log['rows']):n]:
            f=spec['source']; doc=cache[f] if f in cache else cache.setdefault(f,_read(Path(f)))
            q=geometry[dx]['ready']
            with slid(dx),BASE.centered(q):
                a=BASE.arena(width=64,height=48); cad=BASE.arena(width=64,height=48,detail='cad')
                try:
                    r=coach.score(a,coach.GripClearance(cad.model,stock=True),doc,spec,'stock' if dx==0 else 'lateral',3)
                finally:a.close();cad.close()
            interior=any(coach.received(t,True) for t in r['physical']); real=any(coach.received(t) for t in r['physical'])
            log['rows'].append(dict(dx=dx,case=spec,interior=interior,real=real,motion=r));log['complete']=len(log['rows'])==len(roster)
            _write(out,log)
            print(mode,spec['id'],dx,r['matched_prefix'],interior,real,
                  [(s['legal'],s['interior']) for s in r.get('native',[])],flush=True)


class BodyRelativeRaw:
    def __init__(self, raw, dx):
        self.raw=raw; self.dx=dx

    def get(self,x,q):
        local=np.asarray(x).copy(); local[0]-=self.dx
        return self.raw.get(local,q)


def games(n):
    d=check(); g=_read(GEOMETRY); held=_read(HOLD)
    variants=decision(g); geom={v['dx']:v for v in g['rows']}
    if len(variants)!=2 or not held['complete']:
       raise ValueError('Incomplete isolated body generalization')
    test=[r for r in held['rows'] if r['dx']==variants[1]]
    if (sum(r['case']['role']=='low' and r['interior'] for r in test)<2
           or sum(r['case']['role']=='routine' and (r['real'] or coach.received(r['motion']['stock_commands'])) for r in test)!=4):
        raise ValueError('Actual held events do not qualify for new full games')
    benchmark=_read(fixture.RESERVED); season=fixture.season; school=fixture.school
    rng=np.random.default_rng(d['game_reply_seed']); indices=d['game_indexes']; cap=d['cap']
    cards=[dict(benchmark['cards'][i],reply_seed=int(k)) for i,k in zip(indices,rng.integers(1,2**30,len(indices)),strict=True)]
    log=dict(protocol=_hash(PLAN),held=_hash(HOLD),cards=cards,variants=variants,cap=cap,rows=[],complete=False)
    if GAMES.exists():
        old=_read(GAMES)
        if any(old[k]!=v for k,v in log.items() if k not in ('rows','complete')):
            raise ValueError('Changed qualification comparators')
        log=old
    with np.load(BASE.BANK) as f: raw=BASE.DeepField(f,'local')
    coverage=season.fresh.shot.Coverage(_read(season.fresh.RISK)); held_ids={c['id'] for c in cards}
    roster=[(i,c,dx) for i,c in zip(indices,cards,strict=True) for dx in variants]
    with school.enabled():
        for i,card,dx in roster[len(log['rows']):n]:
            ready=np.array(geom[dx]['ready']); adapted=BodyRelativeRaw(raw,dx) if dx else raw
            with slid(dx),BASE.centered(ready):
                a=BASE.arena(); cad=BASE.arena(detail='cad'); camera,rig,stride,lo,hi=BASE.resources(a)
                with patch.object(BASE,'Fragments',lambda b,e:Carried(b,e,practice=[school.course.PRACTICE],held=held_ids)):
                    actor=season.Player(BASE.BANK,adapted,variant=benchmark['variant'],budget=benchmark['budget'],coverage=coverage)
                try:
                    school.course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),_read(season.CARRY))
                    guarded(a,coach.GripClearance(cad.model,stock=True))
                    r=BASE.play(a,camera,rig,stride,lo,hi,ready,card,benchmark['rows'][i]['shot'],adapted,actor,school.course.OPTIC,'frozen',cap)
                    verify_clearance(a)
                    if a.launches!=1 or r['memory']['before']!=r['memory']['after'] or any(v['target']!=list(draw(card,j,'random',None)[0]) for j,v in enumerate(r['placements']['requested'])):
                        raise ValueError('Invalid game training/identity boundary')
                    r['clearance']=a.clearance_result()
                finally:actor.close(); camera.close();cad.close();a.close()
            log['rows'].append(dict(index=i,dx=dx,result=r)); log['complete']=len(log['rows'])==len(roster)
            _write(GAMES,log)
            print('game',i,dx,r['score']['learner'],r['clearance']['penetration_samples'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('plan','geometry','development','assessment','games'));ap.add_argument('--n',type=int,default=64)
    a=ap.parse_args()
    if a.mode=='plan':freeze()
    elif a.mode=='geometry':geometry()
    elif a.mode=='games':games(a.n)
    else:paths(a.mode,a.n)
