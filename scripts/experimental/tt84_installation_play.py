"""Continuous RGB/encoder online transfer to a mechanically qualified installation."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from . import tt84_installation_system as system
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent import draw
from .tt37_opponent_audit import contact_observer, guarded, verify_clearance
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height


ROOT=system.ROOT
PLAN=ROOT/'games.protocol.json'
SEASON=dose.season
COURSE=dose.course
MP=system.MP
TT80=Path('artifacts/tt80_continuous_games/protocol.json')
SEASON_PLAN=ROOT/'season.protocol.json'


class Registration:
    """Map an observed world ball into the old geometry for joint memory only."""
    def __init__(self, origin, position, rotation):
        self.o=np.asarray(origin); self.p=np.asarray(position); self.R=np.asarray(rotation)

    def local(self,x):
        if x is None:return None
        y=np.array(x,copy=True)
        y[:3]=(y[:3]-self.p)@self.R + self.o
        y[3:6]=y[3:6]@self.R
        return y

    def feature(self,value, scales, width=11):
        t=np.array(value,copy=True).reshape(-1,width)
        physical=t*scales
        physical[0,:6]=self.local(physical[0,:6])
        physical[1:,:3]=physical[1:,:3]@self.R
        physical[1:,3:6]=physical[1:,3:6]@self.R
        return (physical/scales).ravel()


class LocalField:
    def __init__(self,raw,register):self.raw=raw;self.register=register
    def get(self,x,q):return self.raw.get(self.register.local(x),q)


class Registered(SEASON.Player):
    def __init__(self,*args,register,**kw):
        super().__init__(*args,**kw);self.register=register
        ask=self.priors.priors;keep=self.priors.keep
        self.priors.priors=lambda x,q,r,limit:ask(register.local(x),q,r,limit)
        self.priors.keep=lambda tape,tag:keep([(t,register.local(x),q,r,s) for t,x,q,r,s in tape],tag)

    def begin(self):
        super().begin(); current=self.motor.predict
        scales=SEASON.fresh.servo.SCALES['local']
        self.motor.predict=lambda f:current(self.register.feature(f,scales))


def freeze():
    # A separate fresh natural draw roster is fixed without reading the live
    # score. Retain 8 whole games, keeping body-selection release IDs out.
    d=system.contract(); doc=_read(dose.PLAN); prior=_read(SEASON.CARRY)
    excludes={s['id'] for s in d['physical']} | {s['id'] for s in prior['cards'][:len(prior['rows'])]}
    rng=np.random.default_rng(840391); pool=[c for c in doc['practice'] if c['id'] not in excludes]; taken=[]
    used=set()
    for i in rng.permutation(len(pool)):
        c=pool[i]
        if c['id'] in used:continue
        used.add(c['id']);taken.append(dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))))
        if len(taken)==8:break
    if len(taken)!=8:raise ValueError('Not enough distinct whole-game measured starts')
    _write(PLAN,dict(protocol=_hash(system.PLAN),carry=_hash(SEASON.CARRY),source_schedule=_hash(dose.PLAN),
                     cards=taken,cap=240,rule='Independent starts and continuous random requested placements. Same carried learned state. The mounted player registers only memory coordinates using the installation survey. Both learn at real legal returns and the calibrated private twin still sees world RGB. No restart inside a point.'))


def season_plan():
    d=_read(TT80); a=_read(PLAN)
    if len(d['practice'])!=128 or d['carry']!=a['carry'] or d['carry']!=_hash(SEASON.CARRY):
        raise ValueError('The centered TT80 all-start roster or 48-game carry changed')
    _write(SEASON_PLAN,dict(protocol=_hash(system.PLAN),carry=_hash(SEASON.CARRY),tt80_sha256=_hash(TT80),
                           cards=d['practice'],cap=240,scope='Exact unfiltered centered TT80 cards and reply seeds, mounted arm only. Adapter follows the verified physical design exactly. Start anew with the same 48-game archived bank. Legal returned action fragments earn online credit. Whole-point winners crossing 100 own returns, and the final peak, get an independently replayed CAD proof.'))


def audit_witness(record,c,a,clear,kind,index,stream,ready,mount):
    a.launch(record['shot'],card=c); a.partner.reset(seed=c['seed']); guarded(a,clear); contacts=[]
    with contact_observer(a,contacts):
        for u in record['actions']:a.exchange(u,MP.FAR_COMMAND)
    verify_clearance(a)
    placed=a.point_placements(); scored=audit(a.events,opening_side=1); actual=a.clearance_result()
    for key,val in (('events',a.events),('placements',placed),('score',scored)):
        if digest_json(val)!=digest_json(record[key]):raise ValueError('Candidate earned tape changed in independent physical replay')
    if sum(e['kind']=='contact' for e in a.events)!=len(contacts):raise ValueError('Missing actual native contact')
    if any(w['draw']!='random' or w['target']!=list(draw(c,j,'random',None)[0]) for j,w in enumerate(placed['requested'])):
        raise ValueError('The candidate tape did not use the standard continuous draws')
    out=ROOT/f'{stream}-{kind}-witness-{index:03}-{c["id"]}.json.gz'
    result=dict(index=index,id=c['id'],kind=kind,stream=stream,ready=ready,mount=mount,card=c,
                score=scored,shot=record['shot'],actions=record['actions'],events=a.events,
                native_contact=contacts,placements=placed,clearance=actual,
                exact_command_events=True,qualifies_100=bool(scored['learner']>=100 and not actual['penetration_samples']))
    _write(out,result); return out


def run(kind,feeds,n,schedule='preliminary',frozen=False):
    schedule_path=PLAN if schedule=='preliminary' else SEASON_PLAN
    p=_read(schedule_path); record=_read(system.OUT); build=_read(system.BUILD)
    if (p['protocol']!=_hash(system.PLAN) or p['carry']!=_hash(SEASON.CARRY) or record['build']!=_hash(system.BUILD)):
        raise ValueError('Qualification or fresh full-game docket changed')
    install=[r for r in record['rows'] if r['body']==1]
    low=[r for r in install if r['case']['role']=='low']; routine=[r for r in install if r['case']['role']=='routine']
    stock=[r for r in record['rows'] if r['body']==0 and r['case']['role']=='low']
    if (len(low)!=sum(r['role']=='low' for r in system.contract()['physical']) or len(routine)<2
            or sum(r['real'] for r in low)<max(2,1+sum(r['real'] for r in stock))
            or not any(r['broad'] for r in low) or not all(r['reference'] or r['real'] for r in routine)):
        raise ValueError('The aggregate mechanical opportunity improvement did not reproduce on actual held balls')
    if schedule!='preliminary':
        if not record['complete']:
            raise ValueError('Complete the full mechanical routine and low assay before the whole season')
        if p['tt80_sha256']!=_hash(TT80) or p['cards']!=_read(TT80)['practice']:
            raise ValueError('The entire independent centered TT80 roster must carry')
        pilot=_read(ROOT/'installed-online.json.gz')
        if len(pilot['rows'])<3 or not any(r['score']['learner']>=3 and not r['clearance']['penetration_samples'] for r in pilot['rows']):
            raise ValueError('The small preliminary batch did not show viable real online receiving')
    item=build['rows'][0 if kind=='stock' else build['chosen']]; m=item['candidate']; ready=item['commission']['ready']
    fixed=build['rows'][0]['candidate']['mount']['base']; reg=Registration(fixed,m['mount']['base'],m['mount']['rotation'])
    shots=MP.measured_launches(feeds); cards=p['cards']; ids={c['id'] for c in cards}
    mark=kind+('-frozen' if frozen else '-online') if schedule=='preliminary' else 'season-'+('frozen' if frozen else 'live')
    out=ROOT/(mark+'.json.gz'); bank=ROOT/(mark+'.npz'); mode='frozen' if frozen else 'live'
    docket=dict(protocol=_hash(schedule_path),body=kind,fixture_receipt=_hash(system.BUILD),rows=[],cap=p['cap'],cards=cards,
                mode=mode,schedule=schedule)
    if out.exists():
        old=_read(out)
        if any(old[k]!=v for k,v in docket.items() if k!='rows'):raise ValueError('Changed frozen game conditions')
        docket=old
    with np.load(MP.BANK) as f:raw0=MP.DeepField(f,'local')
    raw=raw0 if kind=='stock' else LocalField(raw0,reg)
    rank=SEASON.fresh.shot.Coverage(_read(SEASON.fresh.RISK))
    with system.fixture.school.enabled(),system.installed(m['v'],m['mount']),MP.centered(ready):
        a=MP.arena(); cad=MP.arena(detail='cad'); cam,rig,stride,lo,hi=MP.resources(a); clearance=system.q.GripClearance(cad.model,stock=True)
        replayer=MP.arena(width=64,height=48)
        try:
            with patch.object(MP,'Fragments',lambda b,e:Carried(b,e,practice=[COURSE.PRACTICE],held=ids)):
                kws=dict(variant='elastic',budget=28,coverage=rank)
                actor=(SEASON.Player(MP.BANK,raw,**kws) if kind=='stock' else Registered(MP.BANK,raw,register=reg,**kws))
            try:
                actor.priors.home=np.array(ready); prior=_read(SEASON.CARRY)
                COURSE.resume(actor,SEASON.CARRY.with_suffix('').with_suffix('.npz'),prior)
                if docket['rows']:COURSE.resume(actor,bank,dict(docket,rows=prior['rows']+docket['rows']))
                for index,c in enumerate(cards[len(docket['rows']):n],start=len(docket['rows'])):
                    if schedule=='preliminary':guarded(a,clearance)
                    start=a.launches
                    r=MP.play(a,cam,rig,stride,lo,hi,ready,c,shots[c['id']],raw,actor,COURSE.OPTIC,mode,p['cap'],progress=20)
                    if schedule=='preliminary':verify_clearance(a)
                    if (a.launches!=start+1 or any(v['draw']!='random' or list(draw(c,j,'random',None)[0])!=v['target']
                                                  for j,v in enumerate(r['placements']['requested']))):
                        raise ValueError('Opponent law, launch lifecycle or continuous physical scoring changed')
                    if frozen and r['memory']['before']!=r['memory']['after']:raise ValueError('Frozen memory unexpectedly changed')
                    if schedule=='preliminary':r['clearance']=a.clearance_result()
                    best=max((z['score']['learner'] for z in docket['rows']),default=0)
                    if r['score']['learner']>=100 or (schedule!='preliminary' and r['score']['learner']>max(best,42)):
                        saved=audit_witness(r,c,replayer,clearance,kind,index,mark,ready,m)
                        r['verified_witness']=dict(path=str(saved),sha256=_hash(saved));
                    docket['rows'].append(r)
                    docket['checkpoint_sha256']=COURSE.save(actor,bank);_write(out,docket)
                    print(kind,mark,len(docket['rows']),c['id'],r['score']['learner'],r.get('clearance',{}).get('penetration_samples'),flush=True)
                if schedule!='preliminary' and len(docket['rows'])==len(cards):
                    novel=max(range(len(cards)),key=lambda i:docket['rows'][i]['score']['learner']); winner=docket['rows'][novel]
                    if not winner.get('verified_witness'):
                        saved=audit_witness(winner,cards[novel],replayer,clearance,kind,novel,mark,ready,m)
                        winner['verified_witness']=dict(path=str(saved),sha256=_hash(saved));_write(out,docket)
            finally:actor.close()
        finally:cam.close();cad.close();a.close();replayer.close()


def pilot_witness():
    p=_read(PLAN); b=_read(system.BUILD); d=_read(ROOT/'installed-online.json.gz'); item=b['rows'][b['chosen']]; m=item['candidate'];
    if d['protocol']!=_hash(PLAN) or d['fixture_receipt']!=_hash(system.BUILD) or len(d['rows'])!=len(p['cards']):
        raise ValueError('Complete mounted pilot before independently verifying its maximum')
    j=max(range(len(d['rows'])),key=lambda i:d['rows'][i]['score']['learner']); ready=item['commission']['ready']
    with system.fixture.school.enabled(cameras=False),system.installed(m['v'],m['mount']),MP.centered(ready):
        real=MP.arena(width=64,height=48); cad=MP.arena(width=64,height=48,detail='cad')
        try:
            out=audit_witness(d['rows'][j],p['cards'][j],real,system.q.GripClearance(cad.model,stock=True),'installed',j,'pilot',ready,m)
            check=_read(out)
            if check['clearance']!=d['rows'][j]['clearance']:
                raise ValueError('Pilot clearance receipt differs from the independent CAD replay')
            print('exact mounted pilot',j,p['cards'][j]['id'],check['score']['learner'],check['clearance']['samples'],flush=True)
        finally:real.close();cad.close()


def season_witness(index):
    p=_read(SEASON_PLAN); d=_read(ROOT/'season-live.json.gz'); b=_read(system.BUILD)
    r=b['rows'][b['chosen']]; m=r['candidate']; ready=r['commission']['ready']
    if d['protocol']!=_hash(SEASON_PLAN) or d['fixture_receipt']!=_hash(system.BUILD) or d['cards']!=p['cards']:
        raise ValueError('Changed normal installed season conditions')
    if index is None:index=max(range(len(d['rows'])),key=lambda i:d['rows'][i]['score']['learner'])
    if not 0<=index<len(d['rows']):raise ValueError('Only completed archived points can be independently audited')
    c=p['cards'][index]
    with system.fixture.school.enabled(cameras=False),system.installed(m['v'],m['mount']),MP.centered(ready):
        real=MP.arena(width=64,height=48); cad=MP.arena(width=64,height=48,detail='cad')
        try:
            out=audit_witness(d['rows'][index],c,real,system.q.GripClearance(cad.model,stock=True),
                              'installed',index,'season-independent',ready,m)
            check=_read(out)
            print('exact mounted season',index,c['id'],check['score']['learner'],check['clearance']['samples'],flush=True)
        finally:real.close();cad.close()


def summary():
    def snapshot(path):
        copied=path.read_bytes()
        return json.loads(gzip.decompress(copied)),sha256(copied).hexdigest()
    readouts={}
    for name in ('stock-online','installed-online','installed-frozen','season-live','season-frozen'):
        path=ROOT/(name+'.json.gz')
        if not path.exists():continue
        d,signature=snapshot(path); a=np.asarray([r['score']['learner'] for r in d['rows']]); low=np.zeros(2,dtype=int); other=low.copy()
        if not len(a):continue
        for row in d['rows']:
            for offer in row['placements']['delivered']:
                f=offer.get('receive_line')
                if f is None:continue
                shelf=low if projected_height(f)<1.01 else other
                shelf+=np.array([1,bool(offer['near_landed'])])
        readouts[name]=dict(source_sha256=signature,played=len(a),assigned=len(d['cards']),complete=len(a)==len(d['cards']),
                             returns=int(a.sum()),median=float(np.median(a)),peak=int(a.max()),
                             zeros=int(np.sum(a==0)),at_least_20=int(np.sum(a>=20)),at_least_100=int(np.sum(a>=100)),
                             low=dict(offered=int(low[0]),landed=int(low[1])),other=dict(offered=int(other[0]),landed=int(other[1])),
                             audited=[r['verified_witness'] for r in d['rows'] if r.get('verified_witness')])
    equal_body=[]; live_path=ROOT/'installed-online.json.gz'; frozen_path=ROOT/'installed-frozen.json.gz'
    if live_path.exists() and frozen_path.exists():
        l,_=snapshot(live_path); f,_=snapshot(frozen_path)
        if (l['cards']!=f['cards'] or l['rows'][0]['memory']['before']!=f['rows'][0]['memory']['before']):
            raise ValueError('Fixed-body memory comparison changed starting bank or requested card')
        for card,a,b in zip(l['cards'],l['rows'],f['rows']):
            changed=next((i for i,(v,w) in enumerate(zip(a['actions'],b['actions'])) if v!=w),None)
            if changed is None and a['actions']!=b['actions']:changed=min(len(a['actions']),len(b['actions']))
            pts=[]
            for c,d in zip(a['placements']['delivered'],b['placements']['delivered']):
                if any(digest_json(c[k])!=digest_json(d[k]) for k in ('far_contact','near_bounce')):break
                pts.append([c['near_landed'],d['near_landed']])
            t=np.inf if changed is None else changed*MP.DT
            common=[e for e in a['events'] if e['time']<=t+1e-8]
            prefix=digest_json(common)==digest_json([e for e in b['events'] if e['time']<=t+1e-8])
            if not prefix:
                raise ValueError('Memoization comparison physical prefix diverged before its first changed command')
            equal_body.append(dict(id=card['id'],online=a['score']['learner'],frozen=b['score']['learner'],
                                  action_change_tick=changed,exact_initial_offers=pts,exact_event_prefix=prefix,
                                  online_prior_changed=a['memory']['before']!=b['memory']['before'],
                                  own_before_first_change=audit(common,opening_side=1)['learner']))
    matched={}; mounted=ROOT/'season-live.json.gz'
    if mounted.exists():
        ml,ml_hash=snapshot(mounted)
        for label in ('live','frozen'):
            cmp=TT80.with_name(label+'.json.gz')
            if not cmp.exists():continue
            old,cmp_hash=snapshot(cmp); n=min(len(ml['rows']),len(old['rows']))
            if ml['cards'][:n]!=old['cards'][:n]:raise ValueError('Comparison no longer uses the same preassigned draws')
            aa=np.array([z['score']['learner'] for z in ml['rows'][:n]])
            bb=np.array([z['score']['learner'] for z in old['rows'][:n]])
            if n:
                matched[label]=dict(n=n,mount=int(aa.sum()),stock=int(bb.sum()),mount_peak=int(aa.max()),stock_peak=int(bb.max()),
                                    gains=int(np.sum(aa>bb)),ties=int(np.sum(aa==bb)),losses=int(np.sum(aa<bb)),
                                    mounted_source_sha256=ml_hash,centered_source_sha256=cmp_hash)
    _write(ROOT/'online-summary.json',dict(groups=readouts,matched_season_prefix=matched,paired_fixed_body=equal_body))
    pilots=('stock-online','installed-online','installed-frozen')
    if all(k in readouts and readouts[k]['complete'] for k in pilots):
        _write(ROOT/'pilot-comparison.json',dict(groups={k:readouts[k] for k in pilots},paired_fixed_body=equal_body))
    print(readouts, matched, flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','stock','installed','pilot-frozen','pilot-audit','season-plan','season-live','season-frozen','season-audit','summary'))
    a.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'));a.add_argument('--n',type=int)
    p=a.parse_args()
    if p.mode=='plan':freeze()
    elif p.mode=='summary':summary()
    elif p.mode=='pilot-audit':pilot_witness()
    elif p.mode=='season-audit':season_witness(p.n)
    elif p.mode=='pilot-frozen':run('installed',p.feeds,8 if p.n is None else p.n,'preliminary',True)
    elif p.mode=='season-plan':season_plan()
    elif p.mode.startswith('season-'):run('installed',p.feeds,128 if p.n is None else p.n,'season',p.mode=='season-frozen')
    else:run(p.mode,p.feeds,8 if p.n is None else p.n)
