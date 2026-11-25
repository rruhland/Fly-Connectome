"""Continuous RGB/encoder online transfer to a mechanically qualified installation."""

import argparse
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
        if len(pilot['rows'])!=8 or not any(r['score']['learner']>=3 and not r['clearance']['penetration_samples'] for r in pilot['rows']):
            raise ValueError('The small preliminary batch did not show viable real online receiving')
    item=build['rows'][0 if kind=='stock' else build['chosen']]; m=item['candidate']; ready=item['commission']['ready']
    fixed=build['rows'][0]['candidate']['mount']['base']; reg=Registration(fixed,m['mount']['base'],m['mount']['rotation'])
    shots=MP.measured_launches(feeds); cards=p['cards']; ids={c['id'] for c in cards}
    mark=kind+'-online' if schedule=='preliminary' else 'season-'+('frozen' if frozen else 'live')
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


def summary():
    readouts={}
    for name in ('stock-online','installed-online','season-live','season-frozen'):
        path=ROOT/(name+'.json.gz')
        if not path.exists():continue
        d=_read(path); a=np.asarray([r['score']['learner'] for r in d['rows']]); low=np.zeros(2,dtype=int); other=low.copy()
        if not len(a):continue
        for row in d['rows']:
            for offer in row['placements']['delivered']:
                f=offer.get('receive_line')
                if f is None:continue
                shelf=low if projected_height(f)<1.01 else other
                shelf+=np.array([1,bool(offer['near_landed'])])
        readouts[name]=dict(source_sha256=_hash(path),played=len(a),assigned=len(d['cards']),complete=len(a)==len(d['cards']),
                             returns=int(a.sum()),median=float(np.median(a)),peak=int(a.max()),
                             zeros=int(np.sum(a==0)),at_least_20=int(np.sum(a>=20)),at_least_100=int(np.sum(a>=100)),
                             low=dict(offered=int(low[0]),landed=int(low[1])),other=dict(offered=int(other[0]),landed=int(other[1])),
                             audited=[r['verified_witness'] for r in d['rows'] if r.get('verified_witness')])
    _write(ROOT/'online-summary.json',dict(groups=readouts))
    print(readouts,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','stock','installed','season-plan','season-live','season-frozen','summary'))
    a.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'));a.add_argument('--n',type=int)
    p=a.parse_args()
    if p.mode=='plan':freeze()
    elif p.mode=='summary':summary()
    elif p.mode=='season-plan':season_plan()
    elif p.mode.startswith('season-'):run('installed',p.feeds,128 if p.n is None else p.n,'season',p.mode=='season-frozen')
    else:run(p.mode,p.feeds,8 if p.n is None else p.n)
