"""Continuous RGB/encoder online transfer to a mechanically qualified installation."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from . import tt84_installation_system as system
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt37_opponent_audit import guarded, verify_clearance
from .tt39_carry_joints import Carried


ROOT=system.ROOT
PLAN=ROOT/'games.protocol.json'
SEASON=dose.season
COURSE=dose.course
MP=system.MP


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


def run(kind,feeds,n):
    p=_read(PLAN); record=_read(system.OUT); build=_read(system.BUILD)
    if (p['protocol']!=_hash(system.PLAN) or p['carry']!=_hash(SEASON.CARRY) or not record['complete'] or record['build']!=_hash(system.BUILD)):
        raise ValueError('Qualification or fresh full-game docket changed')
    install=[r for r in record['rows'] if r['body']==1]
    low=[r for r in install if r['case']['role']=='low']; routine=[r for r in install if r['case']['role']=='routine']
    stock=[r for r in record['rows'] if r['body']==0 and r['case']['role']=='low']
    if (sum(r['real'] for r in low)<max(2,1+sum(r['real'] for r in stock))
            or not any(r['broad'] for r in low) or not all(r['reference'] or r['real'] for r in routine)):
        raise ValueError('The aggregate mechanical opportunity improvement did not reproduce on actual held balls')
    item=build['rows'][0 if kind=='stock' else build['chosen']]; m=item['candidate']; ready=item['commission']['ready']
    fixed=build['rows'][0]['candidate']['mount']['base']; reg=Registration(fixed,m['mount']['base'],m['mount']['rotation'])
    shots=MP.measured_launches(feeds); cards=p['cards']; ids={c['id'] for c in cards}
    out=ROOT/(kind+'-online.json.gz'); bank=ROOT/(kind+'-online.npz')
    docket=dict(protocol=_hash(PLAN),body=kind,fixture_receipt=_hash(system.OUT),rows=[],cap=p['cap'],cards=cards)
    if out.exists():
        old=_read(out)
        if any(old[k]!=v for k,v in docket.items() if k!='rows'):raise ValueError('Changed frozen game conditions')
        docket=old
    with np.load(MP.BANK) as f:raw0=MP.DeepField(f,'local')
    raw=raw0 if kind=='stock' else LocalField(raw0,reg)
    rank=SEASON.fresh.shot.Coverage(_read(SEASON.fresh.RISK))
    with system.fixture.school.enabled(),system.installed(m['v'],m['mount']),MP.centered(ready):
        a=MP.arena(); cad=MP.arena(detail='cad'); cam,rig,stride,lo,hi=MP.resources(a)
        try:
            with patch.object(MP,'Fragments',lambda b,e:Carried(b,e,practice=[COURSE.PRACTICE],held=ids)):
                kws=dict(variant='elastic',budget=28,coverage=rank)
                actor=(SEASON.Player(MP.BANK,raw,**kws) if kind=='stock' else Registered(MP.BANK,raw,register=reg,**kws))
            try:
                actor.priors.home=np.array(ready); prior=_read(SEASON.CARRY)
                COURSE.resume(actor,SEASON.CARRY.with_suffix('').with_suffix('.npz'),prior)
                if docket['rows']:COURSE.resume(actor,bank,dict(docket,rows=prior['rows']+docket['rows']))
                for c in cards[len(docket['rows']):n]:
                    guarded(a,system.q.GripClearance(cad.model,stock=True)); start=a.launches
                    r=MP.play(a,cam,rig,stride,lo,hi,ready,c,shots[c['id']],raw,actor,COURSE.OPTIC,'live',p['cap'],progress=20)
                    verify_clearance(a)
                    if (a.launches!=start+1 or any(v['draw']!='random' or list(draw(c,j,'random',None)[0])!=v['target']
                                                  for j,v in enumerate(r['placements']['requested']))):
                        raise ValueError('Opponent law, launch lifecycle or continuous physical scoring changed')
                    r['clearance']=a.clearance_result(); docket['rows'].append(r)
                    docket['checkpoint_sha256']=COURSE.save(actor,bank);_write(out,docket)
                    print(kind,len(docket['rows']),c['id'],r['score']['learner'],r['clearance']['penetration_samples'],flush=True)
            finally:actor.close()
        finally:cam.close();cad.close();a.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','stock','installed'))
    a.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'));a.add_argument('--n',type=int,default=8)
    p=a.parse_args();freeze() if p.mode=='plan' else run(p.mode,p.feeds,p.n)
