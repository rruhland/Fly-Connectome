"""Stock TT46 with a camera-only wing branch that is absent until authorization."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_live_wing_v3 as checked

first=checked.first;rp=first.rp;mpc=rp.mpc;prior=rp.pf.prior
ROOT=Path('artifacts/tt55_stock_parent_wing')
PLAN=ROOT/'plan.json'


class Arm(rp.Arm):
    def __init__(self,*a,body=None,wing_options=None,**kw):
        super().__init__(*a,**kw)
        self.wing=body;self.wing_options=wing_options

    def begin(self):
        super().begin()
        self.repertory=None
        self.involved=False
        self.applied=[]
        if self.ready.predictor is not None:self.ready.predictor.estimates.clear()
        if self.wing is not None:self.wing.reset()
        self.body_trace=[] if self.wing is None else self.wing.logs

    def step(self,tick,eye,x,q,ref,base):
        outward=x is not None and x[4]>.9
        if outward and self.wing is not None and self.wing.inward is not None:
            self.involved=False
        # The parent's default prepare/rollout/cadence/lease/RNG is inherited.
        # A dormant or vetoed branch cannot add options to those native trials.
        use=(self.wing is not None and self.involved and not self.wing.declined
             and x is not None and x[4]<-.9 and x[1]<-.49)
        self.repertory=self.wing_options if use else None
        plain=super().step(tick,eye,x,q,ref,base)
        if self.wing is None:return plain
        g=self.wing.command(tick,x,q,ref,plain,self)
        if g is None:return plain
        if self.wing.permission and not np.array_equal(g,plain):
            self.involved=True
            self.applied.append(tick)
        return g


def install(raw,checker,reference,*,enabled,frames):
    p=Arm(mpc.BANK,raw,mode='stock',finish=rp.pf.Finish(),checker=checker,
          predictor=rp.old.logged_predictor(rp.risk_model),repertory=None,
          weak=True,body=checked.Wing(rp._read(first.BANK),checker) if enabled else None,
          wing_options=rp.Joined(None,rp.low_field(rp.low_artifact)) if enabled else None,
          variant=reference['variant'],budget=reference['budget'],
          coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
    p.tap=frames
    return p


def freeze():
    block={2741,12246}|{r['id'] for r in rp._read(rp.pf.plan_path)['rows']}
    earlier=((prior.course.PLAN,'practice'),(prior.season.PLAN,'practice'),(prior.fresh.PLAN,'assessment'),
             (rp.old.WHOLE,'cards'),(rp.plan_path,'cards'),(first.root/'whole-plan.json','cards'),
             (first.root/'retained-whole-plan.json','cards'))
    for path,k in earlier:block.update(c['id'] for c in rp._read(path)[k])
    block.update(c['id'] for c in rp._read(prior.VISION_ROOT/'difficult-plan.json')['cases'])
    for p in (prior.VISION_ROOT/'observations').glob('case-*.npz'):
        with np.load(p) as z:block.add(int(z['id']))
    src=rp._read(prior.course.SOURCE)
    pool=sorted({c['id']:c for c in src['practice']+src['assessment']
                 if c['id'] not in block and c['profile']=='reach' and c['opening']=='random'}.values(),key=lambda c:c['id'])
    rng=np.random.default_rng(55061099)
    cards=[dict(pool[i],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
           for i in rng.permutation(len(pool))[:4]]
    if len(cards)!=4:raise ValueError('Need four untouched independent physical launches')
    rp._write(PLAN,dict(cards=cards,unused_pool=len(pool),carry_sha256=rp._hash(prior.season.CARRY),
                        checked_scout_sha256=rp._hash(Path(checked.__file__)),adapter_sha256=rp._hash(Path(__file__)),
                        risk_sha256=rp._hash(rp.risk_model),bank_sha256=rp._hash(first.BANK),
                        role='True TT46 live-vs-adapter pair from original TT46 memory. Entire paired point from launch. No shared partial-body change.'))


def exact(out):
    with patch.object(first,'install',install):first.run(out)
    d=rp._read(out);d['stock_parent_adapter_sha256']=rp._hash(Path(__file__));rp._write(out,d)


def whole(out,limit):
    lock=rp._read(PLAN);base=rp._read(prior.season.CARRY)
    p0=prior.season.CARRY.with_suffix('').with_suffix('.npz')
    if (lock['carry_sha256']!=rp._hash(prior.season.CARRY) or lock['adapter_sha256']!=rp._hash(Path(__file__)) or
        lock['checked_scout_sha256']!=rp._hash(Path(checked.__file__)) or lock['risk_sha256']!=rp._hash(rp.risk_model) or
        lock['bank_sha256']!=rp._hash(first.BANK)):
        raise ValueError('Changed registered actor or experience')
    doc=dict(plan_sha256=rp._hash(PLAN),cards=lock['cards'][:limit],rows=[],complete=False)
    if out.exists():
        d=rp._read(out)
        if d['plan_sha256']!=doc['plan_sha256'] or d['cards']!=doc['cards']:raise ValueError('Different physical course')
        doc=d
    shots=mpc.measured_launches(Path('/tmp/tt25-rallies.json'))
    boot=base['ready']
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad')
        camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera);checker=prior.Reach(cad,boot,boot)
        try:
            for c in doc['cards'][len(doc['rows']):]:
                row=dict(id=c['id'],variants={})
                for mode in ('tt46-parent','wing-adapter'):
                    with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={c['id']})):
                        if mode=='tt46-parent':
                            a0=prior.season.Player(mpc.BANK,raw,variant=base['variant'],budget=base['budget'],
                                    coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
                        else:
                            a0=install(raw,checker,base,enabled=True,frames=tap)
                    try:
                        prior.course.resume(a0,p0,base)
                        result=mpc.play(a,tap,rig,stride,lo,hi,boot,c,shots[c['id']],raw,a0,prior.course.OPTIC,'live',240,20)
                        if any(t['target']!=list(rp.draw(c,j,'random',None)[0]) for j,t in enumerate(result['placements']['requested'])):
                            raise ValueError('Changed broad opponent')
                        row['variants'][mode]=result|dict(applied=getattr(a0,'applied',[]),body=getattr(a0,'body_trace',[]),
                                                         opts=getattr(a0,'repertory_log',[]))
                    finally:a0.close()
                stock,aug=row['variants'].values()
                first_move=next((k for k,(u,v) in enumerate(zip(stock['actions'],aug['actions'])) if u!=v),None)
                row['first_matched_difference']=first_move
                row['abstains_entire_game']=not aug['applied'] and not any(x['accepted'] for x in aug['opts'])
                if stock['memory']['before']!=aug['memory']['before']:raise ValueError('Point-start memory differed')
                if row['abstains_entire_game'] and (stock['actions']!=aug['actions'] or stock['memory']['after']!=aug['memory']['after'] or
                    prior.digest_json(stock['events'])!=prior.digest_json(aug['events'])):
                    row['native_abstention_mismatch']=True;doc['rows'].append(row);rp._write(out,doc)
                    raise ValueError('An inactive scout altered the parent')
                if first_move is not None and (not aug['applied'] or first_move<aug['applied'][0]):
                    raise ValueError('Commands changed before the authorized branch')
                doc['rows'].append(row);rp._write(out,doc)
                print(c['id'],stock['score']['learner'],aug['score']['learner'],first_move,not aug['applied'],flush=True)
        finally:a.close();camera.close();cad.close()
    doc['complete']=True;rp._write(out,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','exact','whole'))
    p.add_argument('--out',type=Path);p.add_argument('--limit',type=int,default=4);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.phase=='plan':freeze()
    elif a.phase=='exact':exact(a.out or ROOT/'exact.json.gz')
    else:whole(a.out or ROOT/'whole.json.gz',a.limit)
