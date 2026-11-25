"""Preassign and compare complete normal-start points with the causal wing reserve."""

import argparse
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_live_wing as wing

rp=wing.rp;mpc=rp.mpc
PLAN=wing.root/'whole-plan.json'


def freeze(output=PLAN,exclude=None):
    prior=rp.pf.prior
    blocked={2741}|{r['id'] for r in rp._read(rp.pf.plan_path)['rows']}
    for path,k in ((prior.course.PLAN,'practice'),(prior.season.PLAN,'practice'),(prior.fresh.PLAN,'assessment'),
                   (rp.old.WHOLE,'cards'),(rp.plan_path,'cards')):
        blocked.update(c['id'] for c in rp._read(path)[k])
    if exclude is not None:
        blocked.update(c['id'] for c in rp._read(exclude)['cards'])
    blocked.update(c['id'] for c in rp._read(prior.VISION_ROOT/'difficult-plan.json')['cases'])
    for p in (prior.VISION_ROOT/'observations').glob('case-*.npz'):
        with np.load(p) as z:blocked.add(int(z['id']))
    src=rp._read(prior.course.SOURCE)
    pool=sorted({c['id']:c for c in src['practice']+src['assessment']
                 if c['id'] not in blocked and c['profile']=='reach' and c['opening']=='random'}.values(),key=lambda c:c['id'])
    rng=np.random.default_rng(54171031 if exclude is None else 54248033)
    cards=[dict(pool[i],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
           for i in rng.permutation(len(pool))[:4]]
    if len(cards)!=4:raise ValueError('Need four untouched measured releases')
    rp._write(output,dict(cards=cards,unused_pool=len(pool),risk_sha256=rp._hash(rp.risk_model),
                        body_sha256=rp._hash(wing.BANK),role='Fresh preassigned normal starts, paired stock-window actors differ only in causal wing reserve.'))


def run(path,limit,module,plan=PLAN):
    reg=rp._read(plan);prior=rp.pf.prior;src=rp._read(prior.ARCHIVE)
    if reg['risk_sha256']!=rp._hash(rp.risk_model) or reg['body_sha256']!=rp._hash(wing.BANK):
        raise ValueError('Changed frozen sensory or motor packet')
    model=import_module(module)
    boot=src['ready'];cards=reg['cards'][:limit]
    doc=dict(cards=cards,plan_sha256=rp._hash(plan),module=module,model_sha256=rp._hash(Path(model.__file__)),rows=[])
    if path.exists():
        old=rp._read(path)
        if any(old.get(k)!=v for k,v in doc.items() if k!='rows'):raise ValueError('Changed whole-game protocol')
        doc=old
    shots=mpc.measured_launches(Path('/tmp/tt25-rallies.json'))
    carried={**src,'rows':[None]*len(rp._read(prior.season.CARRY)['rows'])+src['rows']}
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad')
        camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera)
        checker=prior.Reach(cad,boot,boot)
        try:
            for c in cards[len(doc['rows']):]:
                row=dict(id=c['id'],variants={})
                for label,enabled in (('prepared-stock',False),('causal-wing',True)):
                    with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={c['id']})):
                        actor=model.install(raw,checker,src,enabled=enabled,frames=tap)
                    try:
                        prior.course.resume(actor,prior.BANK,carried)
                        play=mpc.play(a,tap,rig,stride,lo,hi,boot,c,shots[c['id']],raw,actor,prior.course.OPTIC,'live',240,20)
                        if any(q['target']!=list(rp.draw(c,j,'random',None)[0]) for j,q in enumerate(play['placements']['requested'])):
                            raise ValueError('Changes to the official reactive draw')
                        row['variants'][label]=play|dict(body_trace=actor.body_trace,opts=actor.repertory_log,
                                                        risk=actor.ready.predictor.estimates)
                    finally:actor.close()
                s,w=row['variants'].values()
                if s['memory']['before']!=w['memory']['before']:raise ValueError('Unmatched starting motor experience')
                row['first_changed_tick']=next((i for i,(u,v) in enumerate(zip(s['actions'],w['actions'])) if u!=v),None)
                doc['rows'].append(row);rp._write(path,doc)
                print(c['id'],s['score']['learner'],w['score']['learner'],row['first_changed_tick'],len(w['body_trace']),flush=True)
        finally:a.close();camera.close();cad.close()
    doc['complete']=True;rp._write(path,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=wing.root/'whole.json.gz')
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--module',default='scripts.experimental.tt54_live_wing')
    p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--exclude',type=Path)
    a=p.parse_args();wing.root.mkdir(parents=True,exist_ok=True)
    if a.phase=='plan':freeze(a.plan,exclude=a.exclude)
    else:run(a.out,a.limit,a.module,plan=a.plan)
