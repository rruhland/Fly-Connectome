"""Preflight full-law actual replies without rendering before selecting optical lessons."""
from pathlib import Path

import numpy as np

from . import tt56_central_training as train
from . import tt51_low_reverse_curriculum as optical
from .tt4_transfer_trial import _hash,_read
from .tt19_view_full_game import digest_json
from .tt44_paired_credit import State

ROOT=train.ROOT
SCOUT=ROOT/'scout-native.json.gz'
ACTIVE=ROOT/'active-protocol.json'


def run():
    base=_read(train.PLAN); initial=_read(train.TRAINING)
    if not initial['complete']:
        raise ValueError('Complete the original fixed training before expanding adaptive practice')
    rng=np.random.default_rng(560522)
    record=dict(original_training_sha256=_hash(train.TRAINING),scouts=[],role='Practice-only physical prefilter to pay for camera captures where a materially harder real central delivery exists; final selection still based on actual observed RGB. No held release simulated.')
    selected=[]
    for r in (q for q in base['rows'] if q['split']=='train'):
        spec=r['case']; raw=_read(Path(spec['source'])); j=spec['index']; c,o=raw['cards'][j],raw['rows'][j]
        old_ids={w['seed'] for w in r['draws']}; probes=[];leg=spec['leg'];anchor=r['anchor']
        with train.mpc.centered(raw['ready']):
            a=train.mpc.arena(width=64,height=48)
            try:
                a.launch(o['shot'],card=c);a.partner.reset(seed=c['seed'])
                for u in o['actions'][:anchor]:a.exchange(u,train.mpc.FAR_COMMAND)
                if len(a.partner.requests)!=leg or digest_json(a.events)!=digest_json([e for e in o['events'] if e['time']<=a.data.time+1e-8]):
                    raise ValueError('Restricted developmental practice physical prefix changed')
                root=State(a)
                for seed in rng.integers(1,2**30,size=32).tolist():
                    if seed in old_ids:continue
                    old_ids.add(seed);root.restore(a);a.card=train.changed(c,leg,seed)
                    expert,tag=train.draw(a.card,leg,'random',None)
                    for u in o['actions'][anchor:anchor+115]:
                        _,_,end,cap,_=a.exchange(u,train.mpc.FAR_COMMAND)
                        if end or cap or leg+1 in a.crossings:break
                    p=a.point_placements(); claim=p['requested'][leg]['target']
                    if claim!=list(expert) or tag!='random':
                        raise ValueError('The prefilter may only scout original independent continuous draw')
                    line=a.crossings.get(leg+1)
                    h=None if line is None else train.deep(np.r_[line['xyz'],line['velocity']]).tolist()
                    probes.append(dict(seed=seed,target=claim,evaluator=h,
                        near_offered=len(p['delivered'])>leg, steps=a.steps-anchor))
            finally:
                a.close()
        local=[p for p in probes if p['evaluator'] is not None and abs(p['evaluator'][0])<.118
                                     and .979 < p['evaluator'][1] < 1.016]
        chosen=[]
        for p in sorted(local,key=lambda p:abs(p['evaluator'][1]-.995)+.08*abs(p['evaluator'][0])):
            if any(np.linalg.norm((np.asarray(p['evaluator'])-q['evaluator'])/train.BANK_GUARD)<.25 for q in chosen):
                continue
            chosen.append(p)
            if len(chosen)>=6:break
        selected.append(r|{'draws':[dict(seed=p['seed'],target=p['target']) for p in chosen]})
        record['scouts'].append(dict(id=c['id'],surveyed=probes,selected=chosen))
        optical.save(SCOUT,record)
        print(c['id'],len(probes),len(chosen),[p['evaluator'][:2] for p in chosen],flush=True)
    record['complete']=True;optical.save(SCOUT,record)
    optical.save(ACTIVE,base|dict(rows=selected,coaching_family='signed_geometry',public_earlier_gate=True,native_prefilter_sha256=_hash(SCOUT),
        limits=dict(batch=96,rounds=2,maximum_variants=2,max_physical_candidates=2),
        role='Additional practice only. All 128 physical probes follow unmodified standard independent draws. Native receiving trajectory prioritizes cases near a real challenging center lane before rendering. Actual later RGB determines selection of up to two material directions per release. The private coach samples both signed faces and contact windows from the earlier body attained at the first optical early near-side fit, before the baseline late search has moved it. No held source is simulated or trained.'))


if __name__=='__main__':run()
