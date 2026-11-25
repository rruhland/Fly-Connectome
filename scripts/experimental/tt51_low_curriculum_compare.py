"""Matched official TT37 draws for the separate reward-practiced workspace skill."""
import argparse
from pathlib import Path
from time import process_time,monotonic
from unittest.mock import patch
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as combined
from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from . import tt51_low_contact_practice as skill
from .tt4_transfer_trial import _read,_write,_hash
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent_audit import CourtClearance,guarded,verify_clearance
from .tt39_opportunity_map import projected_height


def stock_model(raw,held,carry):
    with patch.object(mpc,'Fragments',lambda file,engine:course.Carried(file,engine,
        practice=[course.PRACTICE],held=held)):
        model=season.Player(mpc.BANK,raw,coverage=combined.shot.Coverage(_read(combined.RISK)),
            variant='elastic',budget=28)
    document=_read(carry);path=carry.with_suffix('').with_suffix('.npz')
    if carry==season.CARRY:course.resume(model,path,document)
    else:retention.ancestral_restore(model,path,document,season.CARRY)
    return model


def run(a):
    p=_read(skill.PLAN);cards=p['assessment'][:a.n];held={c['id'] for c in cards}
    if a.memory:
        training=_read(a.memory)
        if not training['complete'] or {c['id'] for c in training['cards']}&held:
            raise ValueError('Training must be completed and release-disjoint')
    ready=np.asarray(p['ready']);feeds=mpc.measured_launches(a.feeds)
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    result=dict(schema='tt51_official_broad_paired_v1',cards=cards,ready=ready.tolist(),
        plan_sha256=_hash(skill.PLAN),memory_sha256=_hash(a.memory) if a.memory else None,
        carry_sha256=_hash(a.carry),source_sha256=_hash(Path(__file__)),arms=a.arms,cap=a.cap,
        normal_physical_start=True,unchanged_broad_continuous_opponent=True,rows=[],complete=False)
    clock=monotonic();cpu=process_time()
    with mpc.centered(ready):
        arena=mpc.arena();res=mpc.resources(arena)
        stock=stock_model(raw,held,a.carry) if 'stock' in a.arms else None
        models={};initial={}
        for name in a.arms:
            if name=='stock':continue
            memory=skill.ContactMemory(p['hub'],learn=name=='online',
                packet=training['memory'] if name!='cold' else None)
            models[name]=skill.ContactSkill(p['hub'],memory)
            initial[name]=memory.digest()
        try:
            for c in cards:
                paired=dict(card=c,arms={})
                for name in a.arms:
                    model=stock if name=='stock' else models[name]
                    before=model.priors.digest()
                    row=mpc.play(arena,*res,ready,c,feeds[c['id']],raw,model,course.OPTIC,
                        'live' if name=='online' else 'frozen',a.cap)
                    if any(v['target']!=list(skill.base.STANDARD_DRAW(c,j,'random',None)[0])
                           for j,v in enumerate(row['placements']['requested'])):
                        raise RuntimeError('Official opponent goals changed')
                    if name!='online' and model.priors.digest()!=before:
                        raise RuntimeError('Frozen memory changed')
                    offers=row['placements']['delivered'];bins=dict(low=0,low_returned=0,ordinary=0,ordinary_returned=0)
                    for offer in offers:
                        cross=offer.get('receive_line')
                        if cross is None:continue
                        low=projected_height(cross)<1.
                        bins['low' if low else 'ordinary']+=1
                        bins['low_returned' if low else 'ordinary_returned']+=int(offer['near_landed'])
                    row['evaluator_only_height_bins']=bins
                    paired['arms'][name]=row
                    result['rows']=result['rows'][:len(result['rows'])]+[]
                    result.update(wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
                    _write(a.out,dict(result,in_progress=paired))
                    print(c['id'],name,row['score']['learner'],bins,flush=True)
                result['rows'].append(paired)
                result.update(wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu,
                    actual_control_ticks=sum(len(r['actions']) for x in result['rows'] for r in x['arms'].values()))
                _write(a.out,result)
        finally:
            if stock:stock.close()
            for model in models.values():model.close()
            res[0].close();arena.close()
    result['complete']=True;_write(a.out,result)


def replay(source,output):
    d=_read(source);rows=[]
    with mpc.centered(d['ready']):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        try:
            for c,r in zip(d['cards'],d['rows'],strict=True):
                requests=r['placements']['requested']
                def draw(card,leg,protocol,rng):
                    v=requests[leg];return tuple(v['target']),v['draw']
                with patch.object(skill.base.opponent,'draw',draw):
                    a.launch(r['shot'],card=c);a.partner.reset(seed=c['seed']);guarded(a,clear)
                    for command in r['actions']:a.exchange(command,mpc.FAR_COMMAND)
                verify_clearance(a)
                now=(digest_json(a.events),digest_json(a.point_placements()),digest_json(audit(a.events,opening_side=1)))
                expected=tuple(digest_json(r[k]) for k in ('events','placements','score'))
                if now!=expected:raise RuntimeError('Issued action replay changed physical result')
                rows.append(dict(id=c['id'],legal_returns=r['score']['learner'],clearance=a.clearance_result(),
                    exact_events_and_goals=True,actual_replayed_control_ticks=len(r['actions'])))
                _write(output,dict(source_sha256=_hash(source),rows=rows,complete=False))
        finally:a.close();cad.close()
    _write(output,dict(source_sha256=_hash(source),rows=rows,complete=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('compare','replay'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--memory',type=Path)
    p.add_argument('--carry',type=Path,default=season.CARRY);p.add_argument('--n',type=int,default=4)
    p.add_argument('--cap',type=int,default=40);p.add_argument('--arms',nargs='+',default=['stock','cold','frozen'])
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    a=p.parse_args();replay(a.memory,a.out) if a.phase=='replay' else run(a)
