"""Reproduce preassigned complete points, with strict CAD and event-prefix checks."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt51_partial_finish as pf
from . import tt51_partial_live as whole
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, guarded
from .tt40_long_low import same_trace


def run(path, output):
    prior=pf.prior; mpc=pf.mpc
    d=prior._read(path); reg=prior._read(whole.WHOLE)
    if not d.get('complete') or d['cards']!=reg['cards'][:len(d['rows'])]:
        raise ValueError('Need a complete assigned native shootout')
    boot=prior._read(prior.ARCHIVE)['ready']
    receipt=dict(source_sha256=prior._hash(path),predictor_sha256=d['predictor_sha256'],ready=boot,rows=[])
    with mpc.centered(boot):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(detail='cad',width=64,height=48)
        guard=CourtClearance(cad.model,articulated=True)
        try:
            for c,row in zip(d['cards'],d['rows'],strict=True):
                stock,test=(row['runs'][m] for m in ('stock','partial-finish'))
                first=next((i for i,(u,v) in enumerate(zip(stock['actions'],test['actions'])) if u!=v),None)
                when=len(stock['actions']) if first is None else first
                prefix=[tuple(e for e in r['events'] if e['time']<when*mpc.DT-1e-7) for r in (stock,test)]
                if prefix[0]!=prefix[1] or stock['memory']['before']!=test['memory']['before']:
                    raise ValueError('Variants were not matched before the first changed arm action')
                offers=[r['placements']['delivered'] for r in (stock,test)]
                common=0
                for x,y in zip(*offers):
                    if not all(same_trace(x[k],y[k]) for k in ('far_contact','near_bounce')):break
                    common+=1
                pair=dict(id=c['id'],first_changed_action=first,common_native_offers=common,
                          common_prefix_events=len(prefix[0]),modes={})
                for name,r in row['runs'].items():
                    a.launch(r['shot'],card=c);a.partner.reset(seed=c['seed']);guarded(a,guard)
                    for u in r['actions']:a.exchange(u,mpc.FAR_COMMAND)
                    if digest_json(a.events)!=digest_json(r['events']) or not same_trace(a.point_placements(),r['placements']):
                        raise ValueError('Strict native replay diverged')
                    req=a.point_placements()['requested']
                    if any(p['target']!=list(draw(c,j,'random',None)[0]) for j,p in enumerate(req)):
                        raise ValueError('Replay changed the TT37 request stream')
                    pair['modes'][name]=dict(own=r['score']['learner'],opponent=r['score']['opponent'],
                          physical_events_sha256=digest_json(a.events),strict_clearance=a.clearance_result(),
                          camera_waveform_updates=r['optical_updates'],far_forecasts=len(r['observed_future_forecasts']))
                receipt['rows'].append(pair)
                prior._write(output,receipt)
                print(c['id'],first,common,[(m,v['own'],v['opponent'],v['strict_clearance']['penetration_samples'])
                       for m,v in pair['modes'].items()],flush=True)
        finally:
            a.close();cad.close()
    receipt['complete']=True;prior._write(output,receipt)


def isolate(path, ident, output, predictor):
    """On one already known failure, remove only the added strike choices."""
    prior=pf.prior;mpc=pf.mpc
    data=prior._read(path);source=prior._read(prior.ARCHIVE)
    if data['predictor_sha256']!=prior._hash(predictor):raise ValueError('Changed the applied forecaster')
    original=next(r for r in data['rows'] if r['id']==ident)['runs']['partial-finish']
    c=next(c for c in data['cards'] if c['id']==ident)
    if original['score']['learner']!=1:raise ValueError('This diagnostic is scoped to the first carried reply')
    boot=source['ready']; base={**source,'rows':[None]*len(prior._read(prior.season.CARRY)['rows'])+source['rows']}
    predict=whole.logged_predictor(predictor)
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad')
        cam,rig,stride,lo,hi=mpc.resources(a);tap=whole.Tap(cam)
        checker=prior.Reach(cad,boot,boot)
        with patch.object(mpc,'Fragments',lambda f,e:whole.Tail(f,e,practice=[prior.course.PRACTICE],held={ident})):
            p=whole.Arm(mpc.BANK,raw,mode='partial',finish=pf.Finish(),checker=checker,predictor=predict,
                        coverage=prior.fresh.shot.Coverage(prior._read(prior.fresh.RISK)),
                        budget=source['budget'],variant=source['variant'])
        p.tap=tap;p.priors.owner=p;p.priors.original_skills=0;p.priors.serial_skills=0
        try:
            shots=mpc.measured_launches(Path('/tmp/tt25-rallies.json'))
            prior.course.resume(p,prior.BANK,base)
            r=mpc.play(a,tap,rig,stride,lo,hi,boot,c,shots[ident],raw,p,prior.course.OPTIC,'live',5)
        finally:
            p.close();cam.close();a.close();cad.close()
    match=all(same_trace(r['placements']['delivered'][1][key],original['placements']['delivered'][1][key])
              for key in ('far_contact','near_bounce'))
    ix=next((i for i,(u,v) in enumerate(zip(original['actions'],r['actions'])) if u!=v),None)
    choice=next(t for t in original['followthrough'] if t.get('model_support',0)>0)
    if not match or ix is None or ix<choice['tick'] or r['memory']['before']!=original['memory']['before']:
        raise ValueError('Removed more than the chosen learned finishing move')
    d=dict(role='Training diagnosis after the paired outcome, not a fresh point result',id=ident,source_sha256=prior._hash(path),
           predictor_sha256=prior._hash(predictor),native_offer_matched=True,
           same_controls_until_learned_choice=ix,original_finishing_decision=choice,observation_limited_to_one_carried_receive=True,
           legal_after_disabling=bool(r['placements']['delivered'][1]['near_landed']),replayed=r)
    prior._write(output,d)
    print(ident,ix,d['legal_after_disabling'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=whole.OUT)
    p.add_argument('--output',type=Path)
    p.add_argument('--isolate',type=int)
    p.add_argument('--predictor',type=Path,default=pf.prior.VISION_ROOT/'risk-model.npz')
    a=p.parse_args()
    if a.isolate is None:run(a.source,a.output or pf.root/'whole-audit.json')
    else:isolate(a.source,a.isolate,a.output or pf.root/f'isolated-{a.isolate}.json.gz',a.predictor)
