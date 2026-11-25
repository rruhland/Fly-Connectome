"""Run one untouched checked-scout controller from the authenticated before-63 boundary."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_live_wing_v3 as model
from .tt37_opponent_audit import CourtClearance,guarded
from .tt40_long_low import same_trace

wing=model.first;rp=wing.rp;mpc=rp.mpc;prior=rp.pf.prior
SOURCE=Path('artifacts/tt48_transfer/random-63-tape.json.gz')
BEFORE=Path('artifacts/tt51_point_transfer/after-09.json.gz')
PHYS=wing.root/'checked-record.json.gz'
PLAN=wing.root/'checked-record-plan.json'
NPZ=BEFORE.with_suffix('').with_suffix('.npz')


def facts():
    archived=rp._read(SOURCE);meta=rp._read(BEFORE)
    assert len(archived['cards'])==len(archived['rows'])==1 and len(meta['rows'])==9
    card=archived['cards'][0];r=archived['rows'][0]
    if meta['checkpoint_sha256']!=rp._hash(NPZ) or r['memory']['before']!=meta['rows'][-1]['memory']['after']:
        raise ValueError('Expected authenticated natural point-start experience')
    bond=dict(card=card,reference_source_sha256=rp._hash(SOURCE),nine_receipt_sha256=rp._hash(BEFORE),
              numeric_initial_bank_sha256=rp._hash(NPZ),prior_memory=r['memory']['before'],
              checked_scout_sha256=rp._hash(Path(model.__file__)),base_live_body_sha256=rp._hash(Path(wing.__file__)),
              bank_sha256=rp._hash(wing.BANK),vision_sha256=rp._hash(rp.risk_model),
              late_physical_vocabulary_sha256=rp._hash(rp.low_artifact),reference_own=r['score']['learner'],
              rule='Known benchmark, not fresh. Only one full normal-launch run following this lock. No imposed prefix commands, native-cue switch or post-result retry.')
    return bond,archived,meta


def different(a,b,key='event',atol=0.):
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for k in a:
            d=different(a[k],b[k],f'{key}.{k}',atol)
            if d is not None:return d
        return None
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b,strict=True)):
            d=different(x,y,f'{key}[{i}]',atol)
            if d is not None:return d
        return None
    if a==b:return None
    row=dict(field=key,original=a,recovered=b)
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)):
        row['numerical_offset']=float(b-a)
        if abs(b-a)<=atol:return None
    return row


def run(output,recover=False):
    bond,known,meta=facts()
    if bond!=rp._read(PLAN):raise ValueError('Changed locked policy or native experiment')
    plan_sha=rp._hash(PLAN)
    if recover:
        before=rp._read(output)
        if before!=dict(plan_sha256=plan_sha,started=True,complete=False):
            raise ValueError('Recovery requires the sole report failure with no persisted raw outcomes')
    elif output.exists():raise ValueError('Already ran the known point')
    live=rp._read(wing.root/'checked-whole.json.gz')
    if not live.get('complete') or live.get('model_sha256')!=bond['checked_scout_sha256']:
        raise ValueError('Complete the prospective check before running this known game')
    card=bond['card'];arch=known['rows'][0];boot=known['ready']
    with np.load(mpc.BANK) as bank:raw=mpc.DeepField(bank,'local')
    row=dict(plan_sha256=plan_sha,started=True,complete=False,deterministic_recorder_recovery=recover)
    rp._write(output,row)
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad'); camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera)
        check=prior.Reach(cad,boot,boot)
        with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={card['id']})):
            p=model.install(raw,check,meta,enabled=True,frames=tap)
        try:
            prior.course.resume(p,NPZ,{**meta,'rows':[None]*meta['parent_serial']+meta['rows']})
            if p.priors.digest()!=bond['prior_memory']:raise ValueError('Different point-start motor bank')
            guarded(a,CourtClearance(cad.model,articulated=True))
            play=mpc.play(a,tap,rig,stride,lo,hi,boot,card,arch['shot'],raw,p,prior.course.OPTIC,'live',240,10)
            row.update(score=play['score']['learner'],original=arch['score']['learner'],
                       no_imposed_commands=True,clearance=a.clearance_result(),
                       body=p.body_trace,late_choices=p.repertory_log,forecasts=p.ready.predictor.estimates,run=play)
            rp._write(output,row)
            for k,ell in enumerate(play['placements']['requested']):
                if ell['target']!=list(rp.draw(card,k,'random',None)[0]):raise ValueError('Unexpected opponent draw')
            first=next((i for i,(x,y) in enumerate(zip(arch['actions'],play['actions'])) if x!=y),None)
            when=(first if first is not None else min(len(arch['actions']),len(play['actions'])))*mpc.DT
            prefix=[tuple(e for e in x['events'] if e['time']<when-1e-7) for x in (arch,play)]
            diffs=[]
            for index,(x,y) in enumerate(zip(*prefix)):
                if x==y:continue
                f=different(x,y)
                diffs.append(dict(event=index,event_kind=x.get('kind'),original_time=x['time'],recovered_time=y['time'],
                                  first_different_field=f,within_roundoff=same_trace(x,y),
                                  first_beyond_roundoff=different(x,y,atol=1e-8)))
            prefix_exact=prefix[0]==prefix[1]
            prefix_numeric=(len(prefix[0])==len(prefix[1]) and all(x['within_roundoff'] for x in diffs))
            common=[]
            for x,y in zip(arch['placements']['delivered'],play['placements']['delivered']):
                if not all(same_trace(x[k],y[k]) for k in ('far_contact','near_bounce')):break
                common.append([x['near_landed'],y['near_landed']])
            row.update(complete=True,score=play['score']['learner'],original=arch['score']['learner'],first_changed_tick=first,
                       no_imposed_commands=True,common_actual_offers=common,clearance=a.clearance_result(),
                       event_prefix_exact=prefix_exact,event_prefix_within_roundoff=prefix_numeric,
                       event_prefix_sizes=[len(x) for x in prefix],event_prefix_first_offsets=diffs[:8],
                       body=p.body_trace,late_choices=p.repertory_log,forecasts=p.ready.predictor.estimates,run=play)
            rp._write(output,row)
            print(card['id'],row['score'],row['original'],first,len(common),prefix_exact,prefix_numeric,a.clearance_result(),flush=True)
        finally:p.close();camera.close();a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=PHYS);p.add_argument('--recover-audit-failure',action='store_true');a=p.parse_args()
    if a.phase=='plan':
        if PLAN.exists():raise ValueError('Already locked')
        rp._write(PLAN,facts()[0])
    else:run(a.out,recover=a.recover_audit_failure)
