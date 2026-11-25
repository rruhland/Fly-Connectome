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


def run(output):
    bond,known,meta=facts()
    if bond!=rp._read(PLAN) or output.exists():raise ValueError('Changed locked policy or already ran the known point')
    live=rp._read(wing.root/'checked-whole.json.gz')
    if not live.get('complete') or live.get('model_sha256')!=bond['checked_scout_sha256']:
        raise ValueError('Complete the prospective check before running this known game')
    card=bond['card'];arch=known['rows'][0];boot=known['ready']
    with np.load(mpc.BANK) as bank:raw=mpc.DeepField(bank,'local')
    row=dict(plan_sha256=rp._hash(PLAN),started=True,complete=False)
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
            for k,ell in enumerate(play['placements']['requested']):
                if ell['target']!=list(rp.draw(card,k,'random',None)[0]):raise ValueError('Unexpected opponent draw')
            first=next((i for i,(x,y) in enumerate(zip(arch['actions'],play['actions'])) if x!=y),None)
            when=(first if first is not None else min(len(arch['actions']),len(play['actions'])))*mpc.DT
            prefix=[tuple(e for e in x['events'] if e['time']<when-1e-7) for x in (arch,play)]
            if prefix[0]!=prefix[1]:raise ValueError('Changed events before any command change')
            common=[]
            for x,y in zip(arch['placements']['delivered'],play['placements']['delivered']):
                if not all(same_trace(x[k],y[k]) for k in ('far_contact','near_bounce')):break
                common.append([x['near_landed'],y['near_landed']])
            row.update(complete=True,score=play['score']['learner'],original=arch['score']['learner'],first_changed_tick=first,
                       no_imposed_commands=True,common_actual_offers=common,clearance=a.clearance_result(),
                       body=p.body_trace,late_choices=p.repertory_log,forecasts=p.ready.predictor.estimates,run=play)
            rp._write(output,row)
            print(card['id'],row['score'],row['original'],first,len(common),a.clearance_result(),flush=True)
        finally:p.close();camera.close();a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=PHYS);a=p.parse_args()
    if a.phase=='plan':
        if PLAN.exists():raise ValueError('Already locked')
        rp._write(PLAN,facts()[0])
    else:run(a.out)
