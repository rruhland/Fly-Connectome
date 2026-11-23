"""Validate each live wrist motor candidate on the next untrained release."""

import numpy as np
import torch

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_online_tip_reservoir import physical_setup
from .tt24_pong_arm_local import WINDOW,windows
from .tt24_rolling_motor import RollingMotorLearner
from .tt24_rolling_motor_wrist import CAP,OUTPUT as UNGATED,ReceivingPort,recovery,verify_raw
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt4_transfer_trial import _hash,_read,_write


OUTPUT=ROOT/'rolling-motor-wrist-guard.json.gz'
RESPONSE_FILTER=.74
MIN_WINDOWS=8
MIN_RELATIVE_SINGULAR=.05
MIN_IMPROVEMENT_RAD=.00045
MAX_ERROR_RATIO=.90


def assessment(observed,active,candidate):
    """Score two fixed predictors on one next-release public trajectory."""
    data=dict(commands=observed['requested'],encoders=observed['encoders'])
    if len(data['commands'])<WINDOW:
        return dict(accepted=False,reason='insufficient_windows',windows=0)
    x,y,usable=windows([data],RESPONSE_FILTER)[0]
    eligible=np.flatnonzero(usable[::WINDOW].all(axis=1))*WINDOW
    x,y=x[eligible],y[eligible]
    singular=np.linalg.svd(x,compute_uv=False)
    relative=float(singular[-1]/singular[0]) if len(singular)==5 and singular[0]>0 else 0.
    detail=dict(windows=len(x),relative_singular=relative,singular_values=singular.tolist())
    if len(x)<MIN_WINDOWS or relative<MIN_RELATIVE_SINGULAR:
        return detail|dict(accepted=False,reason='insufficient_excitation')
    former=float(np.median(np.sqrt(np.mean((y-x@active.T)**2,axis=1))))
    pending=float(np.median(np.sqrt(np.mean((y-x@candidate.T)**2,axis=1))))
    accepted=pending<=MAX_ERROR_RATIO*former and former-pending>=MIN_IMPROVEMENT_RAD
    return detail|dict(accepted=bool(accepted),reason='predictive_gain' if accepted else 'insufficient_gain',
                       active_encoder_error_rad=former,candidate_encoder_error_rad=pending)


def same_physics(a,b):
    return all(digest_json(a[k])==digest_json(b[k]) for k in ('actions','metrics','changes_reached'))


def native_release(server,plan):
    server.reset(seed=plan['launcher_seed'],split='held',render=False)
    return [v.copy() for v in (server._ball_position,server._ball_velocity,server._ball_spin)]


def seal(arm,wrist,drive,server,wiring,prior,block,report):
    active=np.asarray(block['practice'][-1]['active_after'])
    original=np.asarray(prior['source_map'])
    original_recall=np.array_equal(active,original)
    for archived in prior['held'][len(block['held']):]:
        plan=archived['plan']
        frozen=archived['conditions']['frozen_source']
        gated=(frozen if original_recall else verify_raw(arm,wrist,native_release(server,plan),
                   'guarded held',ReceivingPort(wrist,active,wiring),plan))
        block['held'].append(dict(plan=plan,guarded=gated,frozen=frozen,reused_frozen=original_recall))
        _write(OUTPUT,report)
        print('guarded held',drive,plan['learner_seed'],
              gated['metrics']['longest_consecutive_rebounds'],frozen['metrics']['longest_consecutive_rebounds'],
              original_recall,flush=True)
    score=lambda role: [r[role]['metrics']['longest_consecutive_rebounds'] for r in block['held']]
    observed=[row['observed'] for row in block['practice']]
    guarded,frozen=np.array(score('guarded')),np.array(score('frozen'))
    block['summary']=dict(accepted=sum(row['check'].get('accepted',False) for row in block['practice']),
                          guarding_retained_original=bool(original_recall),
                          held_guarded_mean=float(guarded.mean()),held_frozen_mean=float(frozen.mean()),
                          paired_wins=int((guarded>frozen).sum()),paired_losses=int((guarded<frozen).sum()),
                          exposed_samples=sum(len(o['requested']) for o in observed),
                          prep_seconds=sum(o['prep_ticks'] for o in observed)/60.,
                          play_seconds=sum(r['played']['observed_seconds'] for r in block['practice']),
                          last_fit_pending=True)
    _write(OUTPUT,report)


def run_drive(index,base,report):
    prior=base['drives'][str(index)]
    path=ROOT/prior['source_checkpoint']
    if _hash(path)!=prior['source_sha256']:
        raise ValueError('source map changed since the original course')
    learner=RollingMotorLearner.load(path)
    original=learner.map.copy()
    block=report['drives'].setdefault(str(index),dict(practice=[],held=[]))
    learner=recovery(learner,block['practice'])
    saved=block['practice'][-1] if block['practice'] else None
    active=original.copy() if saved is None else np.asarray(saved['active_after'])
    arm=InstalledArm('alternate')
    arm.arena.max_steps=CAP
    server=BounceArena(view='none')
    wiring=physical_setup(index)
    wrist=ToolEquilibrium(BUNDLE,.0557)
    wrist.mode='workspace_local'
    try:
        for ordinal,archived in enumerate(prior['practice'][len(block['practice']):],start=len(block['practice'])):
            plan=archived['plan']
            release=native_release(server,plan)
            pending=None if ordinal==0 else learner.map.copy()
            originating=block['practice'][-1] if ordinal>0 else None
            prefix=digest_json([p['observed'] for p in block['practice']])
            if originating is not None and (originating['fit_history_sha256']!=prefix or
                    not np.array_equal(originating['map_after'],pending)):
                raise ValueError('pending fit does not belong to the guarded stream')
            port=ReceivingPort(wrist,active,wiring,events=learner)
            played=verify_raw(arm,wrist,release,'guarded practice',port,plan)
            tracked=dict(requested=port.requests,encoders=port.encoders,prep_ticks=plan['prep_ticks'],
                         play_ticks=len(port.requests)-plan['prep_ticks'])
            if len(tracked['requested'])!=len(played['actions']) or len(tracked['encoders'])!=len(played['actions'])+1:
                raise ValueError('guarded branch dropped an actual public transition')
            replayed_frozen=np.array_equal(active,original)
            if replayed_frozen and not same_physics(played,archived['conditions']['frozen_source']):
                raise ValueError('identical active map failed to reproduce the earlier frozen trajectory')
            check=dict(accepted=False,reason='no_previous_candidate')
            before=active.copy()
            if pending is not None and originating['refit']:
                check=assessment(tracked,active,pending)
                check.update(trained_through=ordinal,trained_prefix_sha256=prefix,
                             validation_seed=plan['learner_seed'])
                if check['accepted']:
                    active=pending.copy()
            fitted=learner.end()
            own_history=[row['observed'] for row in block['practice']]+[tracked]
            block['practice'].append(dict(plan=plan,observed=tracked,played=played,
                active_before=before.tolist(),active_after=active.tolist(),map_after=learner.map.tolist(),
                refit=bool(fitted),fit_history_sha256=digest_json(own_history),
                fit_history_samples=sum(len(o['requested']) for o in own_history),
                check=check,identical_to_frozen_tape=bool(replayed_frozen)))
            _write(OUTPUT,report)
            print('guard boundary',index,ordinal+1,check,flush=True)
        seal(arm,wrist,index,server,wiring,prior,block,report)
    finally:
        server.close()
        arm.arena.close()


def main():
    torch.set_num_threads(1)
    base=_read(UNGATED)
    report=_read(OUTPUT) if OUTPUT.exists() else dict(ungated_sha256=_hash(UNGATED),
            bundle_sha256=_hash(BUNDLE),independent_input_response=True,drives={})
    if report['ungated_sha256']!=_hash(UNGATED) or report['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('the wrist comparison fixture changed')
    for index in (0,1):
        if 'summary' not in report['drives'].get(str(index),{}):
            run_drive(index,base,report)
    _write(OUTPUT,report)
    print({key:d['summary'] for key,d in report['drives'].items()},flush=True)


if __name__=='__main__':
    main()
