"""Continue the committed rolling motor estimator during ordinary altered-arm wrist play."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm,episode,schedule
from .tt24_joint_pong import WARM
from .tt24_learned_tip_transfer import native_point
from .tt24_motor_bank import I,decoder
from .tt24_online_tip_reservoir import physical_setup
from .tt24_pong_arm import historical_retention
from .tt24_pong_arm_tip import Joystick,plan as native_plan
from .tt24_rolling_motor import RollingMotorLearner
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt4_transfer_trial import _hash,_read,_write


OUTPUT=ROOT/'rolling-motor-wrist-course.json.gz'
PRACTICE,HELD=6,4
CAP=480
ROSTER=286128031
MODES=('adaptive','frozen_source','known_evaluator')


class ReceivingPort:
    """Apply learned motor correction before the simulator applies hidden wiring."""

    def __init__(self,wrist,estimate,installation,events=False):
        self.wrist=wrist
        self.inverse=decoder(estimate)
        self.installation=installation
        self.events=events
        self.requests=[]
        self.encoders=[]
        self.pending=None

    def begin_release(self,obs):
        self.wrist.begin_release(obs)
        self.calibration_index=self.wrist.calibration_index
        if self.events is not False:
            first=obs['joints'].astype(float)
            self.encoders.append(first.tolist())
            self.events.begin(first)

    def measured(self,q):
        if self.events is not False and self.pending is not None:
            self.events.observe(self.pending,q)
            self.requests.append(self.pending.tolist())
            self.encoders.append(np.asarray(q,float).tolist())
            self.pending=None

    def act(self,obs):
        self.measured(obs['joints'])
        command=np.clip(self.inverse@self.wrist.act(obs),-1.,1.)
        if self.events is not False:
            self.pending=command.copy()
        return np.clip(self.installation@command,-1.,1.)


def error(matrix,wiring):
    return float(np.linalg.norm(matrix-wiring)/np.linalg.norm(wiring))


def verify_raw(arm,wrist,release,choice,port,plan):
    row=episode(arm,port,plan,release,'bundle')
    port.measured(arm.arena._observation(False)['joints'])
    raw=episode(arm,port,plan,release,'bundle',replay=row['actions'])
    for key in ('actions','metrics','changes_reached'):
        if digest_json(raw[key])!=digest_json(row[key]):
            raise ValueError(f'{choice} wrist physics changed on replay for {plan["learner_seed"]}')
    return row


def checkpoint(path,learner,source_map,native):
    pending=path.with_name(path.stem+'-tmp.npz')
    learner.save(pending)
    with np.load(pending,allow_pickle=False) as raw:
        values=dict(raw)
    np.savez_compressed(pending,**values,source_map=source_map,original_map=I,
                        native_pong_readout=native,bundle_sha256=np.array(_hash(BUNDLE)))
    pending.replace(path)
    restored=RollingMotorLearner.load(path)
    if (restored.completed!=learner.completed or not np.array_equal(restored.map,learner.map) or
            len(restored.blocks)!=len(learner.blocks)):
        raise ValueError('resuming from the receiver checkpoint changed the motor learner')
    for (q0,a),(q1,b) in zip(restored.blocks,learner.blocks,strict=True):
        if not np.array_equal(q0,q1) or not np.array_equal(a,b):
            raise ValueError('rolling command history changed when saved')
    return restored


def replay_source(arena,index,destination,known):
    """Check that the source game and the frozen context survived the wrist updates."""
    archived=_read(ROOT/f'learned-tip-transfer-{index}.json.gz')
    with np.load(destination,allow_pickle=False) as data:
        native=data['native_pong_readout'].copy()
    joystick=Joystick()
    starts,_=native_plan(joystick,count=18,seed=80412718+index)
    seed,pose=starts[-1]
    prior=archived['clips'][-1]
    now=native_point(arena,joystick,seed,pose,known,native)
    if digest_json(now)!=digest_json(prior):
        raise ValueError('the retained Pong policy changed its physical source play')
    return dict(seed=seed,commands=len(now['commands']),native_returns=int(now['hits']),exact=True)


def recovery(learner,prior):
    """Continue from receipts without redoing any completed physical games."""
    for row in prior:
        trajectory=row['observed']
        learner.begin(trajectory['encoders'][0])
        for u,q in zip(trajectory['requested'],trajectory['encoders'][1:],strict=True):
            learner.observe(u,q)
        learner.end()
        if not np.array_equal(learner.map,row['map_after']):
            raise ValueError('persisted chronological learner state differs on resume')
    return learner


def outcomes(rows):
    report={}
    for label in MODES:
        values=[row['conditions'][label] for row in rows]
        streak=np.array([x['metrics']['longest_consecutive_rebounds'] for x in values])
        report[label]=dict(plays=len(values),mean=float(streak.mean()),max=int(streak.max()),
                           ten=int((streak>=10).sum()),second=sum(r['metrics']['rebounds']>=2 for r in values),
                           observed_seconds=float(sum(r['observed_seconds'] for r in values)),
                           cap_completions=sum(r['right_censored'] for r in values))
    return report


def run_drive(index,port_result,native):
    previous=ROOT/f'rolling-motor-{index}.npz'
    estimate=RollingMotorLearner.load(previous)
    source_map=estimate.map.copy()
    source_completed=estimate.completed
    wiring=physical_setup(index)
    plans=[row for row in schedule(seed=ROSTER+index,per_body=PRACTICE+HELD) if row['body']=='alternate']
    record=port_result['drives'].setdefault(str(index),dict(source_checkpoint=previous.name,
        source_sha256=_hash(previous),source_map=source_map.tolist(),source_map_error=error(source_map,wiring),
        practice=[],held=[]))
    if record['source_sha256']!=_hash(previous):
        raise ValueError('the carried Pong checkpoint changed')
    learner=recovery(estimate,record['practice'])
    arm=InstalledArm('alternate')
    arm.arena.max_steps=CAP
    launcher=BounceArena(view='none')
    native_arena=BounceArena(view='none')
    wrist=ToolEquilibrium(BUNDLE,.0557)
    wrist.mode='workspace_local'

    def launch(schedule_row):
        launcher.reset(seed=schedule_row['launcher_seed'],split='held',render=False)
        return [v.copy() for v in (launcher._ball_position,launcher._ball_velocity,launcher._ball_spin)]

    try:
        for i,protocol in enumerate(plans[len(record['practice']):PRACTICE],start=len(record['practice'])):
            release=launch(protocol)
            matrix=learner.map.copy()
            port=ReceivingPort(wrist,matrix,wiring,events=learner)
            start=perf_counter()
            row=verify_raw(arm,wrist,release,'adaptive',port,protocol)
            if len(row['actions'])!=len(port.requests) or len(port.encoders)!=len(port.requests)+1:
                raise ValueError('the learner did not receive every prep and game transition')
            added_before=learner.completed
            fitted=learner.end()
            fixed=verify_raw(arm,wrist,release,'frozen',ReceivingPort(wrist,source_map,wiring),protocol)
            known=verify_raw(arm,wrist,release,'known',ReceivingPort(wrist,wiring,wiring),protocol)
            conditions=dict(adaptive=row,frozen_source=fixed,known_evaluator=known)
            observed=dict(requested=port.requests,encoders=port.encoders,prep_ticks=protocol['prep_ticks'],
                          play_ticks=len(port.requests)-protocol['prep_ticks'])
            record['practice'].append(dict(plan=protocol,observed=observed,conditions=conditions,
                refit=bool(fitted),map_before_error=error(matrix,wiring),map_after_error=error(learner.map,wiring),
                map_after=learner.map.tolist(),source_blocks_evicted=min(source_completed,
                    max(0,learner.completed-source_completed)),closed_at_end=learner.completed-added_before,
                buffered_blocks=[len(q) for _,q in learner.blocks],wall_s=perf_counter()-start))
            _write(OUTPUT,port_result)
            print('live wrist update',index,i+1,len(port.requests),bool(fitted),
                  round(error(matrix,wiring),3),round(error(learner.map,wiring),3),
                  [r['metrics']['longest_consecutive_rebounds'] for r in conditions.values()],flush=True)
        destination=ROOT/f'rolling-motor-wrist-learner-{index}.npz'
        learner=checkpoint(destination,learner,source_map,native)
        record['adapted_sha256']=_hash(destination)
        record['final_map_error']=error(learner.map,wiring)
        with np.load(destination,allow_pickle=False) as saved:
            if not np.array_equal(saved['source_map'],source_map) or not np.array_equal(saved['original_map'],I):
                raise ValueError('starting motor bank contexts changed during adaptation')
        for protocol in plans[PRACTICE+len(record['held']):]:
            release=launch(protocol)
            conditions={label:verify_raw(arm,wrist,release,label,ReceivingPort(wrist,fit,wiring),protocol)
                        for label,fit in zip(MODES,(learner.map,source_map,wiring),strict=True)}
            record['held'].append(dict(plan=protocol,conditions=conditions))
            _write(OUTPUT,port_result)
            print('unexposed wrist',index,protocol['learner_seed'],
                  [r['metrics']['longest_consecutive_rebounds'] for r in conditions.values()],flush=True)
        if _hash(destination)!=record['adapted_sha256']:
            raise ValueError('unexposed test changed the learned state')
        record['retention']=replay_source(native_arena,index,destination,wiring)
        record['exposure']=dict(prep_s=sum(r['observed']['prep_ticks'] for r in record['practice'])/60.,
                                play_s=sum(r['conditions']['adaptive']['observed_seconds'] for r in record['practice']),
                                total_samples=sum(len(r['observed']['requested']) for r in record['practice']),
                                extra_probe_s=0.)
        record['practice_summary']=outcomes(record['practice'])
        record['held_summary']=outcomes(record['held'])
        _write(OUTPUT,port_result)
    finally:
        launcher.close()
        native_arena.close()
        arm.arena.close()


def main():
    torch.set_num_threads(1)
    report=_read(OUTPUT) if OUTPUT.exists() else dict(bundle_sha256=_hash(BUNDLE),preparation_observed=True,
                    learner='committed RollingMotorLearner',roster=ROSTER,post_release_cap_ticks=CAP,drives={})
    if report['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('carried wrist policy changed since the course began')
    native=np.asarray(_read(WARM)['weights'])
    for index in (0,1):
        if 'held_summary' not in report['drives'].get(str(index),{}):
            run_drive(index,report,native)
    if 'original_stock_retention' not in report:
        report['original_stock_retention']=historical_retention()
    report['verified_target_tapes']=sum(3*(len(r['held'])+len(r['practice'])) for r in report['drives'].values())
    _write(OUTPUT,report)
    print({key:row['held_summary'] for key,row in report['drives'].items()},flush=True)


if __name__=='__main__':
    main()
