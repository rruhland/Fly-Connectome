"""Compact actual-game reward and reliability checks for frozen public policies.

Every live decision is compared with an independently deserialized controller.
This is streaming parity, not full archived replay: only digests, actual event
tapes and bounded miss tails survive. No simulator metric is a policy input.
"""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback
import numpy as np

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_relational_persistent_binding import PersistentRelationalPolicy,PersistentRelationalController
from public_entity_pong_native import snapshot,save_public_entities
from exact_visual_transport_copy import install_exact_transport_copy
from run_rally_option_memory import MODEL,EXPECTED,seed_rngs,sha,read,write
from return_metrics import ReturnCounter,native_count_status

CONTRACT='compact-public-servo-independent-controller-streaming-parity-v1'
FILES=('compact_public_pong.py','public_entity_pong_native.py','exact_visual_transport_copy.py',
    'pong_relational_persistent_adapter.py','pong_relational_persistent_binding.py','pong_relational_feedback.py',
    'stateful_pong_comparison.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_continuous_visual.py',
    'run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py','return_metrics.py',
    'run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')
_BASELINE=None


def fitness_key(model_path,seed,max_episode_ticks,runtime_identity):
    payload=(CONTRACT,None if model_path is None else sha(model_path),int(seed),int(max_episode_ticks),runtime_identity,EXPECTED)
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


class BaselineController:
    def __init__(self):
        self.ordinal=0;self.previous_action=1;self.past_hit=False;self.pending_action=None;self.bound_ids=None
    def choose(self,entities,baseline_action):
        self.pending_action=int(baseline_action)
        return int(baseline_action),dict(mode='literal_baseline',ordinal=self.ordinal,past_hit=self.past_hit,
            previous_actual_action=self.previous_action,action=int(baseline_action))
    def observe(self,hits,natural_terminal=False):
        if self.pending_action is None:raise ValueError('actual choice required before feedback')
        self.previous_action=self.pending_action;self.pending_action=None;self.ordinal+=1;self.past_hit |= bool(hits)


def new_controller(model):
    return BaselineController() if model is None else PersistentRelationalController(model)


def controller_state(controller):
    return dict(ordinal=controller.ordinal,previous_action=controller.previous_action,past_hit=controller.past_hit,
        bound_ids=None if controller.bound_ids is None else list(controller.bound_ids))


def time_bins(events,seconds):
    result=[]
    for index in range(3):
        selected=[e for e in events if min((e['tick']-1)//2400,2)==index]
        first=[e for e in selected if e['first']];later=[e for e in selected if not e['first']]
        result.append(dict(start_seconds=20*index,end_seconds=20*(index+1),
            exposure_seconds=float(np.clip(seconds-20*index,0,20)),
            own_opportunities=len(selected),own_returns=sum(e['hit'] for e in selected),
            first_opportunities=len(first),first_returns=sum(e['hit'] for e in first),
            later_opportunities=len(later),later_returns=sum(e['hit'] for e in later)))
    return result


class CompactEvaluator(FreshRBFEvaluator):
    def __init__(self,baseline,model,shadow,config):
        self.model=model;self.shadow_model=shadow;self.controller=new_controller(model);self.shadow=new_controller(shadow)
        self.digest=hashlib.sha256();self.ring=deque(maxlen=64);self.pending_entry=None;self.checked_rows=0;self.mismatch=None
        super().__init__(baseline,config);self.initial_learned_hash=self.learned_hash()
    def learned_hash(self):
        primary=('literal40069',EXPECTED) if self.model is None else self.model.frozen_parameters()
        shadow=('literal40069',EXPECTED) if self.shadow_model is None else self.shadow_model.frozen_parameters()
        return fingerprint((super().learned_hash(),CONTRACT,primary,shadow))
    def _begin_episode(self):
        self.controller=new_controller(self.model);self.shadow=new_controller(self.shadow_model)
        super()._begin_episode()
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler))
        entities=self.latest_encoded['entities'];owned=snapshot(entities)
        before=controller_state(self.controller)
        action,decision=self.controller.choose(entities,baseline)
        other,shadow_decision=self.shadow.choose(owned,baseline)
        entry=dict(ordinal=self.transitions,public_entities=owned,state_before=before,baseline_q=q.tolist(),
            baseline_action=int(baseline),action=int(action),decision=decision)
        self.ring.append(entry);self.pending_entry=entry
        if action!=other or decision!=shadow_decision:
            self.mismatch=dict(primary_action=action,shadow_action=other,primary=decision,shadow=shadow_decision)
            raise AssertionError('independently loaded public controllers disagree')
        encoded=json.dumps(entry,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.digest.update(struct.pack('<I',len(encoded)));self.digest.update(encoded)
        return int(action),q
    def step(self,**kwargs):
        row=super().step(**kwargs)
        if self.pending_entry is None or row['action']!=self.pending_entry['action']:
            raise AssertionError('executed action differs from streamed choice')
        self.controller.observe(row['hits'],row['terminated']);self.shadow.observe(row['hits'],row['terminated'])
        if controller_state(self.controller)!=controller_state(self.shadow):raise AssertionError('feedback histories disagree')
        event={k:row[k] for k in ('action','hits','elapsed_seconds','physics_ticks','terminated','truncated')}
        self.pending_entry['record']=event;self.pending_entry=None;self.checked_rows+=1
        self.digest.update(json.dumps(event,sort_keys=True,separators=(',',':')).encode());return row


def evaluate(model_path,seed,max_episode_ticks,artifact_directory,deadline_monotonic,max_transitions,runtime_identity):
    global _BASELINE
    if max_episode_ticks not in (2400,7200):raise ValueError('fixed 20- or 60-second duration required')
    if not 0<=max_transitions<=max_episode_ticks*5//12:raise ValueError('camera grant exceeds declared duration')
    began=time.monotonic();directory=Path(artifact_directory);directory.mkdir(parents=True,exist_ok=True)
    model_sha=None if model_path is None else sha(model_path)
    key=fitness_key(model_path,seed,max_episode_ticks,runtime_identity)
    common=dict(seed=int(seed),max_episode_ticks=int(max_episode_ticks),runtime_identity=runtime_identity,
        model_sha256=model_sha,fitness_key=key)
    if max_transitions==0 or time.monotonic()>=deadline_monotonic:
        return dict(common,complete=False,not_started=True,fitness_eligible=False,transitions=0,initial_cameras=0,
            transition_camera_upper_bound=0,initial_camera_upper_bound=0)
    seed_rngs()
    if _BASELINE is None:
        assert sha(MODEL/'rbf-model.npz')==EXPECTED;_BASELINE=load_readout(MODEL/'rbf-model.npz')
    model=None if model_path is None else PersistentRelationalPolicy.load(model_path)
    shadow=None if model_path is None else PersistentRelationalPolicy.load(model_path)
    if model is not None:
        assert fingerprint(model.frozen_parameters())==fingerprint(shadow.frozen_parameters())
        assert not np.shares_memory(model.original.coefficients,shadow.original.coefficients)
        assert not np.shares_memory(model.original.scales,shadow.original.scales)
    session=CompactEvaluator(_BASELINE,model,shadow,PongEvaluationConfig(seed=int(seed),episodes=1,
        max_episode_ticks=int(max_episode_ticks),opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json')))
    install=install_exact_transport_copy(session);counter=ReturnCounter();events=[];rows=[];tail=None
    attempted=0;begin_attempted=False;error=None;reason='complete';frozen=False;miss_pending=False
    game=session.env.game;original=game.step
    def instrument(drive,**kwargs):
        nonlocal miss_pending
        previous=counter.opportunities;returns=counter.returns;tick=int(game.rally_steps[0])+1
        out=counter.step(game,original,drive,**kwargs)
        if counter.opportunities>previous:
            hit=counter.returns>returns;events.append(dict(tick=tick,hit=bool(hit),first=previous==0))
            if not hit:miss_pending=True
        return out
    game.step=instrument
    try:
        if time.monotonic()>=deadline_monotonic:reason='deadline_before_initial'
        else:
            begin_attempted=True;session._begin_episode()
            while not session.complete:
                if session.transitions>=max_transitions:reason='camera_grant';break
                if time.monotonic()>=deadline_monotonic:reason='wall_deadline';break
                attempted+=1;rows.append(session.step(check_frozen=False))
                if miss_pending and tail is None:tail=copy.deepcopy(list(session.ring))
        session.verify_frozen();frozen=True
    except Exception:
        error=traceback.format_exc();reason='worker_failure'
        if tail is None:tail=copy.deepcopy(list(session.ring))
    finally:del game.step
    report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(
        truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')})
    complete=bool(session.complete and not error and frozen)
    model_frozen=model_path is None or sha(model_path)==model_sha
    complete &= model_frozen
    if complete:
        assert native_count_status(annotation,report)['valid_terminal_counts']
        assert len(events)==annotation['own_opportunities'] and sum(e['hit'] for e in events)==report['hits']
    tape=directory/(key+'-events.npz')
    data=dict(action=np.array([r['action'] for r in rows],np.int8),hits=np.array([r['hits'] for r in rows],np.int8),
        elapsed_seconds=np.array([r['elapsed_seconds'] for r in rows]),physics_ticks=np.array([r['physics_ticks'] for r in rows],np.int8),
        terminated=np.array([r['terminated'] for r in rows],bool),truncated=np.array([r['truncated'] for r in rows],bool),
        opportunity_tick=np.array([e['tick'] for e in events],np.int32),opportunity_hit=np.array([e['hit'] for e in events],bool),
        opportunity_first=np.array([e['first'] for e in events],bool))
    np.savez_compressed(tape,**data)
    tail_files=None
    if tail:
        public=directory/(key+'-miss-tail-public.npz');details=directory/(key+'-miss-tail.json')
        save_public_entities(public,[e['public_entities'] for e in tail])
        write(details,[{k:v for k,v in e.items() if k!='public_entities'} for e in tail])
        tail_files=dict(public_file=str(public),public_sha256=sha(public),decision_file=str(details),decision_sha256=sha(details),rows=len(tail))
    first=annotation['first_incoming'];hits=int(report['hits'])
    result=dict(common,contract=CONTRACT,complete=complete,fitness_eligible=complete,fitness=hits if complete else None,
        own_hits=hits,first_return=None if first is None else bool(first['success']),later_hits=max(0,hits-1),
        report=report,annotation=annotation,time_bins=time_bins(events,report['physics_seconds']),
        transitions=session.transitions,initial_cameras=session.initial_cameras,
        transition_camera_upper_bound=max(attempted,session.transitions),initial_camera_upper_bound=max(session.initial_cameras,int(begin_attempted)),
        attempted_transition_cameras=attempted,ambiguous_last_step=attempted>session.transitions,
        stop_reason=reason,error=error,frozen_parameters_checked=frozen,model_frozen=model_frozen,
        tape_file=str(tape),tape_sha256=sha(tape),miss_tail=tail_files,transport_copy=install,
        streaming_audit=dict(passed=error is None,checked_rows=session.checked_rows,completed_transitions=session.transitions,
            independently_deserialized_candidate_models=2 if model is not None else 0,
            actual_public_input_and_decision_digest=session.digest.hexdigest(),
            full_archived_replay=False,public_inputs_retained_only_in_bounded_miss_tails=True),
        baseline_sha256=EXPECTED,wall_seconds=time.monotonic()-began)
    receipt=directory/(key+'.json');write(receipt,result);result['receipt_file']=str(receipt);return result
