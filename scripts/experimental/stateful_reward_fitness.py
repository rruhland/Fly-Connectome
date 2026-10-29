"""Reusable full-game scalar reward fitness, without challenger raw archives."""
import hashlib
import importlib
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
from pong_continuous_visual import canonical_scene
from run_rally_option_memory import MODEL,EXPECTED,seed_rngs,sha,write,read

CONTRACT='stateful-legal-scene-fullgame-ownhit-fitness-v1'
_BASELINE=None


class CameraBudget:
    def __init__(self,limit):
        self.limit=limit;self.used=0;self.reserved=0;self.initials=0
    def reserve(self,grant,limit=None):
        bound=self.limit if limit is None else min(self.limit,limit)
        if self.used+self.reserved+grant>bound:return False
        self.reserved+=grant;return True
    def finish(self,grant,result):
        self.reserved-=grant
        self.used+=result.get('transition_camera_upper_bound',result['transitions'])
        self.initials+=result.get('initial_camera_upper_bound',result['initial_cameras'])
    def state(self):
        return dict(transition_upper_bound=self.used,inflight_transition_grants=self.reserved,
            initial_camera_upper_bound=self.initials,transition_limit=self.limit)


def fitness_key(model_hash,seed,identity):
    payload=(CONTRACT,model_hash,int(seed),identity,EXPECTED,2400,.65,1)
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


class RewardEvaluator(FreshRBFEvaluator):
    def __init__(self,baseline,model,adapter,config):
        self.model=model;self.adapter=adapter;self.controller=adapter.new_controller(model)
        self.digest=hashlib.sha256();self.past_hit=False
        super().__init__(baseline,config);self.initial_learned_hash=self.learned_hash()
    def learned_hash(self):
        return fingerprint((super().learned_hash(),CONTRACT,self.adapter.CONTRACT,self.model.frozen_parameters()))
    def _begin_episode(self):
        self.controller=self.adapter.new_controller(self.model);self.past_hit=False
        super()._begin_episode()
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler))
        scene=canonical_scene(self.latest_encoded)
        action,decision=self.adapter.choose(self.controller,scene,baseline,q)
        if not np.isin(action,(0,1,2)) or (not self.past_hit and action!=baseline):
            raise ValueError('legal command and exact baseline before actual own hit required')
        self.digest.update(struct.pack('<q?qq',self.transitions,self.past_hit,int(action),len(scene)))
        self.digest.update(np.asarray(scene,dtype='<f8').tobytes())
        self.digest.update(np.asarray(q,dtype='<f8').tobytes())
        self.digest.update(json.dumps(decision,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        return int(action),q
    def step(self,**kwargs):
        row=super().step(**kwargs);self.adapter.observe(self.controller,row)
        self.past_hit |= bool(row['hits']);return row


def evaluate(adapter_name,payload,seed,directory,deadline,max_transitions,identity):
    global _BASELINE
    adapter=importlib.import_module(adapter_name);model=adapter.from_payload(payload)
    model_hash=adapter.model_hash(payload);key=fitness_key(model_hash,seed,identity)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    if not 0<=max_transitions<=1000:raise ValueError('at most1000 transition cameras per actual game')
    if max_transitions==0 or time.monotonic()>=deadline:
        return dict(seed=seed,complete=False,not_started=True,fitness_eligible=False,transitions=0,
            initial_cameras=0,transition_camera_upper_bound=0,initial_camera_upper_bound=0,model_hash=model_hash,fitness_key=key)
    seed_rngs()
    if _BASELINE is None:
        assert sha(MODEL/'rbf-model.npz')==EXPECTED;_BASELINE=load_readout(MODEL/'rbf-model.npz')
    session=RewardEvaluator(_BASELINE,model,adapter,PongEvaluationConfig(seed=int(seed),episodes=1,
        max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json')))
    rows=[];attempted=0;begin_attempted=False;error=None;stop='complete';frozen=False
    try:
        if time.monotonic()>=deadline:stop='deadline_before_initial'
        else:
            begin_attempted=True;session._begin_episode()
            while not session.complete:
                if session.transitions>=max_transitions:stop='camera_grant';break
                if time.monotonic()>=deadline:stop='wall_deadline';break
                attempted+=1;rows.append(session.step(check_frozen=False))
        session.verify_frozen();frozen=True
    except Exception:error=traceback.format_exc();stop='worker_failure'
    report=session.summary();complete=bool(session.complete and not error and frozen)
    tape=directory/(key+'-transitions.npz')
    data=dict(action=np.array([r['action'] for r in rows],np.int8),hits=np.array([r['hits'] for r in rows],np.int8),
        elapsed_seconds=np.array([r['elapsed_seconds'] for r in rows]),physics_ticks=np.array([r['physics_ticks'] for r in rows],np.int8),
        terminated=np.array([r['terminated'] for r in rows],bool),truncated=np.array([r['truncated'] for r in rows],bool))
    np.savez_compressed(tape,**data);hits=int(data['hits'].sum())
    result=dict(contract=CONTRACT,adapter=adapter_name,seed=int(seed),model_hash=model_hash,fitness_key=key,
        identity=identity,complete=complete,fitness_eligible=complete,own_hits=hits,later_hits=max(0,hits-1),
        games_with_own_hit=int(hits>0),fitness=hits if complete else None,report=report,
        transitions=session.transitions,initial_cameras=session.initial_cameras,
        transition_camera_upper_bound=max(attempted,session.transitions),
        initial_camera_upper_bound=max(session.initial_cameras,int(begin_attempted)),
        attempted_transition_cameras=attempted,ambiguous_last_step=attempted>session.transitions,
        stop_reason=stop,frozen_parameters_checked=frozen,observations_exact=error is None,
        decision_digest=session.digest.hexdigest(),digest_is_not_replay_audit=True,
        transition_file=str(tape),transition_sha256=sha(tape),baseline_sha256=EXPECTED,
        raw_features_retained=False,scenes_retained=False,error=error,wall_seconds=time.monotonic()-began)
    receipt=directory/(key+'.json');write(receipt,result);result['receipt_file']=str(receipt);return result
