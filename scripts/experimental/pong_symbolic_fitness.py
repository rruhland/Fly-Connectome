"""Reward-only complete native games for frozen anonymous symbolic policies.

Challengers retain a compact actual transition tape and decision digest, without
raw feature or scene archives. A digest is provenance, not an exhaustive replay
certificate. Final endpoint qualification captures full legal inputs separately.
"""
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
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_continuous_visual import canonical_scene
from run_rally_option_memory import MODEL, EXPECTED, seed_rngs, sha, write, read

CONTRACT = 'symbolic-native-fullgame-rewardonly-baselineuntilactualhit-v1'
_BASELINE = None


def fitness_key(program_hash, seed, policy_identity):
    value = dict(contract=CONTRACT, program_hash=program_hash, seed=int(seed),
        policy_identity=policy_identity, baseline_sha256=EXPECTED,
        max_episode_ticks=2400, opponent_speed=.65, action_hold_frames=1)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


class RewardOnlyEvaluator(FreshRBFEvaluator):
    def __init__(self, baseline, program, config):
        self.program=program;self.past_hit=False;self.decision_digest=hashlib.sha256()
        super().__init__(baseline,config);self.initial_learned_hash=self.learned_hash()

    def learned_hash(self):
        return fingerprint((super().learned_hash(),CONTRACT,self.program.frozen_parameters()))

    def _begin_episode(self):
        self.past_hit=False
        super()._begin_episode()

    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler))
        scene=canonical_scene(self.latest_encoded)
        if not self.past_hit:
            action,scores=baseline,q
        else:
            action,scores=self.program.choose(scene,q,self.transitions%5,self.past_hit)
        scores=np.asarray(scores,np.float64)
        if not np.isin(action,(0,1,2)) or scores.shape!=(3,) or not np.isfinite(scores).all():
            raise ValueError('symbolic policy must return one legal command and finite three scores')
        self.decision_digest.update(struct.pack('<q?qq',self.transitions,self.past_hit,int(action),len(scene)))
        self.decision_digest.update(np.asarray(scene,dtype='<f8').tobytes())
        self.decision_digest.update(np.asarray(q,dtype='<f8').tobytes())
        self.decision_digest.update(np.asarray(scores,dtype='<f8').tobytes())
        return int(action),scores

    def step(self,**kwargs):
        row=super().step(**kwargs);self.past_hit |= bool(row['hits']);return row


def evaluate(payload,seed,directory,deadline,max_transitions,policy_identity):
    """Pool-compatible job. Only completed exact games have valid fitness."""
    from pong_symbolic_program import Program
    global _BASELINE
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    program=Program.from_dict(payload);program_hash=program.active_hash()
    key=fitness_key(program_hash,seed,policy_identity);began=time.monotonic()
    if max_transitions<0 or max_transitions>1000:
        raise ValueError('game camera grant must be in [0,1000]')
    if max_transitions==0 or time.monotonic()>=deadline:
        return dict(seed=seed,complete=False,not_started=True,fitness_eligible=False,
            transitions=0,initial_cameras=0,transition_camera_upper_bound=0,initial_camera_upper_bound=0,
            program_hash=program_hash,fitness_key=key)
    seed_rngs()
    if _BASELINE is None:
        assert sha(MODEL/'rbf-model.npz')==EXPECTED
        _BASELINE=load_readout(MODEL/'rbf-model.npz')
    session=RewardOnlyEvaluator(_BASELINE,program,PongEvaluationConfig(seed=int(seed),episodes=1,
        max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json')))
    rows=[];attempted=0;begin_attempted=False;error=None;stop='complete';frozen=False
    try:
        if time.monotonic()>=deadline:
            stop='deadline_before_initial'
        else:
            begin_attempted=True;session._begin_episode()
            while not session.complete:
                if session.transitions>=max_transitions:
                    stop='camera_grant';break
                if time.monotonic()>=deadline:
                    stop='wall_deadline';break
                attempted+=1
                rows.append(session.step(check_frozen=False))
        session.verify_frozen();frozen=True
    except Exception:
        error=traceback.format_exc();stop='worker_failure'
    report=session.summary();complete=bool(session.complete and not error and frozen)
    tape=directory/(key+'-transitions.npz')
    arrays=dict(action=np.array([r['action'] for r in rows],np.int8),
        hits=np.array([r['hits'] for r in rows],np.int8),
        elapsed_seconds=np.array([r['elapsed_seconds'] for r in rows],np.float64),
        physics_ticks=np.array([r['physics_ticks'] for r in rows],np.int8),
        terminated=np.array([r['terminated'] for r in rows],bool),
        truncated=np.array([r['truncated'] for r in rows],bool))
    np.savez_compressed(tape,**arrays)
    hits=int(arrays['hits'].sum())
    result=dict(contract=CONTRACT,seed=int(seed),program_hash=program_hash,fitness_key=key,
        policy_identity=policy_identity,complete=complete,fitness_eligible=complete,
        own_hits=hits,games_with_own_hit=int(hits>0),later_hits=max(0,hits-1),
        fitness=hits if complete else None,report=report,transitions=session.transitions,
        initial_cameras=session.initial_cameras,transition_camera_upper_bound=max(attempted,session.transitions),
        initial_camera_upper_bound=max(session.initial_cameras,int(begin_attempted)),
        attempted_transition_cameras=attempted,ambiguous_last_step=attempted>session.transitions,
        stop_reason=stop,frozen_parameters_checked=frozen,observations_exact=error is None,
        decision_digest=session.decision_digest.hexdigest(),digest_is_not_replay_audit=True,
        transition_file=str(tape),transition_sha256=sha(tape),baseline_sha256=EXPECTED,
        raw_features_retained=False,scenes_retained=False,error=error,wall_seconds=time.monotonic()-began)
    receipt=directory/(key+'.json');write(receipt,result);result['receipt_file']=str(receipt)
    return result
