"""Actual legal-observation calibration and full-rally analytic output credit."""
import random,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_full_rally_actor import FullRallyActor,probabilities,score
from pong_continuous_visual import canonical_scene,save_scenes
from rally_raw_capture import BOOKKEEPING,SparseRows
import run_practical_pong_v2 as base


class ContinuousActor(FullRallyActor):
    def __init__(self,readout,config,temperature,action_seed):
        self.temperature=temperature;self.actor_rng=np.random.default_rng(action_seed);self.eligibility=np.zeros((2,256));self.credit=self.eligibility.copy();self.actor_rows=[];FreshRBFEvaluator.__init__(self,readout,config)

    def _choose_policy(self):
        f=self.rbf_readout.continuous.features(canonical_scene(self.latest_encoded));greedy,q=self.rbf_readout.choose_features(*policy_features(self.latest_encoded,self.scaler),f);p=probabilities(q,self.temperature);a=int(self.actor_rng.choice(3,p=p));self.eligibility+=score(a,p,f,self.temperature);self.actor_rows.append((a,greedy,q,p));return a,q


def task(directory,model_root,job,deadline,max_cameras,temperature=None):
    directory=Path(directory);model_root=Path(model_root);began=time.monotonic();readout=load_readout(model_root/'rbf-model.npz');digest=base.sha(model_root/'rbf-model.npz');W_sha=base.sha(model_root/'proposal.npy');assert np.array_equal(readout.base.weights,np.load(model_root/'proposal.npy'));random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING);config=PongEvaluationConfig(seed=job['seed'],episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=base.read(model_root/'radial-cap.json'));session=FreshRBFEvaluator(readout,config) if temperature is None else ContinuousActor(readout,config,temperature,job['action_seed']);raw=SparseRows();scenes=[];records=[]
    while not session.complete and session.transitions<max_cameras and time.monotonic()<deadline:
        if session.initial_cameras==session.episodes:session._begin_episode()
        raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    session.verify_frozen();report=session.summary();tag=job['tag'];sp.save_npz(directory/(tag+'-raw.npz'),raw.matrix());save_scenes(directory/(tag+'-scenes.npz'),scenes);q=np.asarray([r['q'] for r in records]);actions=np.asarray([r['action'] for r in records]);hits=np.asarray([r['hits'] for r in records]);payload=dict(q=q,actions=actions,hits=hits)
    if temperature is not None:payload.update(p=np.array([r[3] for r in session.actor_rows]),greedy=np.array([r[1] for r in session.actor_rows]),eligibility=session.eligibility,credit=session.credit)
    np.savez_compressed(directory/(tag+'-scores.npz'),**payload);result=dict(job,complete=report['complete'],allocation_complete=report['complete'] or session.transitions==max_cameras,hits=int(sum(hits)),completed_rally_hits=report['hits'],unfinished_episode_hits=0 if report['complete'] else session.episode_hits,transitions=session.transitions,initials=session.initial_cameras,report=report,model_sha256=digest,base_weights_sha256=W_sha,scenes_file=str(directory/(tag+'-scenes.npz')),raw_feature_file=str(directory/(tag+'-raw.npz')),scores_file=str(directory/(tag+'-scores.npz')),temperature=temperature,parameters_frozen=True,wall_seconds=time.monotonic()-began);assert sum(hits)==report['hits']+(0 if report['complete'] else session.episode_hits) and base.sha(model_root/'rbf-model.npz')==digest and base.sha(model_root/'proposal.npy')==W_sha;base.atomic_json(directory/(tag+'.json'),result)
    return dict(result,scenes=scenes,q=q) if temperature is None else dict(result,eligibility=session.eligibility,credit=session.credit)
