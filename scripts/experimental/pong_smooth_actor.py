"""Optimizer-only smooth corrections to existing raw W; nonlinear R stays fixed."""
import random,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from fly_connectome.control.codec import fingerprint
from pong_full_rally_actor import FullRallyActor,probabilities,score
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_constrained_residual import CONTRAST
from policy_search_math import spatial_basis
from rally_raw_capture import BOOKKEEPING,SparseRows
import run_practical_pong_v2 as base


def correction_basis(layout,dimension):
    matrix,_=spatial_basis(layout,dimension,3);grid=np.zeros(dimension,dtype=bool)
    for name in ('current','future','relations','context'):
        start,end=layout[name];grid[start:end]=True
    matrix=matrix[np.any(matrix[:,grid]!=0,axis=1)];assert matrix.shape==(198,4389) and not np.any(matrix[:,~grid]);np.testing.assert_allclose(matrix@matrix.T,np.eye(198),atol=2e-15);return matrix


def projected_metric(G,B,raw_denom):
    J=B*np.asarray(raw_denom)[None,:];return J@G[:B.shape[1],:B.shape[1]]@J.T


def mapped_weights(reference,coefficients,B):
    return reference.base.weights+CONTRAST@coefficients@B


class SmoothActor(FullRallyActor):
    def __init__(self,readout,config,temperature,action_seed,B):
        self.temperature=temperature;self.actor_action_seed=action_seed;self.actor_rng=np.random.default_rng(action_seed);self.correction_basis=B;self.eligibility=np.zeros((2,len(B)));self.credit=self.eligibility.copy();self.actor_rows=[];FreshRBFEvaluator.__init__(self,readout,config)

    def learned_hash(self):return fingerprint((super().learned_hash(),self.correction_basis,self.temperature,self.actor_action_seed))

    def _choose_policy(self):
        ix,x=policy_features(self.latest_encoded,self.scaler);greedy,q=self.rbf_readout.choose(ix,x);p=probabilities(q,self.temperature);action=int(self.actor_rng.choice(3,p=p));f=self.correction_basis[:,ix]@x;self.eligibility+=score(action,p,f,self.temperature);self.actor_rows.append((action,greedy,q,p));return action,q


def task(directory,model_root,job,temperature,B,deadline,max_cameras):
    began=time.monotonic();directory=Path(directory);root=Path(model_root);policy=load_readout(root/'rbf-model.npz');model_sha=base.sha(root/'rbf-model.npz');W_sha=base.sha(root/'proposal.npy');assert np.array_equal(policy.base.weights,np.load(root/'proposal.npy'));random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING);B.flags.writeable=False
    config=PongEvaluationConfig(seed=job['seed'],episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=base.read(root/'radial-cap.json'));session=SmoothActor(policy,config,temperature,job['action_seed'],B);raw=SparseRows();hits=[]
    while not session.complete and session.transitions<max_cameras and time.monotonic()<deadline:
        if session.initial_cameras==session.episodes:session._begin_episode()
        raw.add(session);hits.append(session.step(check_frozen=False)['hits'])
    session.verify_frozen();report=session.summary();rows=session.actor_rows;tag=f"batch_{job['batch']:02d}_episode_{job['episode']}";sp.save_npz(directory/(tag+'-raw.npz'),raw.matrix());np.savez_compressed(directory/(tag+'-scores.npz'),actions=np.array([r[0] for r in rows]),greedy=np.array([r[1] for r in rows]),q=np.array([r[2] for r in rows]),p=np.array([r[3] for r in rows]),hits=hits,eligibility=session.eligibility,credit=session.credit)
    result=dict(job,complete=report['complete'],hits=report['hits'],transitions=session.transitions,initials=session.initial_cameras,temperature=temperature,report=report,model_sha256=model_sha,base_weights_sha256=W_sha,raw_feature_file=str(directory/(tag+'-raw.npz')),scores_file=str(directory/(tag+'-scores.npz')),eligibility_shape=list(session.eligibility.shape),nongreedy_choices=sum(r[0]!=r[1] for r in rows),expected_nongreedy_choices=sum(1-r[3][r[1]] for r in rows),parameters_frozen=True,wall_seconds=time.monotonic()-began)
    assert report['hits']==sum(hits) and base.sha(root/'rbf-model.npz')==model_sha and base.sha(root/'proposal.npy')==W_sha;base.atomic_json(directory/(tag+'.json'),result);return dict(result,eligibility=session.eligibility,credit=session.credit)
