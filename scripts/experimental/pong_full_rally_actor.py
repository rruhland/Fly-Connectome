"""Full-rally own-hit score eligibility on the existing frozen visual map."""
import random,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_constrained_residual import CONTRAST
from rally_raw_capture import BOOKKEEPING,SparseRows
import run_practical_pong_v2 as base


def probabilities(q,temperature):
    assert np.isfinite(temperature) and temperature>0
    z=np.asarray(q,dtype=float);z=(z-np.max(z,axis=-1,keepdims=True))/temperature;p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)


def calibrate_temperature(q,target=.05):
    ties=np.sum(q==np.max(q,axis=1,keepdims=True),axis=1)
    if np.mean(1-1/ties)>=target:raise ValueError('initial exact ties make target exploration rate unattainable')
    greedy=np.argmax(q,axis=1);rows=np.arange(len(q));lo=np.finfo(float).tiny;hi=max(float(np.ptp(q,axis=1).max()),1e-12)
    while np.mean(1-probabilities(q,hi)[rows,greedy])<target:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if np.mean(1-probabilities(q,mid)[rows,greedy])<target:lo=mid
        else:hi=mid
    p=probabilities(q,hi);actual=float(np.mean(1-p[rows,greedy]));assert abs(actual-target)<1e-10;return hi,actual


def score(action,p,f,temperature):
    residual=-np.asarray(p).copy();residual[action]+=1
    return np.outer(residual@CONTRAST,f)/temperature


def paired_gradient(episodes):
    assert len(episodes)==4 and all(e['complete'] for e in episodes)
    assert episodes[0]['seed']==episodes[1]['seed'] and episodes[2]['seed']==episodes[3]['seed']
    partner=[1,0,3,2]
    return sum(e['credit']-episodes[partner[i]]['hits']*e['eligibility'] for i,e in enumerate(episodes))/4


def capped_step(gradient,L,F,q,temperature,initial_q_rms):
    direction=la.cho_solve((L,True),gradient.T,check_finite=False).T;dq=(F@direction.T)@CONTRAST.T
    raw=float(np.sqrt(np.mean(dq*dq)));cap=.01*initial_q_rms;p=probabilities(q,temperature)
    scale=min(1.,cap/raw) if raw else 0.
    def tv(s):return float(np.mean(np.sum(abs(probabilities(q+s*dq,temperature)-p),axis=1)/2))
    if tv(scale)>.01:
        lo=0.;hi=scale
        for _ in range(50):
            mid=(lo+hi)/2
            if tv(mid)>.01:hi=mid
            else:lo=mid
        scale=lo
    delta=direction*scale;change=dq*scale
    return delta,dict(gradient_norm=float(np.linalg.norm(gradient)),preconditioned_function_rms=raw,initial_centered_q_rms=initial_q_rms,fixed_function_rms_cap=cap,applied_function_rms=float(np.sqrt(np.mean(change*change))),probability_TV=tv(scale),fixed_probability_TV_cap=.01,scale=scale,calibration_greedy_action_changes=int(np.count_nonzero(np.argmax(q+change,axis=1)!=np.argmax(q,axis=1))),calibration_rows=len(q),nonzero_update=bool(np.any(delta)))


class FullRallyActor(FreshRBFEvaluator):
    def __init__(self,readout,config,temperature,action_seed,raw_denom,phi_denom):
        self.temperature=temperature;self.actor_rng=np.random.default_rng(action_seed);self.raw_denom=raw_denom;self.phi_denom=phi_denom;self.eligibility=np.zeros((2,len(raw_denom)+readout.basis.output_dim));self.credit=self.eligibility.copy();self.actor_rows=[]
        super().__init__(readout,config)

    def _choose_policy(self):
        ix,x=policy_features(self.latest_encoded,self.scaler);greedy,q=self.rbf_readout.choose(ix,x);p=probabilities(q,self.temperature);action=int(self.actor_rng.choice(3,p=p));f=np.zeros(self.eligibility.shape[1]);f[ix]=x/self.raw_denom[ix];f[len(self.raw_denom):]=self.rbf_readout.basis.features(ix,x)/self.phi_denom;self.eligibility+=score(action,p,f,self.temperature);self.actor_rows.append((action,greedy,q,p));return action,q

    def step(self,*,check_frozen=True):
        row=super().step(check_frozen=check_frozen)
        # The eligibility at a hit contains only actions up to that reward.
        self.credit+=row['hits']*self.eligibility
        return row


def task(directory,model_root,job,temperature,raw_denom,phi_denom,deadline,max_cameras):
    began=time.monotonic();directory=Path(directory);root=Path(model_root);policy=load_readout(root/'rbf-model.npz');model_sha=base.sha(root/'rbf-model.npz');W_sha=base.sha(root/'proposal.npy');assert np.array_equal(policy.base.weights,np.load(root/'proposal.npy'));random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING)
    config=PongEvaluationConfig(seed=job['seed'],episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=base.read(root/'radial-cap.json'));session=FullRallyActor(policy,config,temperature,job['action_seed'],raw_denom,phi_denom);raw=SparseRows();hits=[]
    while not session.complete and session.transitions<max_cameras and time.monotonic()<deadline:
        if session.initial_cameras==session.episodes:session._begin_episode()
        raw.add(session);hits.append(session.step(check_frozen=False)['hits'])
    session.verify_frozen();report=session.summary();rows=session.actor_rows;tag=f"batch_{job['batch']:02d}_episode_{job['episode']}";sp.save_npz(directory/(tag+'-raw.npz'),raw.matrix());np.savez_compressed(directory/(tag+'-scores.npz'),actions=np.array([r[0] for r in rows]),greedy=np.array([r[1] for r in rows]),q=np.array([r[2] for r in rows]),p=np.array([r[3] for r in rows]),hits=hits,eligibility=session.eligibility,credit=session.credit)
    result=dict(job,complete=report['complete'],hits=report['hits'],transitions=session.transitions,initials=session.initial_cameras,temperature=temperature,report=report,model_sha256=model_sha,base_weights_sha256=W_sha,raw_feature_file=str(directory/(tag+'-raw.npz')),scores_file=str(directory/(tag+'-scores.npz')),nongreedy_choices=sum(r[0]!=r[1] for r in rows),expected_nongreedy_choices=sum(1-r[3][r[1]] for r in rows),parameters_frozen=True,wall_seconds=time.monotonic()-began)
    assert report['hits']==sum(hits) and base.sha(root/'rbf-model.npz')==model_sha and base.sha(root/'proposal.npy')==W_sha;base.atomic_json(directory/(tag+'.json'),result)
    return dict(result,eligibility=session.eligibility,credit=session.credit)
