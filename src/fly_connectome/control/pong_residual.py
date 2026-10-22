"""Fixed generic tanh residual; only the two output heads learn, without BP.

Both stochastic training and greedy transfer use the same separately capped x
and uncapped bounded h. The original linear branch keeps its arithmetic order.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .codec import fingerprint
from .pong_actor import FirstHitActorEpisode,probabilities,score_eligibility
from .pong_evaluation import FreshPongEvaluator
from .projection import FixedPolicyProjection

MAP_KIND='fixed-tanh-residual-v1'


def array_hash(a):return hashlib.sha256(np.asarray(a,dtype=np.float64).tobytes()).hexdigest()


class FixedResidualMap:
    def __init__(self,config,input_dimension):
        if (type(config) is not dict or set(config)!={'kind','file','file_sha256','A_sha256','c_sha256','seed','shape'}
                or config['kind']!=MAP_KIND or config['shape']!=[64,input_dimension]
                or type(config['seed']) is not int or not 0<=config['seed']<2**32):
            raise ValueError('declared fixed64 tanh map required')
        self.config=copy.deepcopy(config)
        path=Path(config['file'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['file_sha256']:raise ValueError('residual map file hash differs')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'A','c'}:raise ValueError('residual map arrays differ')
            self.A=data['A'].copy();self.c=data['c'].copy()
        self.validate_embedded()

    def validate_embedded(self):
        if (self.A.dtype!=np.dtype('float64') or self.c.dtype!=np.dtype('float64')
                or list(self.A.shape)!=self.config['shape'] or self.c.shape!=(64,)
                or not np.isfinite(self.A).all() or not np.isfinite(self.c).all()
                or array_hash(self.A)!=self.config['A_sha256'] or array_hash(self.c)!=self.config['c_sha256']):
            raise ValueError('embedded fixed residual map changed')

    def features(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(self.A.shape[1],) or not np.isfinite(x).all():raise ValueError('matching finite capped x required')
        return np.tanh(self.A@(x/16.)+self.c)


def residual_probabilities(theta,B,x,mapping,temperature,bias=None):
    # Calling the original function preserves its exact B=0 arithmetic and p.
    q,p=probabilities(theta,x,temperature,bias)
    B=np.asarray(B,dtype=np.float64);h=mapping.features(x)
    if B.shape!=(3,64) or not np.isfinite(B).all():raise ValueError('finite3x64 residual head required')
    if np.any(B):
        q=q+B@h
        biased=q if bias is None else q+np.asarray(bias,dtype=np.float64)
        if not np.isfinite(biased).all():raise FloatingPointError('nonfinite residual scores')
        logits=(biased-biased.max())/temperature;p=np.exp(logits);p/=p.sum()
    return q,p,h


def combined_scores(theta,B,z,h):
    return np.asarray(z)@np.asarray(theta).T+np.asarray(h)@np.asarray(B).T


def residual_batch_update(theta,B,eligibilities,residual_eligibilities,rewards,z,h,*,alpha,step_cap):
    theta=np.asarray(theta,dtype=np.float64);B=np.asarray(B,dtype=np.float64)
    es=np.asarray(eligibilities,dtype=np.float64);fs=np.asarray(residual_eligibilities,dtype=np.float64)
    rs=np.asarray(rewards,dtype=np.float64);z=np.asarray(z,dtype=np.float64);h=np.asarray(h,dtype=np.float64)
    if (theta.ndim!=2 or theta.shape[0]!=3 or B.shape!=(3,64) or es.shape!=(4,*theta.shape)
            or fs.shape!=(4,3,64) or rs.shape!=(4,) or not np.isin(rs,(0,1)).all()
            or z.ndim!=2 or z.shape[1]!=theta.shape[1] or h.shape!=(len(z),64)
            or not all(np.isfinite(v).all() for v in (theta,B,es,fs,z,h))
            or not np.isfinite(alpha) or not np.isfinite(step_cap) or alpha<=0 or step_cap<=0):
        raise ValueError('one complete finite paired residual batch required')
    baseline=rs[[1,0,3,2]];weights=rs-baseline
    g=np.einsum('i,ijk->jk',weights,es)/4.;gb=np.einsum('i,ijk->jk',weights,fs)/4.
    dt=alpha*g;db=alpha*gb;dt-=dt.mean(axis=0,keepdims=True);db-=db.mean(axis=0,keepdims=True)
    dq=combined_scores(dt,db,z,h);raw=float(np.sqrt(np.mean(dq*dq)))
    scale=min(1.,step_cap/raw) if raw else 1.;dt*=scale;db*=scale
    proposed=theta+dt;proposedB=B+db
    if not np.isfinite(proposed).all() or not np.isfinite(proposedB).all():raise FloatingPointError('nonfinite residual update')
    within=int(rs[0]!=rs[1])+int(rs[2]!=rs[3])
    diagnostics=dict(rewards=rs.astype(int).tolist(),baselines=baseline.tolist(),mixed_outcomes=bool(0<rs.sum()<4),
        gradient_norm=float(np.sqrt(np.sum(g*g)+np.sum(gb*gb))),raw_update_function_rms=raw,
        aggregate_clip_scale=scale,applied_update_function_rms=float(np.sqrt(np.mean(combined_scores(dt,db,z,h)**2))),
        linear_branch_update_function_rms=float(np.sqrt(np.mean((z@dt.T)**2))),
        residual_branch_update_function_rms=float(np.sqrt(np.mean((h@db.T)**2))),
        nonzero_update=bool(np.any(proposed!=theta) or np.any(proposedB!=B)),
        calibration_greedy_changes=int(np.count_nonzero(np.argmax(combined_scores(proposed,proposedB,z,h),axis=1)!=np.argmax(combined_scores(theta,B,z,h),axis=1))),
        baseline='two-fresh-starts-two-independent-actions-v1',within_start_disagreeing_pairs=within,within_start_pairs=2,
        cross_start_disagreeing_pairs=sum(int(rs[i]!=rs[j]) for i,j in ((0,2),(0,3),(1,2),(1,3))),cross_start_pairs=4,
        cross_start_only_mixed_batch=bool(0<rs.sum()<4 and within==0))
    return proposed,proposedB,diagnostics


class ResidualFirstHitActorEpisode(FirstHitActorEpisode):
    durable_kind='pong-residual-first-hit-actor-v1'

    def __init__(self,theta,B,config,*,residual_map,**kwargs):
        if kwargs.get('baseline') is not None:raise ValueError('residual trial has no learned baseline')
        self.residual_map=FixedResidualMap(residual_map,np.asarray(theta).shape[1])
        self.B=np.asarray(B,dtype=np.float64).copy()
        if self.B.shape!=(3,64) or not np.isfinite(self.B).all():raise ValueError('finite3x64 residual head required')
        self.residual_eligibility=np.zeros_like(self.B)
        self.max_residual_feature_norm2=0.;self.last_residual_features=None
        super().__init__(theta,config,**kwargs)

    def learned_hash(self):
        return fingerprint((super().learned_hash(),MAP_KIND,self.B,self.residual_map))

    def _action_probabilities(self,x,bias):
        q,p,h=residual_probabilities(self.theta,self.B,x,self.residual_map,self.temperature,bias)
        self.last_residual_features=h
        return q,p

    def _accumulate_score(self,action,p,x):
        score=super()._accumulate_score(action,p,x)
        self.residual_eligibility+=score_eligibility(action,p,self.last_residual_features,self.temperature)
        self.max_residual_feature_norm2=max(self.max_residual_feature_norm2,float(self.last_residual_features@self.last_residual_features))
        return score

    def summary(self):
        result=super().summary()
        result['actor']['residual']=dict(kind=MAP_KIND,B_sha256=array_hash(self.B),map_config=self.residual_map.config,
            eligibility_sha256=array_hash(self.residual_eligibility),eligibility_norm=float(np.linalg.norm(self.residual_eligibility)),
            max_feature_norm2=self.max_residual_feature_norm2)
        return result

    @classmethod
    def validate_restored(cls,session):
        super().validate_restored(session);session.residual_map.validate_embedded()
        if (hasattr(session,'actor_baseline') or session.B.shape!=(3,64) or session.residual_eligibility.shape!=(3,64)
                or not np.isfinite(session.B).all() or not np.isfinite(session.residual_eligibility).all()
                or session.residual_map.A.shape[1]!=session.theta.shape[1]
                or not 0<=session.max_residual_feature_norm2<=64.+1e-10):
            raise ValueError('invalid restored residual actor')
        return session


class ProjectedResidualGreedy(FreshPongEvaluator):
    """Greedy evaluation in the exact actor coordinates, with no exploration."""
    durable_kind='pong-projected-residual-greedy-v1'
    __getstate__=FirstHitActorEpisode.__getstate__
    __setstate__=FirstHitActorEpisode.__setstate__

    def __init__(self,theta,B,config,*,projection,residual_map):
        if (config.episodes!=1 or config.policy_kind!='trained' or config.action_hold_frames!=1
                or config.action_seed is not None or config.policy_scaling is not None
                or config.skip_baseline_perception or config.blank_camera):
            raise ValueError('projected greedy requires ordinary single episode K=1')
        self.actor_projection=FixedPolicyProjection(4389,projection)
        self.theta=np.asarray(theta,dtype=np.float64).copy();self.B=np.asarray(B,dtype=np.float64).copy()
        self.residual_map=FixedResidualMap(residual_map,self.actor_projection.output_dimension)
        if self.theta.shape!=(3,self.actor_projection.output_dimension) or self.B.shape!=(3,64) or not np.isfinite(self.B).all():
            raise ValueError('matching finite projected and residual heads required')
        super().__init__(self.actor_projection.mapped_weights(self.theta),config)

    def _choose_policy(self):
        _,x=self.actor_projection.features(self.latest_encoded)
        q,_,_=residual_probabilities(self.theta,self.B,x,self.residual_map,1.)
        return int(np.argmax(q)),q

    def learned_hash(self):
        return fingerprint((super().learned_hash(),self.theta,self.B,self.actor_projection,self.residual_map))

    def summary(self):
        result=super().summary();result['projected_greedy']=dict(theta_sha256=array_hash(self.theta),B_sha256=array_hash(self.B),map_config=self.residual_map.config,
            semantics='same separately capped x and h as actor; q=theta@x+B@h; np.argmax without bias')
        return result

    @classmethod
    def validate_restored(cls,session):
        FreshPongEvaluator.validate_restored.__func__(cls,session)
        session.actor_projection.validate_embedded();session.residual_map.validate_embedded()
        if (session.theta.shape!=(3,session.actor_projection.output_dimension) or session.B.shape!=(3,64)
                or not np.isfinite(session.B).all() or not np.array_equal(session.policy.weights,session.actor_projection.mapped_weights(session.theta))):
            raise ValueError('invalid restored projected greedy head')
        return session
