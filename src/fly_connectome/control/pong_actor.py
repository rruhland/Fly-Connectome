"""Finite-episode linear actor eligibility; no autodiff or visual learning.

The head is immutable for an entire episode. A coordinator applies the scalar
Monte Carlo outcome only after a complete independent four-episode batch.
"""
import math
import numpy as np
from .codec import fingerprint
from .pong_evaluation import FreshPongEvaluator
from .projection import FixedPolicyProjection


def probabilities(theta, x, temperature, bias=None):
    theta=np.asarray(theta,dtype=np.float64);x=np.asarray(x,dtype=np.float64)
    if (theta.ndim!=2 or theta.shape[0]!=3 or x.shape!=(theta.shape[1],)
            or not np.isfinite(theta).all() or not np.isfinite(x).all()
            or not math.isfinite(temperature) or temperature<=0):
        raise ValueError('finite matching head/features and positive temperature required')
    if bias is not None:
        bias=np.asarray(bias,dtype=np.float64)
        if bias.shape!=(3,) or not np.isfinite(bias).all():raise ValueError('finite three-action bias required')
    q=theta@x;biased=q if bias is None else q+bias
    if biased.shape!=(3,) or not np.isfinite(biased).all():raise ValueError('finite three-action bias required')
    logits=(biased-biased.max())/temperature
    p=np.exp(logits);p/=p.sum()
    return q,p


def validate_bias(config):
    if config is None:return None
    if (type(config) is not dict or set(config)!={'kind','sigma','rho','seed'}
            or config['kind']!='centered-gaussian-ar1-v1'
            or type(config['seed']) is not int or not 0<=config['seed']<2**32
            or not isinstance(config['sigma'],(int,float)) or not math.isfinite(config['sigma']) or config['sigma']<=0
            or not isinstance(config['rho'],(int,float)) or not math.isfinite(config['rho']) or not 0<=config['rho']<1):
        raise ValueError('declared finite theta-independent Gaussian bias required')
    return dict(config)


def score_eligibility(action,p,x,temperature):
    residual=-np.asarray(p,dtype=np.float64).copy();residual[action]+=1.
    return np.outer(residual,np.asarray(x,dtype=np.float64))/temperature


def batch_update(theta,eligibilities,rewards,calibration,*,alpha=1e-5,step_cap,paired_starts=False):
    """LOO or paired-start scalar baseline; cap aggregate, never trajectories."""
    theta=np.asarray(theta,dtype=np.float64)
    es=np.asarray(eligibilities,dtype=np.float64);rs=np.asarray(rewards,dtype=np.float64)
    z=np.asarray(calibration,dtype=np.float64)
    if (type(paired_starts) is not bool or rs.shape!=(4,) or es.shape!=(4,*theta.shape) or not np.isfinite(es).all()
            or not np.isin(rs,(0.,1.)).all() or z.ndim!=2 or z.shape[1]!=theta.shape[1]
            or not np.isfinite(z).all() or alpha<=0 or step_cap<=0):
        raise ValueError('one complete finite four-episode batch required')
    baselines=rs[[1,0,3,2]] if paired_starts else (rs.sum()-rs)/3.
    gradient=np.einsum('i,ijk->jk',rs-baselines,es)/4.
    delta=alpha*gradient;delta-=delta.mean(axis=0,keepdims=True)
    raw_rms=float(np.sqrt(np.mean((z@delta.T)**2)))
    scale=min(1.,step_cap/raw_rms) if raw_rms else 1.
    delta*=scale;proposed=theta+delta
    if not np.isfinite(proposed).all():raise FloatingPointError('nonfinite actor update')
    diagnostics=dict(rewards=rs.astype(int).tolist(),baselines=baselines.tolist(),
        mixed_outcomes=bool(0<rs.sum()<4),gradient_norm=float(np.linalg.norm(gradient)),
        raw_update_function_rms=raw_rms,aggregate_clip_scale=scale,
        applied_update_function_rms=float(np.sqrt(np.mean((z@delta.T)**2))),
        nonzero_update=bool(np.any(proposed!=theta)),
        calibration_greedy_changes=int(np.count_nonzero(np.argmax(z@proposed.T,axis=1)!=np.argmax(z@theta.T,axis=1))))
    if paired_starts:
        within=int(rs[0]!=rs[1])+int(rs[2]!=rs[3])
        diagnostics.update(baseline='two-fresh-starts-two-independent-actions-v1',
            within_start_disagreeing_pairs=within,within_start_pairs=2,
            cross_start_disagreeing_pairs=sum(int(rs[i]!=rs[j]) for i,j in ((0,2),(0,3),(1,2),(1,3))),
            cross_start_pairs=4,cross_start_only_mixed_batch=bool(0<rs.sum()<4 and within==0))
    return proposed,diagnostics


class FirstHitActorEpisode(FreshPongEvaluator):
    durable_kind='pong-first-hit-actor-v1'

    def __init__(self,theta,config,*,projection,temperature,action_seed,noise=None):
        if (config.episodes!=1 or config.policy_kind!='trained' or config.action_hold_frames!=1
                or config.action_seed is not None or config.policy_scaling is not None
                or config.skip_baseline_perception or config.blank_camera
                or type(action_seed) is not int or not 0<=action_seed<2**32):
            raise ValueError('actor requires one live first-hit episode and K=1')
        self.actor_projection=FixedPolicyProjection(4389,projection)
        self.theta=np.asarray(theta,dtype=np.float64).copy()
        if self.theta.shape!=(3,self.actor_projection.output_dimension) or not np.isfinite(self.theta).all():
            raise ValueError('finite projected three-action actor head required')
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('positive finite temperature required')
        self.temperature=float(temperature);self.actor_action_seed=action_seed
        self.actor_rng=np.random.default_rng(action_seed)
        noise=validate_bias(noise)
        if noise is not None:
            if noise['seed']==action_seed:raise ValueError('independent bias and action RNG seeds required')
            self.actor_bias_config=noise
            self.bias_rng=np.random.default_rng(noise['seed'])
            self.bias=self._bias_innovation()  # Stationary initial law, not zero.
            self.bias_draws=1
        self.eligibility=np.zeros_like(self.theta)
        self.decisions=self.nongreedy_decisions=0
        self.entropy_sum=self.expected_nongreedy_sum=self.log_probability_sum=0.
        self.max_projected_feature_norm2=0.
        self.last_actor_choice=None
        super().__init__(self.actor_projection.mapped_weights(self.theta),config,first_hit_cutoff=True)

    def __getstate__(self):
        # Actor episodes borrow one pinned external T. Avoid duplicating its
        # 7.5MB matrix in hundreds of short episode snapshots. Full fingerprints
        # still traverse the ACTUAL matrix in __dict__, not this pickle encoding.
        state=self.__dict__.copy()
        state['_actor_projection_checkpoint']=state.pop('actor_projection').config
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        config=self.__dict__.pop('_actor_projection_checkpoint')
        # Hash/shape/content validation fails closed for missing or changed T.
        self.actor_projection=FixedPolicyProjection(4389,config)

    def learned_hash(self):
        # Mutable RNG/eligibility/diagnostics are in the FULL durable graph, not
        # this immutable parameter anchor. The projection is an exact required external checkpoint artifact.
        parameters=(super().learned_hash(),self.theta,self.temperature,self.actor_action_seed,self.actor_projection)
        if hasattr(self,'actor_bias_config'):parameters+=(self.actor_bias_config,)
        return fingerprint(parameters)

    def _bias_innovation(self):
        normal=self.bias_rng.standard_normal(3)
        return self.actor_bias_config['sigma']*(normal-normal.mean())

    def _choose_policy(self):
        _,x=self.actor_projection.features(self.latest_encoded)
        noise=getattr(self,'actor_bias_config',None)
        if noise is not None and self.decisions:
            rho=noise['rho']
            self.bias=rho*self.bias+math.sqrt(1.-rho*rho)*self._bias_innovation()
            self.bias_draws+=1
        q,p=probabilities(self.theta,x,self.temperature,None if noise is None else self.bias)
        action=int(self.actor_rng.choice(3,p=p))
        self.eligibility+=score_eligibility(action,p,x,self.temperature)
        nonzero=p>0;entropy=-float(p[nonzero]@np.log(p[nonzero]))
        greedy=int(np.argmax(q));nongreedy=action!=greedy
        logp=float(np.log(p[action]));norm2=float(x@x)
        self.decisions+=1;self.nongreedy_decisions+=int(nongreedy)
        self.entropy_sum+=entropy;self.expected_nongreedy_sum+=1.-float(p[greedy]);self.log_probability_sum+=logp
        self.max_projected_feature_norm2=max(self.max_projected_feature_norm2,norm2)
        self.last_actor_choice=dict(decision=self.decisions,probabilities=p.tolist(),
            greedy_action=greedy,non_greedy=nongreedy,entropy=entropy,log_probability=logp,
            projected_feature_norm2=norm2)
        if noise is not None:
            self.last_actor_choice.update(bias=self.bias.tolist(),biased_q=(q+self.bias).tolist(),
                biased_greedy_action=int(np.argmax(q+self.bias)),bias_draws=self.bias_draws)
        return action,q

    def step(self, *, check_frozen=True):
        record=super().step(check_frozen=check_frozen)
        record['actor']=self.last_actor_choice.copy()
        return record

    def summary(self):
        result=super().summary();n=self.decisions
        result['actor']=dict(temperature=self.temperature,action_seed=self.actor_action_seed,
            decisions=n,non_greedy_decisions=self.nongreedy_decisions,
            non_greedy_fraction=self.nongreedy_decisions/n if n else None,
            mean_entropy=self.entropy_sum/n if n else None,
            expected_non_greedy_fraction=self.expected_nongreedy_sum/n if n else None,
            log_probability_sum=self.log_probability_sum,
            eligibility_norm=float(np.linalg.norm(self.eligibility)),
            eligibility_sha256=__import__('hashlib').sha256(self.eligibility.tobytes()).hexdigest(),
            max_projected_feature_norm2=self.max_projected_feature_norm2)
        if hasattr(self,'actor_bias_config'):
            result['actor'].update(noise=dict(self.actor_bias_config),bias_draws=self.bias_draws)
        return result

    @classmethod
    def validate_restored(cls,session):
        FreshPongEvaluator.validate_restored.__func__(cls,session)
        session.actor_projection.validate_embedded()
        noise=validate_bias(getattr(session,'actor_bias_config',None))
        if noise is not None and (session.bias.shape!=(3,) or not np.isfinite(session.bias).all()
                or session.bias_draws!=max(1,session.decisions) or noise['seed']==session.actor_action_seed):
            raise ValueError('invalid restored bias process')
        if (session.config.action_hold_frames!=1 or session.config.episodes!=1
                or session.config.policy_kind!='trained' or session.config.action_seed is not None
                or session.config.policy_scaling is not None or not session.first_hit_cutoff
                or session.decisions!=session.transitions or session.eligibility.shape!=session.theta.shape
                or not np.isfinite(session.eligibility).all()
                or not np.array_equal(session.policy.weights,session.actor_projection.mapped_weights(session.theta))):
            raise ValueError('invalid restored first-hit actor state')
        return session
