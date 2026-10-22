"""Finite-episode linear actor eligibility; no autodiff or visual learning.

The head is immutable for an entire episode. A coordinator applies the scalar
Monte Carlo outcome only after a complete independent four-episode batch.
"""
import math
import numpy as np
from .codec import fingerprint
from .pong_evaluation import FreshPongEvaluator
from .projection import FixedPolicyProjection


def probabilities(theta, x, temperature):
    theta=np.asarray(theta,dtype=np.float64);x=np.asarray(x,dtype=np.float64)
    if (theta.ndim!=2 or theta.shape[0]!=3 or x.shape!=(theta.shape[1],)
            or not np.isfinite(theta).all() or not np.isfinite(x).all()
            or not math.isfinite(temperature) or temperature<=0):
        raise ValueError('finite matching head/features and positive temperature required')
    q=theta@x;logits=(q-q.max())/temperature
    p=np.exp(logits);p/=p.sum()
    return q,p


def score_eligibility(action,p,x,temperature):
    residual=-np.asarray(p,dtype=np.float64).copy();residual[action]+=1.
    return np.outer(residual,np.asarray(x,dtype=np.float64))/temperature


def batch_update(theta,eligibilities,rewards,calibration,*,alpha=1e-5,step_cap):
    """Exact LOO scalar baseline; center/cap the aggregate, never trajectories."""
    theta=np.asarray(theta,dtype=np.float64)
    es=np.asarray(eligibilities,dtype=np.float64);rs=np.asarray(rewards,dtype=np.float64)
    z=np.asarray(calibration,dtype=np.float64)
    if (rs.shape!=(4,) or es.shape!=(4,*theta.shape) or not np.isfinite(es).all()
            or not np.isin(rs,(0.,1.)).all() or z.ndim!=2 or z.shape[1]!=theta.shape[1]
            or not np.isfinite(z).all() or alpha<=0 or step_cap<=0):
        raise ValueError('one complete finite four-episode batch required')
    baselines=(rs.sum()-rs)/3.
    gradient=np.einsum('i,ijk->jk',rs-baselines,es)/4.
    delta=alpha*gradient;delta-=delta.mean(axis=0,keepdims=True)
    raw_rms=float(np.sqrt(np.mean((z@delta.T)**2)))
    scale=min(1.,step_cap/raw_rms) if raw_rms else 1.
    delta*=scale;proposed=theta+delta
    if not np.isfinite(proposed).all():raise FloatingPointError('nonfinite actor update')
    return proposed,dict(rewards=rs.astype(int).tolist(),baselines=baselines.tolist(),
        mixed_outcomes=bool(0<rs.sum()<4),gradient_norm=float(np.linalg.norm(gradient)),
        raw_update_function_rms=raw_rms,aggregate_clip_scale=scale,
        applied_update_function_rms=float(np.sqrt(np.mean((z@delta.T)**2))),
        nonzero_update=bool(np.any(proposed!=theta)),
        calibration_greedy_changes=int(np.count_nonzero(np.argmax(z@proposed.T,axis=1)!=np.argmax(z@theta.T,axis=1))))


class FirstHitActorEpisode(FreshPongEvaluator):
    durable_kind='pong-first-hit-actor-v1'

    def __init__(self,theta,config,*,projection,temperature,action_seed):
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
        return fingerprint((super().learned_hash(),self.theta,self.temperature,self.actor_action_seed,self.actor_projection))

    def _choose_policy(self):
        _,x=self.actor_projection.features(self.latest_encoded)
        q,p=probabilities(self.theta,x,self.temperature)
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
        return action,q

    def step(self):
        record=super().step()
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
        return result

    @classmethod
    def validate_restored(cls,session):
        FreshPongEvaluator.validate_restored.__func__(cls,session)
        session.actor_projection.validate_embedded()
        if (session.config.action_hold_frames!=1 or session.config.episodes!=1
                or session.config.policy_kind!='trained' or session.config.action_seed is not None
                or session.config.policy_scaling is not None or not session.first_hit_cutoff
                or session.decisions!=session.transitions or session.eligibility.shape!=session.theta.shape
                or not np.isfinite(session.eligibility).all()
                or not np.array_equal(session.policy.weights,session.actor_projection.mapped_weights(session.theta))):
            raise ValueError('invalid restored first-hit actor state')
        return session
