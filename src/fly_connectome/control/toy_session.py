"""Registered fixed-template toy sessions; compatible with verified durable replay."""
from dataclasses import dataclass,asdict
import hashlib
import json
import numpy as np
import torch
from fly_connectome.vision import load_default,VisualStateEncoder
from .fresh_evaluation import EvaluationConfig,VisibleChoiceTask,FreshFrozenEvaluator
from .sarsa import TrueOnlineSarsa
from .session import LearningSession,MODEL_HASH,VISION_CONFIGURATION,model_bytes,source_signature
from .scaling import make_policy_transform,validate_scaling,block_squared_norms


@dataclass(frozen=True)
class ToySpec:
    task: str
    seed: int
    episodes: int
    mode: str = 'learn'
    lam: float = .8
    policy_kind: str = 'trained'
    action_seed: int | None = None
    diagnostic: bool = False
    period: float = .02
    alpha: float = .0001
    gamma_per_second: float = .95
    epsilon: float = .2
    policy_scaling: dict | None = None

    def validate(self):
        validate_scaling(self.policy_scaling)
        if (self.task not in ('immediate','delayed') or self.mode not in ('learn','evaluate')
                or self.lam not in (0.,.8) or self.period != .02 or self.alpha != .0001
                or self.gamma_per_second != .95 or self.epsilon != .2
                or self.policy_kind not in ('trained','initial','random')
                or type(self.seed) is not int or self.seed < 0
                or type(self.episodes) is not int or self.episodes <= 0):
            raise ValueError('invalid toy configuration')
        if self.mode=='evaluate' and self.episodes%2:
            raise ValueError('evaluation must be balanced')
        if self.mode=='learn' and self.policy_kind!='trained':
            raise ValueError('baseline policies do not train')
        if (self.policy_kind=='random') != (self.action_seed is not None):
            raise ValueError('random baseline requires its independent action seed')
        if not self.diagnostic:
            if self.policy_scaling is not None:raise ValueError('registered toy transport is unscaled')
            base=(2101 if self.task=='immediate' else 2201) if self.mode=='learn' else (3101 if self.task=='immediate' else 3201)
            expected=(200 if self.task=='immediate' else 25) if self.mode=='learn' else 20
            if self.seed not in range(base,base+3) or self.episodes!=expected:
                raise ValueError('allocation outside preregistration')
            if self.policy_kind=='random':
                expected_seed=(4101 if self.task=='immediate' else 4201)+(self.seed-base)
                if self.action_seed!=expected_seed:
                    raise ValueError('incorrect registered random-action seed')

    @property
    def left_count(self):
        if self.episodes%2==0: return self.episodes//2
        return self.episodes//2+(self.seed%2)

    @property
    def transition_budget(self):
        return self.episodes*(1 if self.task=='immediate' else 8)

    def signature(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


class ToySession(LearningSession):
    durable_kind = 'toy-session-v1'

    def __init__(self,spec,*,weights=None):
        spec.validate()
        model_bytes()
        self.spec,self.spec_hash=spec,spec.signature()
        self.source=source_signature()
        self.model_hash,self.vision_configuration=MODEL_HASH,VISION_CONFIGURATION
        self.transitions=self.episodes=0
        self.initial_cameras=self.transition_cameras=0
        self.physics_seconds=0.
        self.evaluator=None
        if spec.mode=='evaluate':
            dimension=VisualStateEncoder(64,64,.02).dimension
            if spec.policy_kind in ('initial','random'):
                weights=np.zeros((2,dimension))
            if weights is None: raise ValueError('trained evaluation needs frozen weights')
            self.evaluator=FreshFrozenEvaluator(weights,EvaluationConfig(spec.task,spec.seed,spec.episodes),
                                                 random_action_seed=spec.action_seed,policy_scaling=spec.policy_scaling)
            self.env,self.vision,self.encoder=self.evaluator.task,self.evaluator.vision,self.evaluator.encoder
            self.previous=self.evaluator.previous
            self.learner=None
        else:
            self.env=VisibleChoiceTask(spec)
            self.vision=load_default()
            self.encoder=VisualStateEncoder(64,64,.02)
            self.scaler=make_policy_transform(self.encoder.layout,self.encoder.dimension,spec.policy_scaling)
            self.previous=torch.zeros(64,64,dtype=torch.bool)
            self.learner=TrueOnlineSarsa(2,self.encoder.dimension,alpha=spec.alpha,lam=spec.lam,
                                        gamma_per_second=spec.gamma_per_second,epsilon=spec.epsilon,seed=spec.seed)
            if weights is not None: raise ValueError('registered training starts from zero weights')
            self._start_episode()
        self.capture_rng()

    @property
    def complete(self):
        return self.transitions==self.spec.transition_budget

    def _start_episode(self):
        frame=self.env.reset()
        self.vision.reset_state();self.encoder.reset_state();self.previous.zero_()
        self.initial_cameras+=1
        self.latest_encoded=self.observe(frame)
        self.learner.start(*self.policy_features(self.latest_encoded))

    def step(self):
        if self.complete: raise StopIteration('registered allocation complete')
        if self.spec.mode=='evaluate':
            if not self.env.active: self.initial_cameras+=1
            record=self.evaluator.step().copy()
            terminal=record['terminated']
        else:
            input_norms=block_squared_norms(self.latest_encoded)
            policy_norm2=float(self.learner.pending_features @ self.learner.pending_features)
            action=self.learner.pending_action
            frame,reward,terminal,elapsed=self.env.step(action)
            final=self.observe(frame)
            _,info=self.learner.advance(reward,elapsed,*self.policy_features(final),terminated=terminal)
            self.latest_encoded=final
            record=dict(transition=self.transitions,episode=self.env.episode,action=action,reward=reward,
                        terminated=terminal,truncated=False,elapsed_seconds=elapsed,**info)
            record['input_block_norm2']=input_norms
            record['policy_feature_norm2']=policy_norm2
        # Scoring labels are logged after the update and never enter learner features.
        record.update(target_side=int(self.env.sides[self.env.episode]),selected_side=int(self.env.selected),
                      delay_step=self.env.elapsed_steps)
        self.transitions+=1
        self.transition_cameras+=1
        self.physics_seconds+=record['elapsed_seconds']
        if terminal:
            self.episodes+=1
            if self.spec.mode=='learn' and not self.complete:self._start_episode()
        self.capture_rng()
        return record

    @classmethod
    def validate_restored(cls,session):
        if type(session) is not cls or session.source!=source_signature():
            raise ValueError('incompatible toy source/type')
        session.spec.validate()
        if (session.spec_hash!=session.spec.signature() or session.model_hash!=MODEL_HASH
                or session.vision_configuration!=VISION_CONFIGURATION
                or not 0<=session.transitions<=session.spec.transition_budget):
            raise ValueError('incompatible toy configuration')
        if session.evaluator is not None:
            if (session.vision is not session.evaluator.vision or session.env is not session.evaluator.task
                    or session.previous is not session.evaluator.previous):
                raise ValueError('evaluation object alias lost')
        import random
        model_bytes()
        random.setstate(session.python_rng);np.random.set_state(session.numpy_rng);torch.set_rng_state(session.torch_rng)
        return session
