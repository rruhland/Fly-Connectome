"""One genuinely frozen fresh pendulum trial; no policy or visual updates."""
import copy
from dataclasses import dataclass
import random
import numpy as np
import torch
from fly_connectome.vision import load_default,VisualStateEncoder
from .codec import fingerprint
from .fresh_evaluation import preserve_global_rngs
from .online_vision import (PersistentVisionSession,VisionSnapshot,learned_visual_state,
                            observer_tensors)
from .pendulum import PendulumCamera,PendulumPhysics
from .policy import FrozenLinearPolicy
from .scaling import make_policy_transform,policy_features
from .session import LearningSession,source_signature,model_bytes,MODEL_HASH,VISION_CONFIGURATION


@dataclass(frozen=True)
class PendulumEvaluationConfig:
    seed: int
    max_episode_ticks: int=600
    physics: PendulumPhysics=PendulumPhysics()
    policy_kind: str='trained'
    action_seed: int|None=None
    policy_scaling: dict|None=None
    skip_baseline_perception: bool=False
    action_hold_frames: int=1

    @classmethod
    def from_dict(cls,options):
        options=copy.deepcopy(options)
        if isinstance(options.get('physics'),dict):options['physics']=PendulumPhysics(**options['physics'])
        return cls(**options)

    def validate(self):
        if type(self.physics) is not PendulumPhysics:raise ValueError('typed pendulum physics required')
        self.physics.validate()
        if (type(self.seed) is not int or self.seed<0
                or type(self.max_episode_ticks) is not int or self.max_episode_ticks<1
                or self.policy_kind not in ('trained','negative','noop','positive','random')
                or type(self.skip_baseline_perception) is not bool
                or type(self.action_hold_frames) is not int or self.action_hold_frames<1
                or (self.policy_kind=='trained' and self.action_hold_frames!=1)
                or (self.skip_baseline_perception and self.policy_kind=='trained')
                or (self.policy_kind=='random' and (type(self.action_seed) is not int or self.action_seed<0))):
            raise ValueError('one fresh trial, fixed/legal policy and explicit random-action seed required')


class FreshPendulumEvaluator:
    """Own cloned visual memory, mapped raw head and initial-state/action RNGs."""
    durable_kind='pendulum-evaluation-v1'
    __getstate__=PersistentVisionSession.__getstate__
    __setstate__=PersistentVisionSession.__setstate__
    retained_vision=PersistentVisionSession.retained_vision
    capture_rng=LearningSession.capture_rng

    def __init__(self,weights,config,*,vision_snapshot=None):
        with preserve_global_rngs():self._initialize(weights,config,vision_snapshot)

    def _initialize(self,weights,config,vision_snapshot):
        config.validate();model_bytes();self.config=copy.deepcopy(config)
        self.source=source_signature();self.model_hash=MODEL_HASH;self.vision_configuration=VISION_CONFIGURATION
        self.env=PendulumCamera(config.seed,max_episode_ticks=config.max_episode_ticks,config=config.physics)
        self.vision=load_default();observer=fingerprint(observer_tensors(self.vision))
        self.starting_visual_snapshot=None;self.starting_visual_learning_observations=0
        if vision_snapshot is not None:
            if type(vision_snapshot) is not VisionSnapshot:raise ValueError('owned visual snapshot required')
            self.vision=vision_snapshot.clone_model()
            if fingerprint(observer_tensors(self.vision))!=observer:raise ValueError('fixed observer changed')
            self.starting_visual_snapshot=vision_snapshot.content_hash
            self.starting_visual_learning_observations=vision_snapshot.learning_observations
        self.vision.reset_state();self.encoder=VisualStateEncoder(64,64,.02)
        self.scaler=make_policy_transform(self.encoder.layout,self.encoder.dimension,config.policy_scaling)
        if weights is None and config.policy_kind!='trained':weights=np.zeros((3,self.encoder.dimension))
        self.policy=FrozenLinearPolicy(weights)
        if self.policy.weights.shape!=(3,self.encoder.dimension):raise ValueError('three mapped raw action heads required')
        self.action_rng=np.random.default_rng(0 if config.action_seed is None else config.action_seed)
        self.previous=torch.zeros(64,64,dtype=torch.bool)
        self.transitions=self.transition_cameras=self.episodes=0;self.initial_cameras=1
        self.visual_observations=self.visual_learning_observations=0
        self.physics_seconds=0.;self.terminated=self.truncated=False;self.action_counts=[0,0,0]
        self.hold_remaining=0;self.held_action=None
        self.latest_encoded=self._observe(self.env.render())
        self.initial_learned_hash=self.learned_hash();self.capture_rng()

    @property
    def complete(self):return self.terminated or self.truncated

    def _observe(self,frame):
        if self.config.skip_baseline_perception:return None
        events=torch.stack((self.previous & ~frame,~self.previous & frame)).float();self.previous.copy_(frame)
        self.visual_observations+=1
        return self.encoder.encode(self.vision.step(events,frame.float(),learn=False),frame.float())

    def learned_hash(self):
        return fingerprint((self.policy.weights,observer_tensors(self.vision),learned_visual_state(self.vision),
                            self.scaler,self.config,self.env.config,self.starting_visual_snapshot,
                            self.starting_visual_learning_observations))

    def step(self):
        if self.complete:raise StopIteration('fresh pendulum trial complete')
        with preserve_global_rngs():
            random.setstate(self.python_rng);np.random.set_state(self.numpy_rng);torch.set_rng_state(self.torch_rng)
            if self.learned_hash()!=self.initial_learned_hash:raise RuntimeError('frozen pendulum readout/vision changed')
            if self.config.policy_kind=='trained':
                action,_=self.policy.choose(*policy_features(self.latest_encoded,self.scaler))
            elif self.config.policy_kind=='random':
                if not self.hold_remaining:
                    self.held_action=int(self.action_rng.integers(3));self.hold_remaining=self.config.action_hold_frames
                action=self.held_action;self.hold_remaining-=1
            else:action={'negative':0,'noop':1,'positive':2}[self.config.policy_kind]
            result=self.env.step(action);self.latest_encoded=self._observe(result.frame)
            self.transitions+=1;self.transition_cameras+=1;self.physics_seconds+=result.elapsed_seconds
            self.terminated,self.truncated=result.terminated,result.truncated
            self.episodes=int(self.complete);self.action_counts[action]+=1
            if self.learned_hash()!=self.initial_learned_hash:raise RuntimeError('frozen evaluation updated visual memory')
            self.capture_rng()
            return dict(transition=self.transitions-1,episode=0,action=action,reward=result.reward,
                elapsed_seconds=result.elapsed_seconds,physics_ticks=result.physics_ticks,
                terminated=result.terminated,truncated=result.truncated,hits=0,
                random_hold_remaining=self.hold_remaining if self.config.policy_kind=='random' else None,
                visual_learning_enabled=False,visual_observations=self.visual_observations)

    def summary(self):
        return dict(complete=self.complete,episodes=self.episodes,transitions=self.transitions,
            initial_cameras=self.initial_cameras,transition_cameras=self.transition_cameras,
            survival_seconds=self.physics_seconds,total_reward=self.physics_seconds,
            failure=bool(self.terminated),time_limit=bool(self.truncated),action_counts=list(self.action_counts),
            visual_observations=self.visual_observations,visual_learning_observations=0,
            cumulative_visual_learning_observations=self.starting_visual_learning_observations,
            starting_visual_snapshot=self.starting_visual_snapshot)

    @classmethod
    def validate_restored(cls,session):
        if (type(session) is not cls or session.source!=source_signature()
                or session.model_hash!=MODEL_HASH or session.vision_configuration!=VISION_CONFIGURATION
                or session.env.config!=session.config.physics
                or session.env.max_episode_ticks!=session.config.max_episode_ticks
                or session.complete!=session.env.needs_reset or session.episodes!=int(session.complete)
                or session.transition_cameras!=session.transitions or session.initial_cameras!=1
                or session.visual_observations!=(0 if session.config.skip_baseline_perception else session.transitions+1)
                or session.visual_learning_observations!=0 or sum(session.action_counts)!=session.transitions
                or not 0<=session.hold_remaining<session.config.action_hold_frames
                or (session.config.policy_kind!='random' and (session.hold_remaining or session.held_action is not None))
                or (session.config.policy_kind=='random' and session.transitions>0 and session.held_action not in (0,1,2))
                or session.learned_hash()!=session.initial_learned_hash):
            raise ValueError('invalid fresh frozen pendulum source/parameters/counters')
        session.config.validate();model_bytes();return session
