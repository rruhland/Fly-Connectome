"""Fresh score-only Pong evaluation; only copied frozen readout weights transfer."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from fly_connectome.vision import load_default, VisualStateEncoder
from .codec import fingerprint, frozen_tensors
from .environment import PongCamera
from .fresh_evaluation import preserve_global_rngs
from .policy import FrozenLinearPolicy
from .session import LearningSession, MODEL_HASH, VISION_CONFIGURATION, model_bytes, source_signature
from .scaling import make_policy_transform,policy_features,validate_scaling,block_squared_norms,policy_block_squared_norms


@dataclass(frozen=True)
class PongEvaluationConfig:
    seed: int
    episodes: int = 12
    max_episode_ticks: int = 2400
    policy_kind: str = 'trained'
    action_seed: int | None = None
    skip_baseline_perception: bool = False
    policy_scaling: dict | None = None
    blank_camera: bool = False
    opponent_speed: float = .45
    action_hold_frames: int = 1

    def validate(self):
        validate_scaling(self.policy_scaling)
        if (type(self.seed) is not int or self.seed < 0
                or type(self.episodes) is not int or self.episodes < 1
                or type(self.max_episode_ticks) is not int or self.max_episode_ticks < 1
                or type(self.action_hold_frames) is not int or self.action_hold_frames < 1
                or self.policy_kind not in ('trained','initial','random','noop','up')
                or (self.policy_kind == 'random') != (self.action_seed is not None)
                or (self.action_seed is not None and (type(self.action_seed) is not int or self.action_seed < 0))
                or not isinstance(self.opponent_speed,(int,float)) or not math.isfinite(self.opponent_speed) or self.opponent_speed<=0
                or (self.skip_baseline_perception and self.policy_kind == 'trained')):
            raise ValueError('invalid fresh Pong evaluation configuration')


class FreshPongEvaluator:
    durable_kind = 'pong-evaluation-v1'
    capture_rng = LearningSession.capture_rng

    def __init__(self, weights, config):
        config.validate()
        with preserve_global_rngs():
            model_bytes()
            self.config = config
            self.source = source_signature()
            self.model_hash, self.vision_configuration = MODEL_HASH, VISION_CONFIGURATION
            # Constructor is the first serve. Do not reset and accidentally skip it.
            self.env = PongCamera(config.seed, max_episode_ticks=config.max_episode_ticks,
                                   opponent_speed=config.opponent_speed)
            self.vision = load_default()
            self.encoder = VisualStateEncoder(64,64,.02)
            self.scaler=make_policy_transform(self.encoder.layout,self.encoder.dimension,config.policy_scaling)
            dimension = self.encoder.dimension
            if config.policy_kind != 'trained':
                weights = np.zeros((3,dimension))
            if weights is None:
                raise ValueError('trained evaluation requires frozen readout weights')
            self.policy = FrozenLinearPolicy(weights)
            if self.policy.weights.shape != (3,dimension):
                raise ValueError('three-action weights matching the pinned encoder required')
            self.action_rng = None if config.action_seed is None else np.random.default_rng(config.action_seed)
            self.previous = torch.zeros(64,64,dtype=torch.bool)
            self.latest_encoded = None
            self.transitions = self.episodes = 0
            self.initial_cameras = self.transition_cameras = 0
            self.physics_seconds = self.episode_seconds = 0.
            self.episode_return = 0.
            self.episode_hits = 0
            self.rallies = []
            if config.action_hold_frames!=1:
                self.hold_remaining=0
                self.held_action=None
                self.macro_choices=0
            self.initial_learned_hash = self.learned_hash()
        self.capture_rng()

    def learned_hash(self):
        return fingerprint((self.policy.weights, frozen_tensors(self), self.vision.dynamics,
                            self.vision.calibration,self.scaler,self.config,self.env.game.config))

    @property
    def complete(self):
        return self.episodes == self.config.episodes

    def _observe(self, frame):
        if self.config.skip_baseline_perception:
            return
        if self.config.blank_camera:frame=torch.zeros_like(frame)
        events = torch.stack((self.previous & ~frame,~self.previous & frame)).float()
        self.previous.copy_(frame)
        self.latest_encoded = self.encoder.encode(self.vision.step(events,frame.float(),learn=False),frame.float())

    def _begin_episode(self):
        if self.config.action_hold_frames!=1:
            self.hold_remaining=0;self.held_action=None
        frame = self.env.reset() if self.env.needs_reset else self.env.game.render(64,64)[0].clone()
        self.vision.reset_state(); self.encoder.reset_state(); self.previous.zero_()
        self._observe(frame)
        self.initial_cameras += 1
        self.episode_seconds = self.episode_return = 0.
        self.episode_hits = 0

    def step(self):
        if self.complete:
            raise StopIteration('fresh Pong allocation complete')
        with preserve_global_rngs():
            if self.initial_cameras == self.episodes:
                self._begin_episode()
            q = None
            input_norms=None if self.latest_encoded is None else block_squared_norms(self.latest_encoded)
            policy_norms=None if self.latest_encoded is None else policy_block_squared_norms(self.latest_encoded,self.scaler)
            choice=self.config.action_hold_frames==1 or self.hold_remaining==0
            if not choice:
                action=self.held_action
            elif self.config.policy_kind == 'random':
                action = int(self.action_rng.integers(3))
            elif self.config.policy_kind == 'noop':
                action = 1
            elif self.config.policy_kind == 'up':
                action = 2
            elif self.config.policy_kind == 'initial':
                action = 0
            else:
                action, values = self.policy.choose(*policy_features(self.latest_encoded,self.scaler))
                q = values.tolist()
            if self.config.action_hold_frames!=1:
                if choice:
                    self.macro_choices+=1;self.held_action=action
                    self.hold_remaining=self.config.action_hold_frames
                self.hold_remaining-=1
            result = self.env.step(action)
            self._observe(result.frame)
            self.transitions += 1; self.transition_cameras += 1
            self.physics_seconds += result.elapsed_seconds
            self.episode_seconds += result.elapsed_seconds
            self.episode_return += result.reward; self.episode_hits += result.hits
            record = dict(transition=self.transitions-1,episode=self.episodes,action=action,q=q,
                          reward=result.reward,hits=result.hits,terminated=result.terminated,
                          truncated=result.truncated,elapsed_seconds=result.elapsed_seconds,
                          physics_ticks=result.physics_ticks,input_block_norm2=input_norms,policy_block_norm2=policy_norms)
            if self.config.action_hold_frames!=1:
                record.update(macro_choice=choice,macro_choices=self.macro_choices,
                              hold_frame=self.config.action_hold_frames-self.hold_remaining)
            if result.terminated or result.truncated:
                if self.config.action_hold_frames!=1:
                    self.hold_remaining=0;self.held_action=None
                self.rallies.append(dict(episode=self.episodes,reward=self.episode_return,
                    seconds=self.episode_seconds,hits=self.episode_hits,
                    terminated=result.terminated,truncated=result.truncated))
                self.episodes += 1
            if self.learned_hash() != self.initial_learned_hash:
                raise RuntimeError('evaluation changed frozen policy, vision or bank state')
        self.capture_rng()
        return record

    def summary(self):
        wins = sum(r['reward'] > 0 for r in self.rallies)
        losses = sum(r['reward'] < 0 for r in self.rallies)
        completed = sum(r['terminated'] for r in self.rallies)
        total = sum(r['reward'] for r in self.rallies)
        summary=dict(config=asdict(self.config),complete=self.complete,transitions=self.transitions,
            episodes=self.episodes,initial_cameras=self.initial_cameras,
            transition_cameras=self.transition_cameras,physics_seconds=self.physics_seconds,
            wins=wins,losses=losses,truncations=sum(r['truncated'] for r in self.rallies),
            natural_serve_wins=sum(r['reward']>0 and r['hits']==0 for r in self.rallies),
            own_hit_followed_wins=sum(r['reward']>0 and r['hits']>0 for r in self.rallies),
            completion_fraction=completed/self.episodes if self.episodes else None,
            win_fraction=wins/self.episodes if self.episodes else None,
            win_fraction_completed=wins/completed if completed else None,
            score_per_rally=total/self.episodes if self.episodes else None,total_reward=total,
            return_per_physical_second=total/self.physics_seconds if self.physics_seconds else None,
            hits=sum(r['hits'] for r in self.rallies),rallies=self.rallies.copy())
        if self.config.action_hold_frames!=1:summary['macro_choices']=self.macro_choices
        return summary

    @classmethod
    def validate_restored(cls, session):
        if (type(session) is not cls or session.source != source_signature()
                or session.model_hash != MODEL_HASH or session.vision_configuration != VISION_CONFIGURATION):
            raise ValueError('incompatible fresh Pong source/model/configuration')
        session.config.validate()
        model_bytes()
        if session.learned_hash() != session.initial_learned_hash:
            raise ValueError('restored frozen evaluation parameters changed')
        import random
        random.setstate(session.python_rng); np.random.set_state(session.numpy_rng); torch.set_rng_state(session.torch_rng)
        return session
