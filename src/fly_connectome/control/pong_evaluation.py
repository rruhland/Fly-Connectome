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


def validate_camera_intervention(value):
    """Return owned timing-only sensor configuration; no policy/physics inputs."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {'kind','warmup_observations'}
            or value['kind'] not in ('warm_hold','warm_missing')
            or type(value['warmup_observations']) is not int or value['warmup_observations'] < 1):
        raise ValueError('invalid frozen camera intervention')
    return dict(value)


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
                or not isinstance(self.opponent_speed,(int,float)) or not math.isfinite(self.opponent_speed) or self.opponent_speed<0
                or (self.skip_baseline_perception and self.policy_kind == 'trained')):
            raise ValueError('invalid fresh Pong evaluation configuration')


class FreshPongEvaluator:
    durable_kind = 'pong-evaluation-v1'
    capture_rng = LearningSession.capture_rng

    def __init__(self, weights, config, *, first_hit_cutoff=False, camera_intervention=None):
        config.validate()
        camera_intervention=validate_camera_intervention(camera_intervention)
        if type(first_hit_cutoff) is not bool or (first_hit_cutoff and config.episodes != 1):
            raise ValueError("first-hit task requires a bool flag and exactly one episode")
        if camera_intervention is not None and (first_hit_cutoff or config.blank_camera
                or config.skip_baseline_perception or config.policy_kind != 'trained'):
            raise ValueError('camera intervention requires full frozen trained-policy evaluation')
        with preserve_global_rngs():
            model_bytes()
            self.config = config
            if first_hit_cutoff:
                self.first_hit_cutoff = True
            if camera_intervention is not None:
                self.camera_intervention=camera_intervention
                self.camera_observations=0
                self.held_camera_frame=None
                self.last_observation_mode='unobserved'
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
        parameters=(self.policy.weights, frozen_tensors(self), self.vision.dynamics,
                    self.vision.calibration,self.scaler,self.config,self.env.game.config)
        if hasattr(self,"first_hit_cutoff"):
            parameters += (("first-own-hit-task-v1",self.first_hit_cutoff),)
        if hasattr(self,'camera_intervention'):
            parameters += (('camera-feedback-v1',self.camera_intervention),)
        return fingerprint(parameters)

    @property
    def complete(self):
        return self.episodes == self.config.episodes

    def _observe(self, frame):
        if self.config.skip_baseline_perception:
            return
        if self.config.blank_camera:frame=torch.zeros_like(frame)
        intervention=getattr(self,'camera_intervention',None)
        mode='live'
        if intervention is not None and self.camera_observations>=intervention['warmup_observations']:
            mode=intervention['kind']
            frame=self.held_camera_frame if mode=='warm_hold' else None
        if frame is None:
            # No disappearance event or hidden new pixels are supplied. The
            # normal tracker/encoder receive their existing missing-frame API.
            events=torch.zeros((2,*self.previous.shape),dtype=torch.float32)
            self.latest_encoded=self.encoder.encode(self.vision.step(events,None,learn=False),None)
        else:
            events=torch.stack((self.previous & ~frame,~self.previous & frame)).float()
            self.previous.copy_(frame)
            # Preserve the default tensor ownership/call path exactly.
            self.latest_encoded=self.encoder.encode(self.vision.step(events,frame.float(),learn=False),frame.float())
        if intervention is not None:
            self.camera_observations+=1
            self.last_observation_mode=mode
            if self.camera_observations==intervention['warmup_observations']:
                self.held_camera_frame=frame.clone()

    def _begin_episode(self):
        if hasattr(self,'camera_intervention'):
            self.camera_observations=0;self.held_camera_frame=None;self.last_observation_mode='unobserved'
        if self.config.action_hold_frames!=1:
            self.hold_remaining=0;self.held_action=None
        frame = self.env.reset() if self.env.needs_reset else self.env.game.render(64,64)[0].clone()
        self.vision.reset_state(); self.encoder.reset_state(); self.previous.zero_()
        self._observe(frame)
        self.initial_cameras += 1
        self.episode_seconds = self.episode_return = 0.
        self.episode_hits = 0

    def _choose_policy(self):
        return self.policy.choose(*policy_features(self.latest_encoded,self.scaler))

    def verify_frozen(self):
        actual = self.learned_hash()
        if actual != self.initial_learned_hash:
            raise RuntimeError('evaluation changed frozen policy, vision or bank state')
        return actual

    def step(self, *, check_frozen=True):
        if type(check_frozen) is not bool:
            raise ValueError('boolean frozen-state check required')
        if self.complete:
            raise StopIteration('fresh Pong allocation complete')
        with preserve_global_rngs():
            if self.initial_cameras == self.episodes:
                self._begin_episode()
            camera_input=(self.last_observation_mode,self.camera_observations) if hasattr(self,'camera_intervention') else None
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
                action, values = self._choose_policy()
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
            if camera_input is not None:
                record['camera_observation']=dict(input_mode=camera_input[0],input_count=camera_input[1],
                    next_mode=self.last_observation_mode,next_count=self.camera_observations)
            if self.config.action_hold_frames!=1:
                record.update(macro_choice=choice,macro_choices=self.macro_choices,
                              hold_frame=self.config.action_hold_frames-self.hold_remaining)
            task_stopped = bool(getattr(self,"first_hit_cutoff",False) and result.hits > 0
                                and not result.terminated and not result.truncated)
            if getattr(self,"first_hit_cutoff",False):
                record.update(task_stopped=task_stopped,
                              task_stop_reason="first_own_hit" if task_stopped else None)
            if result.terminated or result.truncated or task_stopped:
                if self.config.action_hold_frames!=1:
                    self.hold_remaining=0;self.held_action=None
                self.rallies.append(dict(episode=self.episodes,reward=self.episode_return,
                    seconds=self.episode_seconds,hits=self.episode_hits,
                    terminated=result.terminated,truncated=result.truncated))
                if getattr(self,"first_hit_cutoff",False):
                    self.rallies[-1].update(task_stopped=task_stopped,
                        task_stop_reason="first_own_hit" if task_stopped else None)
                self.episodes += 1
            if check_frozen:
                self.verify_frozen()
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
        if getattr(self,'first_hit_cutoff',False):
            summary.update(task_stops=sum(r['task_stopped'] for r in self.rallies),
                           task_definition='first-own-hit-task-v1')
        if hasattr(self,'camera_intervention'):
            summary.update(camera_intervention=dict(self.camera_intervention),
                           current_episode_observations=self.camera_observations)
        return summary

    @classmethod
    def validate_restored(cls, session):
        if (type(session) is not cls or session.source != source_signature()
                or session.model_hash != MODEL_HASH or session.vision_configuration != VISION_CONFIGURATION):
            raise ValueError('incompatible fresh Pong source/model/configuration')
        session.config.validate()
        if hasattr(session,"first_hit_cutoff") and (session.first_hit_cutoff is not True or session.config.episodes != 1):
            raise ValueError("invalid restored first-hit task")
        if hasattr(session,'camera_intervention'):
            mode=validate_camera_intervention(session.camera_intervention)
            count=session.camera_observations;cache=session.held_camera_frame
            if (mode is None or session.config.policy_kind!='trained' or session.config.blank_camera
                    or session.config.skip_baseline_perception or getattr(session,'first_hit_cutoff',False)
                    or type(count) is not int or count<0
                    or session.last_observation_mode!=('unobserved' if count==0 else
                        'live' if count<=mode['warmup_observations'] else mode['kind'])
                    or (cache is not None)!=(count>=mode['warmup_observations'])
                    or (cache is not None and (not isinstance(cache,torch.Tensor)
                        or cache.device.type!='cpu' or cache.dtype!=torch.bool or cache.shape!=(64,64)))):
                raise ValueError('invalid restored camera intervention state')
        model_bytes()
        if session.learned_hash() != session.initial_learned_hash:
            raise ValueError('restored frozen evaluation parameters changed')
        import random
        random.setstate(session.python_rng); np.random.set_state(session.numpy_rng); torch.set_rng_state(session.torch_rng)
        return session
