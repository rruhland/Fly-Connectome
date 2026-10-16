"""Reconstructed camera-only learning session with explicit episode boundaries.

Trusted local pickle snapshots include the complete object graph, not vision.save.
Immutable-bank deduplication and interval/delta replay remain later stages.
"""
import copy
import hashlib
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from fly_connectome.vision import load_default, VisualStateEncoder
from .environment import PongCamera
from .sarsa import TrueOnlineSarsa
from .scaling import make_policy_transform,policy_features

MODEL_HASH = 'a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173'
VISION_CONFIGURATION = 'load_default:legacy-v1+upgrade_temporal_context:consensus-v3:64x64'


def model_bytes():
    raw = (Path(__file__).parents[1]/'vision/data/default.pt').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_HASH:
        raise ValueError('promoted visual prior hash mismatch')
    return raw


def source_signature():
    root = Path(__file__).parents[1]
    paths = [root/'pong.py', *sorted((root/'control').glob('*.py')),
             *sorted((root/'vision').glob('*.py'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class LearningSession:
    durable_kind = 'pong-learning-v1'
    def __init__(self, seed=1101, *, max_episode_ticks=None, weights=None,
                 alpha=.0001, lam=.8, gamma_per_second=.95, epsilon=.2, policy_scaling=None,
                 opponent_speed=None, action_hold_frames=1):
        if type(action_hold_frames) is not int or action_hold_frames<1 or not 0<=lam<=1:
            raise ValueError('positive integer camera frames per action required')
        model_bytes()
        self.config = dict(seed=seed,max_episode_ticks=max_episode_ticks,alpha=alpha,lam=lam,
                           gamma_per_second=gamma_per_second,epsilon=epsilon)
        self.model_hash, self.vision_configuration = MODEL_HASH, VISION_CONFIGURATION
        self.source = source_signature()
        if opponent_speed is not None:self.config['opponent_speed']=opponent_speed
        if action_hold_frames!=1:
            self.config.update(action_hold_frames=action_hold_frames,macro_lambda=lam**action_hold_frames)
            self.action_hold_frames=action_hold_frames
            self.hold_frames=0
            self.hold_reward=self.hold_elapsed=0.
            self.macro_choices=0
        self.env = PongCamera(seed, max_episode_ticks=max_episode_ticks,
                              opponent_speed=.45 if opponent_speed is None else opponent_speed)
        self.body_alias = self.env.game.body
        self.vision = load_default()
        self.encoder = VisualStateEncoder(64,64,.02)
        self.scaler = make_policy_transform(self.encoder.layout,self.encoder.dimension,policy_scaling)
        if policy_scaling is not None:self.config['policy_scaling']=self.scaler.config
        self.learner = TrueOnlineSarsa(3,self.encoder.dimension,alpha=alpha,lam=lam**action_hold_frames,
                                      gamma_per_second=gamma_per_second,epsilon=epsilon,seed=seed,weights=weights)
        self.previous = torch.zeros(64,64,dtype=torch.bool)
        self.transitions = self.episodes = 0
        self.initial_cameras = 1
        self.transition_cameras = 0
        self.physics_seconds = 0.
        self.latest_encoded = self.observe(self.env.game.render(64,64)[0])
        self.learner.start(*self.policy_features(self.latest_encoded))
        self.capture_rng()

    def capture_rng(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()

    def observe(self, frame):
        events = torch.stack((self.previous & ~frame, ~self.previous & frame)).float()
        self.previous.copy_(frame)
        return self.encoder.encode(self.vision.step(events,frame.float(),learn=False),frame.float())

    def policy_features(self,encoded):
        return policy_features(encoded,getattr(self,'scaler',None))

    def step(self):
        if getattr(self,'action_hold_frames',1)!=1:return self._step_hold()
        action = self.learner.pending_action
        active = self.latest_encoded
        q_values = self.learner.weights @ self.learner.pending_features
        policy_norm2=float(self.learner.pending_features @ self.learner.pending_features)
        result = self.env.step(action)
        final = self.observe(result.frame)
        _, info = self.learner.advance(result.reward,result.elapsed_seconds,
                                      *self.policy_features(final),
                                      terminated=result.terminated,truncated=result.truncated)
        self.latest_encoded = final
        self.transitions += 1
        self.transition_cameras += 1
        self.physics_seconds += result.elapsed_seconds
        record = dict(transition=self.transitions-1,action=action,reward=result.reward,
                      elapsed_seconds=result.elapsed_seconds,physics_ticks=result.physics_ticks,
                      feature_norm2=float(active['values'].square().sum()),
                      policy_feature_norm2=policy_norm2,
                      active_features=len(active['indices']),entities=len(active['entities']),
                      forecasts=len(active['forecasts']),q_spread=float(np.ptp(q_values)),
                      terminated=result.terminated,truncated=result.truncated,hits=result.hits,**info)
        if result.terminated or result.truncated:
            self.episodes += 1
            self.initial_cameras += 1
            frame = self.env.reset()
            self.vision.reset_state()
            self.encoder.reset_state()
            self.previous.zero_()
            self.latest_encoded = self.observe(frame)
            self.learner.start(*self.policy_features(self.latest_encoded))
        self.capture_rng()
        return record

    def _step_hold(self):
        """One camera/journal sample; a held action can span several samples.

        feature_norm2 describes this camera's raw transport. In contrast,
        policy_feature_norm2 and q_spread describe the pending macro-start
        feature vector used by the learner. Nonupdate target fields are None.
        """
        if self.hold_frames==0:self.macro_choices+=1
        action=self.learner.pending_action
        active=self.latest_encoded
        q_values=self.learner.weights@self.learner.pending_features
        policy_norm2=float(self.learner.pending_features@self.learner.pending_features)
        result=self.env.step(action)
        final=self.observe(result.frame)
        self.hold_reward+=self.learner.gamma_per_second**self.hold_elapsed*result.reward
        self.hold_elapsed+=result.elapsed_seconds
        self.hold_frames+=1
        frames,reward,elapsed=self.hold_frames,self.hold_reward,self.hold_elapsed
        update=frames==self.action_hold_frames or result.terminated or result.truncated
        if update:
            _,info=self.learner.advance(reward,elapsed,*self.policy_features(final),
                                       terminated=result.terminated,truncated=result.truncated)
            self.hold_frames=0
            self.hold_reward=self.hold_elapsed=0.
        else:
            info=dict(q=float(q_values[action]),q_next=None,delta=None,
                      outgoing_discount=None,bootstrap_action=None)
        self.latest_encoded=final
        self.transitions+=1;self.transition_cameras+=1
        self.physics_seconds+=result.elapsed_seconds
        record=dict(transition=self.transitions-1,action=action,reward=result.reward,
            elapsed_seconds=result.elapsed_seconds,physics_ticks=result.physics_ticks,
            feature_norm2=float(active['values'].square().sum()),policy_feature_norm2=policy_norm2,
            active_features=len(active['indices']),entities=len(active['entities']),forecasts=len(active['forecasts']),
            q_spread=float(np.ptp(q_values)),terminated=result.terminated,truncated=result.truncated,hits=result.hits,
            macro_update=bool(update),hold_frame=frames,macro_reward=reward,macro_elapsed_seconds=elapsed,
            macro_choices=self.macro_choices,macro_updates=self.learner.transitions,**info)
        if result.terminated or result.truncated:
            self.episodes+=1;self.initial_cameras+=1
            frame=self.env.reset()
            self.vision.reset_state();self.encoder.reset_state();self.previous.zero_()
            self.latest_encoded=self.observe(frame)
            self.learner.start(*self.policy_features(self.latest_encoded))
        self.capture_rng()
        return record

    def snapshot(self):
        self.capture_rng()
        return pickle.dumps(self,protocol=5)

    @classmethod
    def restore(cls, payload):
        # Only trusted locally generated payloads: hashes are integrity, not authentication.
        return cls.validate_restored(pickle.loads(payload))

    @classmethod
    def validate_restored(cls, session):
        if (type(session) is not cls or session.source != source_signature()
                or session.model_hash != MODEL_HASH or session.vision_configuration != VISION_CONFIGURATION):
            raise ValueError('incompatible source/model/configuration')
        model_bytes()
        if session.body_alias is not session.env.game.body:
            raise ValueError('physics alias lost')
        random.setstate(session.python_rng)
        np.random.set_state(session.numpy_rng)
        torch.set_rng_state(session.torch_rng)
        return session

    def evaluate(self, transitions):
        """Frozen trajectory-clone diagnostic; it starts a new control hold.

        The clone begins at the current camera, with no training session/RNG
        mutation. Its hold phase restarts even when training is mid-hold.
        Use FreshPongEvaluator for independent competence evaluation.
        """
        if type(transitions) is not int or not 0 <= transitions <= 200:
            raise ValueError('bounded evaluation count required')
        other = copy.deepcopy(self)
        policy = self.learner.frozen()
        records = []
        hold_remaining=0
        for _ in range(transitions):
            encoded = other.latest_encoded
            if not hold_remaining:
                action, q = policy.choose(*other.policy_features(encoded))
                hold_remaining=getattr(other,'action_hold_frames',1)
            result = other.env.step(action)
            hold_remaining-=1
            other.latest_encoded = other.observe(result.frame)
            records.append(dict(action=action,q=q.tolist(),reward=result.reward,
                                elapsed_seconds=result.elapsed_seconds))
            if result.terminated or result.truncated:
                hold_remaining=0
                other.vision.reset_state()
                other.encoder.reset_state()
                other.previous.zero_()
                other.latest_encoded = other.observe(other.env.reset())
        return records
