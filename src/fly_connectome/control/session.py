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
    def __init__(self, seed=1101, *, max_episode_ticks=None, weights=None):
        model_bytes()
        self.model_hash, self.vision_configuration = MODEL_HASH, VISION_CONFIGURATION
        self.source = source_signature()
        self.env = PongCamera(seed, max_episode_ticks=max_episode_ticks)
        self.body_alias = self.env.game.body
        self.vision = load_default()
        self.encoder = VisualStateEncoder(64,64,.02)
        self.learner = TrueOnlineSarsa(3,self.encoder.dimension,alpha=.0001,lam=.8,
                                      gamma_per_second=.95,epsilon=.2,seed=seed,weights=weights)
        self.previous = torch.zeros(64,64,dtype=torch.bool)
        self.transitions = self.episodes = 0
        self.physics_seconds = 0.
        self.latest_encoded = self.observe(self.env.game.render(64,64)[0])
        self.learner.start(self.latest_encoded['indices'].numpy(), self.latest_encoded['values'].numpy())
        self.capture_rng()

    def capture_rng(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()

    def observe(self, frame):
        events = torch.stack((self.previous & ~frame, ~self.previous & frame)).float()
        self.previous.copy_(frame)
        return self.encoder.encode(self.vision.step(events,frame.float(),learn=False),frame.float())

    def step(self):
        action = self.learner.pending_action
        result = self.env.step(action)
        final = self.observe(result.frame)
        _, info = self.learner.advance(result.reward,result.elapsed_seconds,
                                      final['indices'].numpy(),final['values'].numpy(),
                                      terminated=result.terminated,truncated=result.truncated)
        self.latest_encoded = final
        self.transitions += 1
        self.physics_seconds += result.elapsed_seconds
        record = dict(transition=self.transitions-1,action=action,reward=result.reward,
                      elapsed_seconds=result.elapsed_seconds,physics_ticks=result.physics_ticks,
                      terminated=result.terminated,truncated=result.truncated,**info)
        if result.terminated or result.truncated:
            self.episodes += 1
            frame = self.env.reset()
            self.vision.reset_state()
            self.encoder.reset_state()
            self.previous.zero_()
            self.latest_encoded = self.observe(frame)
            self.learner.start(self.latest_encoded['indices'].numpy(),self.latest_encoded['values'].numpy())
        self.capture_rng()
        return record

    def snapshot(self):
        self.capture_rng()
        return pickle.dumps(self,protocol=5)

    @classmethod
    def restore(cls, payload):
        # Only trusted locally generated payloads: hashes are integrity, not authentication.
        session = pickle.loads(payload)
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
        """Evaluate a deep copy; no reset or mutation of the training session/RNGs."""
        if type(transitions) is not int or not 0 <= transitions <= 200:
            raise ValueError('bounded evaluation count required')
        other = copy.deepcopy(self)
        policy = self.learner.frozen()
        records = []
        for _ in range(transitions):
            encoded = other.latest_encoded
            action, q = policy.choose(encoded['indices'].numpy(),encoded['values'].numpy())
            result = other.env.step(action)
            other.latest_encoded = other.observe(result.frame)
            records.append(dict(action=action,q=q.tolist(),reward=result.reward,
                                elapsed_seconds=result.elapsed_seconds))
            if result.terminated or result.truncated:
                other.vision.reset_state()
                other.encoder.reset_state()
                other.previous.zero_()
                other.latest_encoded = other.observe(other.env.reset())
        return records
