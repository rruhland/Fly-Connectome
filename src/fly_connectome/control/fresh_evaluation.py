"""Fresh-scene frozen evaluation for registered visible-choice toy tasks.

Only a defensive copy of readout weights crosses the training/evaluation boundary.
This module does not launch evaluations automatically or implement toy training.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import random
import time

import numpy as np
import torch
from fly_connectome.vision import load_default, VisualStateEncoder
from .codec import fingerprint, frozen_tensors
from .policy import FrozenLinearPolicy
from .session import MODEL_HASH, VISION_CONFIGURATION, model_bytes


@contextmanager
def preserve_global_rngs():
    """Synchronous boundary, including exception paths; local task RNG is separate."""
    python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(torch_state)


@dataclass(frozen=True)
class EvaluationConfig:
    task: str
    seed: int
    episodes: int = 20
    period: float = .02

    def __post_init__(self):
        if (self.task not in ('immediate','delayed') or type(self.seed) is not int
                or self.seed < 0 or type(self.episodes) is not int
                or self.episodes <= 0 or self.episodes % 2 or self.period != .02):
            raise ValueError('known task, nonnegative seed, positive even episode count and .02 cadence required')

    @classmethod
    def registered(cls, task, replicate):
        if task not in ('immediate','delayed') or type(replicate) is not int or not 0 <= replicate < 3:
            raise ValueError('known task and replicate 0..2 required')
        return cls(task,(3101 if task=='immediate' else 3201)+replicate)


class VisibleChoiceTask:
    """Balanced held-out scenes. Latch first action; retain cue and consequence."""
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        left = getattr(config,'left_count',config.episodes//2)
        self.sides = self.rng.permutation(np.array([0]*left+[1]*(config.episodes-left)))
        self.episode = -1
        self.elapsed_steps = 0
        self.selected = None
        self.active = False

    def reset(self):
        if self.active:
            raise ValueError('cannot reset an unfinished evaluation episode')
        if self.episode+1 >= self.config.episodes:
            raise StopIteration('evaluation schedule complete')
        self.episode += 1
        self.elapsed_steps = 0
        self.selected = None
        self.active = True
        return self.render()

    def render(self):
        if self.episode < 0:
            raise ValueError('start a fresh episode before rendering')
        frame = torch.zeros(64,64,dtype=torch.bool)
        column = (16,48)[int(self.sides[self.episode])]
        frame[15:18,column-1:column+2] = True
        if self.selected is not None:
            column = (16,48)[self.selected]
            frame[43:46,column-3:column+4] = True
        return frame

    def step(self, action):
        if not self.active or type(action) is not int or action not in (0,1):
            raise ValueError('active episode and legal action required')
        if self.selected is None:
            self.selected = action
        self.elapsed_steps += 1
        terminal = self.elapsed_steps == (1 if self.config.task=='immediate' else 8)
        reward = (1. if self.selected == self.sides[self.episode] else -1.) if terminal else 0.
        self.active = not terminal
        return self.render(),reward,terminal,self.config.period


class FreshFrozenEvaluator:
    def __init__(self, weights, config, *, random_action_seed=None):
        with preserve_global_rngs():
            model_bytes()  # Verify pinned prior before constructing fresh perception.
            self.config = config
            self.model_hash, self.vision_configuration = MODEL_HASH,VISION_CONFIGURATION
            self.task = VisibleChoiceTask(config)
            self.action_rng = None if random_action_seed is None else np.random.default_rng(random_action_seed)
            self.vision = load_default()
            self.encoder = VisualStateEncoder(64,64,config.period)
            self.policy = FrozenLinearPolicy(weights)
            if self.policy.weights.shape != (2,self.encoder.dimension):
                raise ValueError('two-action weights matching the pinned encoder required')
            self.previous = torch.zeros(64,64,dtype=torch.bool)
            self.encoded = None
            self.records = []
            self.transitions = 0
            self.initial_learned_hash = self.learned_hash()

    def learned_hash(self):
        # Frozen tensors plus calibration and private bank RNG state, excluding caches/history.
        return fingerprint((self.policy.weights,frozen_tensors(self),self.vision.dynamics,self.vision.calibration))

    def _observe(self, frame):
        events = torch.stack((self.previous & ~frame,~self.previous & frame)).float()
        self.previous.copy_(frame)
        state = self.vision.step(events,frame.float(),learn=False)
        self.encoded = self.encoder.encode(state,frame.float())

    def begin_episode(self):
        with preserve_global_rngs():
            frame = self.task.reset()
            self.vision.reset_state()
            self.encoder.reset_state()
            self.previous.zero_()
            self._observe(frame)

    def step(self):
        with preserve_global_rngs():
            if not self.task.active:
                self.begin_episode()
            action,q = self.policy.choose(self.encoded['indices'].numpy(),self.encoded['values'].numpy())
            if self.action_rng is not None:
                action = int(self.action_rng.integers(2))
            frame,reward,terminal,elapsed = self.task.step(action)
            self._observe(frame)
            self.transitions += 1
            record = dict(episode=self.task.episode,transition=self.transitions-1,action=action,
                          q=q.tolist(),reward=reward,terminated=terminal,truncated=False,
                          elapsed_seconds=elapsed)
            if self.learned_hash() != self.initial_learned_hash:
                raise RuntimeError('evaluation changed frozen parameters or bank state')
            self.records.append(record)
            return record

    @property
    def complete(self):
        return self.task.episode == self.config.episodes-1 and not self.task.active

    def run_chunk(self, max_transitions=200, max_seconds=180.):
        if (type(max_transitions) is not int or not 1 <= max_transitions <= 200
                or not 0 < max_seconds <= 180):
            raise ValueError('evaluation chunks require <=200 transitions and <=180 seconds')
        started = time.monotonic()
        rows = []
        while not self.complete and len(rows) < max_transitions:
            if time.monotonic()-started >= max_seconds:
                break
            rows.append(self.step())
        return dict(records=rows,complete=self.complete,wall_seconds=time.monotonic()-started)

# Compatibility name retained for the initial evaluator tests.
VisibleChoiceEvaluationTask = VisibleChoiceTask
