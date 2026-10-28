"""Anonymous paired-reward option memory; no gradients or privileged inputs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from fly_connectome.control.scaling import policy_features
from pong_continuous_visual import canonical_scene, save_scenes, load_scenes
from pong_fixed_rbf import FreshRBFEvaluator

SCHEMA = 'anonymous-paired-option-memory-v1'
HOLD = 16


def scene_rows(scene):
    rows = np.asarray(scene, dtype=float).reshape(-1, 6).copy()
    assert np.isfinite(rows).all() and np.isin(rows[:, 4:], [0, 1]).all()
    rows[:, :2] = np.clip(rows[:, :2], -1., 1.)
    rows[:, 2:4] /= 1. + np.abs(rows[:, 2:4])
    return rows


def row_distance(left, right):
    """Assignment ignores entity IDs/order; unmatched content has fixed unit cost."""
    n = max(len(left), len(right))
    if n == 0:
        return 0.
    costs = np.ones((n, n))
    costs[len(left):, len(right):] = 0.
    if len(left) and len(right):
        costs[:len(left), :len(right)] = np.mean(
            (left[:, None, :] - right[None, :, :]) ** 2, axis=2) / 4.
    a, b = linear_sum_assignment(costs)
    return float(costs[a, b].sum() / n)


def scene_distance(left, right):
    return row_distance(scene_rows(left), scene_rows(right))


class OptionMemory:
    schema = SCHEMA

    def __init__(self, scenes, advantages, valid, radius):
        self.scenes = tuple(np.asarray(s, dtype=float).reshape(-1, 6).copy() for s in scenes)
        self.rows = tuple(scene_rows(s) for s in self.scenes)
        self.advantages = np.asarray(advantages, dtype=float).copy()
        self.valid = np.asarray(valid, dtype=bool).copy()
        self.radius = float(radius)
        assert self.advantages.shape == self.valid.shape == (len(scenes), 3)
        assert np.isfinite(self.advantages).all() and np.isfinite(self.radius) and self.radius > 0
        for value in (*self.scenes, *self.rows, self.advantages, self.valid):
            value.flags.writeable = False

    def inspect(self, scene):
        rows = scene_rows(scene)
        distances = np.array([row_distance(rows, other) for other in self.rows])
        neighbors = distances <= self.radius
        support = neighbors[:, None] & self.valid
        counts = support.sum(0)
        means = np.divide((support * self.advantages).sum(0), counts,
                          out=np.zeros(3), where=counts > 0)
        positive = (counts > 0) & (means > 0.)
        option = int(np.argmax(np.where(positive, means, -np.inf))) if positive.any() else 3
        return dict(option=option, mean_advantages=means.tolist(), support_counts=counts.tolist(),
                    neighbor_sites=np.flatnonzero(neighbors).tolist(),
                    nearest_distance=float(distances.min()) if len(distances) else None)

    def frozen_parameters(self):
        return SCHEMA, self.scenes, self.advantages, self.valid, self.radius, HOLD


def save_memory(path, memory):
    lengths = np.array([len(s) for s in memory.scenes], dtype=np.int64)
    with open(path, 'wb') as stream:
        np.savez_compressed(stream, kind=SCHEMA,
            entities=np.concatenate(memory.scenes) if memory.scenes else np.empty((0, 6)),
            indptr=np.r_[0, np.cumsum(lengths)], advantages=memory.advantages,
            valid=memory.valid, radius=memory.radius, hold=HOLD)


def load_memory(path):
    with np.load(path, allow_pickle=False) as f:
        assert str(f['kind']) == SCHEMA and int(f['hold']) == HOLD
        scenes = [f['entities'][a:b].copy() for a, b in zip(f['indptr'][:-1], f['indptr'][1:])]
        return OptionMemory(scenes, f['advantages'], f['valid'], float(f['radius']))


class MemoryController:
    """Only an earned alternative holds; baseline is reconsidered every camera."""
    def __init__(self, memory):
        self.memory = memory
        self.remaining = 0
        self.held_action = None

    def choose(self, scene, baseline_action):
        before = self.remaining
        if before:
            action = self.held_action
            self.remaining -= 1
            decision = dict(option=action, mode='earned_hold', hold_before=before,
                            hold_after=self.remaining)
        else:
            decision = self.memory.inspect(scene)
            action = int(baseline_action) if decision['option'] == 3 else decision['option']
            self.held_action = None if decision['option'] == 3 else action
            self.remaining = 0 if decision['option'] == 3 else HOLD - 1
            decision.update(mode='baseline_camera' if decision['option'] == 3 else 'earned_start',
                            hold_before=0, hold_after=self.remaining)
        decision.update(action=int(action), baseline_action=int(baseline_action))
        return int(action), decision


class OptionMemoryEvaluator(FreshRBFEvaluator):
    def __init__(self, original, memory, config):
        self.option_memory = memory
        self.controller = MemoryController(memory)
        self.option_choices = []
        super().__init__(original, config)
        self.initial_learned_hash = self.learned_hash()

    def learned_hash(self):
        from fly_connectome.control.durable import fingerprint
        previous = super().learned_hash()
        return fingerprint((previous, self.option_memory.frozen_parameters()))

    def _begin_episode(self):
        self.controller = MemoryController(self.option_memory)
        super()._begin_episode()

    def _choose_policy(self):
        baseline_action, q = self.rbf_readout.choose(*policy_features(self.latest_encoded, self.scaler))
        action, choice = self.controller.choose(canonical_scene(self.latest_encoded), baseline_action)
        self.option_choices.append(choice)
        return action, q
