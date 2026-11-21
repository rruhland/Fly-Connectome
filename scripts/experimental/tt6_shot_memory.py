"""Recall shot outcomes from anonymous camera-fitted motor responses."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .tt4_arena import PADDLE_RADIUS
from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous import ContinualPlayer


def response_signature(response, settling):
    """Summarize unit-rate displacement at short, medium and settled horizons."""
    xyz = np.asarray(response)[:3] / PADDLE_RADIUS
    sampled = np.asarray(settling)[:, [1, 4, 7]]
    covariances = np.einsum('ac,bc,ct->tab', xyz, xyz, sampled**2)
    return np.r_[covariances.ravel(), (xyz @ xyz.T).ravel()] / 3.


class ShotMemory:
    """Estimate a local expected point reward with shrinkage toward zero."""

    def __init__(self, bandwidth, contexts=()):
        self.bandwidth = float(bandwidth)
        self.contexts = [(np.asarray(row['signature']), np.array(row['counts']), np.array(row['totals']))
                         for row in contexts]

    def estimate(self, signature):
        if not self.contexts:
            return np.zeros(4)
        keys, counts, totals = map(np.asarray, zip(*self.contexts, strict=True))
        distance = np.linalg.norm(keys - signature, axis=1)
        weight = np.exp(-.5 * (distance / self.bandwidth)**2)
        return (weight @ totals) / (weight @ counts + 4.)

    def observe(self, signature, option, reward):
        for key, counts, totals in self.contexts:
            if np.array_equal(key, signature):
                break
        else:
            counts, totals = np.zeros(4, dtype=int), np.zeros(4)
            self.contexts.append((signature.copy(), counts, totals))
        counts[option] += 1
        totals[option] += reward

    def state(self):
        return dict(schema='tt6-local-shot-memory-v1', bandwidth=self.bandwidth,
                    contexts=[dict(signature=key.tolist(), counts=counts.tolist(), totals=totals.tolist())
                              for key, counts, totals in self.contexts])

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'tt6-local-shot-memory-v1':
            raise ValueError('unsupported shot memory format')
        return cls(state['bandwidth'], state['contexts'])


class LocalShotPlayer(ContinualPlayer):
    def __init__(self, adapter, prior, factory, response, timing, *, shot_memory, **kwargs):
        self.shot_memory = shot_memory
        self.signature = response_signature(adapter.jacobian(np.zeros(adapter.action_size)), adapter.kernels)
        super().__init__(adapter, prior, factory, response, timing, **kwargs)

    def reset(self):
        super().reset()
        self.shot_estimates = self.shot_memory.estimate(self.signature)
        self.option = int(np.argmax(self.shot_estimates))

    def finish(self, reward):
        outcome = super().finish(reward)
        self.shot_memory.observe(self.signature, self.option, reward)
        outcome['local_shot_estimates'] = self.shot_estimates.tolist()
        return outcome


def collect(paths):
    histories = []
    for path in paths:
        for visit in _read(path)['visits']:
            fit = visit['fit']
            key = response_signature(fit['response_per_command'], fit['settling_curves'])
            histories.append((key, visit['points']))
    distinct = np.unique([key for key, _ in histories], axis=0)
    distances = cdist(distinct, distinct)
    np.fill_diagonal(distances, np.inf)
    bandwidth = float(np.median(distances.min(axis=1))) if len(distinct) > 1 else .20
    memory = ShotMemory(bandwidth)
    for signature, points in histories:
        for point in points:
            memory.observe(signature, point['option'], point['public_reward'])
    return dict(**memory.state(), inputs_sha256={path.name: _hash(path) for path in paths})


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('practice', nargs='+', type=Path)
    args = cli.parse_args()
    _write(args.output, collect(args.practice))
