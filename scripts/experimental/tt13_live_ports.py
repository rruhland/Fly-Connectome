"""Recognize a change in motor ports from the player's own in-game stereo observations."""
from dataclasses import replace
from itertools import permutations, product

import numpy as np
from scipy.signal import lfilter

from .tt6_body_control import POSE_WEIGHT
from .tt12_motor_recall import align, rewire


def effective(moves, kernels):
    q = moves.cumsum(axis=0)
    for i, kernel in enumerate(kernels):
        q[:, i] += lfilter(kernel - 1., [1.], moves[:, i])
    return q


def error(prediction, observations):
    diff = np.nan_to_num((prediction - observations) * POSE_WEIGHT)
    return np.mean(np.minimum(.32, np.linalg.norm(diff, axis=-1)), axis=-1)


def evaluate(model, moves, ticks, poses, train):
    q = effective(moves, model.kernels)
    predictions = model.pose(q)[np.maximum(0, ticks[None, :] - np.arange(5)[:, None])]
    train_cost = error(predictions[:, train], poses[train])
    lag = np.argmin(train_cost)
    return float(train_cost[lag]), float(error(predictions[lag, ~train], poses[~train]))


def proposals(source, moves, ticks, poses):
    n = source.action_size
    if n > 5:
        _, info = align(source, (moves[None], (np.zeros_like(ticks), ticks), poses))
        return [(tuple(info['order']),tuple(info['signs']))]
    # Screen correspondences with only linear motion on a subsample. Then
    # score the shortlisted models with their quadratic geometry and dynamics.
    stride = max(1, len(ticks)//20)
    obs = poses[::stride]
    when = np.maximum(0,ticks[::stride]-2)
    orders = np.array(list(permutations(range(n))))
    signs = np.array(list(product((-1,1), repeat=n)))
    # q[old,new,t] is the effective response of a new port under an old kernel.
    q = np.stack([(moves.cumsum(0) + lfilter(k-1., [1.], moves, axis=0))[when].T
                  for k in source.kernels])
    basis = q[orders, np.arange(n), :, None] * source.coefficients[1:1+n][orders, :, None].swapaxes(2,3)
    predictions = np.einsum('si,pitk->pstk', signs, basis) + source.home
    costs = error(predictions, obs)
    ranked = np.argsort(costs,axis=None)[:8]
    return [(tuple(orders[p].tolist()),tuple(signs[s].tolist())) for p,s in zip(*np.unravel_index(ranked,costs.shape))]


def local_model(reference, moves, ticks, poses):
    """Fit new linear motor directions with a generic (shared) settling curve."""
    n = reference.action_size
    kernels = np.broadcast_to(reference.kernels.mean(0), reference.kernels.shape).copy()
    q = effective(moves,kernels)
    candidates = []
    for lag in range(5):
        x = q[np.maximum(0,ticks-lag)]
        coef = reference.coefficients.copy()
        coef[1:] = 0.
        for j in range(4):
            seen = np.isfinite(poses[:,j])
            if seen.sum() <= n:
                continue
            z = x[seen]
            coef[1:n+1,j] = np.linalg.solve(z.T @ z + 1.5*np.eye(n), z.T @ (poses[seen,j]-coef[0,j]))
        model = replace(reference,coefficients=coef,kernels=kernels,lags=np.full(n,reference.lags.mean()))
        candidates.append((error(model.pose(x),poses),model))
    return min(candidates,key=lambda entry:entry[0])[1]


class PortMemory:
    def __init__(self, model, mode):
        self.mode = mode
        self.source = model
        self.models = [model]
        self.keys = [(tuple(range(model.action_size)), (1,)*model.action_size)]
        self.active = 0

    def observe(self, actions, trace):
        # These are requested actions and the existing player's public RGB blade
        # sightings, including the initial observation before the first action.
        if len(trace) != len(actions)+1:
            raise ValueError('unaligned visual and motor history')
        moves = np.vstack([np.zeros(self.source.action_size), actions])
        poses = np.array([[*row['blade'], row['blade_pitch'] if row['blade_pitch'] is not None else np.nan]
                          if row['blade'] is not None else [np.nan]*4 for row in trace])
        ticks = np.flatnonzero(np.isfinite(poses[:,:3]).all(axis=1))
        train = (ticks//4)%2 == 0
        if min(train.sum(),(~train).sum()) < 5:
            return dict(active=self.active, reason='insufficient sightings')
        observed = poses[ticks]
        current = evaluate(self.models[self.active],moves,ticks,observed,train)
        if self.mode == 'fixed' or max(current)<.065:
            return dict(active=self.active, reason='kept', errors=current)
        choices = list(zip(self.keys, self.models))
        if self.mode == 'remap':
            for key in proposals(self.source,moves,ticks[train],observed[train]):
                if key not in self.keys:
                    choices.append((key,rewire(self.source,*key)))
        elif self.mode == 'refit':
            choices.append((len(self.models),local_model(self.source,moves,ticks[train],observed[train])))
        scored = [(evaluate(m,moves,ticks,observed,train),i) for i,(_,m) in enumerate(choices)]
        # The candidate is chosen using only the training blocks. Independent
        # camera blocks from this point must support it before the next point.
        (fit,held),i = min(scored,key=lambda row:row[0][0])
        old = self.active
        if fit < .85*current[0] and held < .85*current[1] and held+.015 < current[1]:
            key,model = choices[i]
            if i >= len(self.models):
                self.keys.append(key)
                self.models.append(model)
                i = len(self.models)-1
            self.active = i
        return dict(previous=old, active=self.active, errors=current, candidate_errors=(fit,held),
                    bank=len(self.models))
