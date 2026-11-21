"""Identify a signed motor installation from one simultaneous 12-command RGB probe."""
from itertools import permutations, product

import numpy as np
from scipy.signal import lfilter

from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe


def excitation(source):
    """Choose a bounded tape from the old settling curve, before observing the rig."""
    size = source.action_size
    rng = np.random.default_rng(610352 + size)
    # Two-tick pulses exercise every port concurrently. Favor distinguishable
    # traces after smoothing by the old actuator's average settling response.
    draws = rng.choice([-.7, .7], size=(2048, 5, size))
    candidates = np.pad(np.repeat(draws, 2, axis=1), ((0, 0), (0, 2), (0, 0)))
    cumulative = np.cumsum(candidates, axis=1)
    bounded = np.max(abs(cumulative), axis=(1, 2)) < 3.0
    candidates, cumulative = candidates[bounded], cumulative[bounded]
    if not len(candidates):
        raise ValueError('no excitation within the command excursion budget')
    response = cumulative + lfilter(source.kernels.mean(axis=0) - 1., [1.], candidates, axis=1)
    centered = response - response.mean(axis=1, keepdims=True)
    # Even when some port motions overlap, the full trace should resolve each
    # port's contribution to each observed position or orientation coordinate.
    singular = np.linalg.svd(centered, compute_uv=False)
    return candidates[np.argmax(singular[:, -1])]


def prediction_bank(source, actions):
    size = source.action_size
    cmds = np.vstack((np.zeros(size), actions))
    sums = np.cumsum(cmds, axis=0)
    # Each old channel has its own response curve.
    filtered = np.stack([sums + lfilter(k - 1., [1.], cmds, axis=0) for k in source.kernels])
    signs = np.array(list(product((-1, 1), repeat=size)))
    poses, keys = [], []
    for order in permutations(range(size)):
        inverse = np.argsort(order)
        response = filtered[np.arange(size), :, inverse].T
        poses.append(source.pose(response[None, :, :] * signs[:, None, inverse]))
        keys.extend([list(order), sign.tolist()] for sign in signs)
    return keys, np.concatenate(poses)


def correspondence_cost(source, predicted, observed):
    # Scale by the old arm's ordinary movement per axis. The paddle-radius
    # weight used for striking undervalues pitch when identifying a motor.
    gain = np.linalg.norm(source.jacobian(np.zeros(source.action_size)), axis=1)
    scale = np.median(gain[:3]) / np.maximum(gain, .01)
    residual = predicted - observed[None, :, :]
    xyz = residual[:, :, :3]
    xyz -= np.median(xyz, axis=1, keepdims=True)
    squared = np.sum((xyz * scale[:3])**2, axis=2)
    visible = np.isfinite(observed[:, 3])
    if np.any(visible):
        pitch = residual[:, visible, 3]
        pitch -= np.median(pitch, axis=1, keepdims=True)
        squared[:, visible] += (scale[3] * pitch)**2
    return np.minimum(np.sqrt(squared), .25).mean(axis=1)


def decode(source, actions, ticks, seen):
    """Enumerate signed correspondences against the saved nonlinear camera model."""
    size = source.action_size
    if len(ticks) < 3:
        return [list(range(size)), [1] * size], dict(observed=len(ticks), best_error_m=None, margin_m=None)
    keys, poses = prediction_bank(source, actions)
    costs = correspondence_cost(source, poses[:, ticks], np.array(seen, dtype=float))
    two = np.partition(costs, 1)[:2]
    return keys[costs.argmin()], dict(observed=len(ticks), best_error_m=float(two[0]), margin_m=float(two[1]-two[0]))


def probe(source, wiring, seed, tracker):
    commands = excitation(source)
    history = collect_probe(wiring, seed, commands)
    actions = [a.tolist() for a, _ in history[1:]]
    _, (_, ticks), optical = _samples([history], tracker)
    seen = [[float(x) if np.isfinite(x) else None for x in row] for row in optical]
    key, fit = decode(source, actions, ticks, seen)
    return dict(key=key, fit=fit, actions=actions, ticks=ticks.tolist(), seen=seen,
                stereo_pairs=len(history), elapsed_s=len(actions)*wiring.arena.control_dt)
