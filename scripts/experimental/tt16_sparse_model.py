"""Use the ranked search-motion fit to choose two individual RGB motor queries."""

from itertools import permutations, product

import numpy as np

from .tt13_live_ports import effective
from .tt15_pitch_probe import optical_error


SEARCH_SCALE = .012
QUERY_SCALE = .005
QUERY_COUNT = 2


def candidate_bank(width):
    orders = np.array(list(permutations(range(width))))
    signs = np.array(list(product([-1, 1], repeat=width)))
    return np.repeat(orders, len(signs), axis=0), np.tile(signs, (len(orders), 1))


def motion_ranking(reference, actions, positions, curve):
    """Preserve each TT15 search score instead of taking only the best key."""
    width = reference.shape[1]
    orders, signs = candidate_bank(width)
    bank = reference.T[orders] * signs[..., None]
    path = np.asarray(positions)
    states = effective(np.vstack((np.zeros(width), actions)), np.tile(curve, (width, 1)))
    times = np.arange(max(20, len(path) - 96), len(path), 3)
    best = np.full(len(orders), np.inf)
    for lag in (0, 2, 4, 6):
        errors = np.zeros(len(orders))
        for span in (5, 12):
            requested = states[times-lag] - states[times-lag-span]
            movement = path[times] - path[times-span]
            errors += np.linalg.norm(requested @ bank - movement, axis=2).clip(max=.15).mean(axis=1)
        best = np.minimum(best, errors * .5)
    return orders, signs, best


def next_port(orders, signs, costs, used):
    """Probe the largest remaining uncertainty over an axis and its sign."""
    width = orders.shape[1]
    probabilities = np.exp(np.min(costs) - costs)
    probabilities /= probabilities.sum()
    entropy = np.full(width, -np.inf)
    labels = 2 * orders + (signs > 0)
    for port in sorted(set(range(width)) - set(used)):
        weights = np.bincount(labels[:, port], weights=probabilities, minlength=2 * width)
        active = weights > 0.
        entropy[port] = -float(weights[active] @ np.log(weights[active]))
    return int(np.argmax(entropy))


def port_scores(source, port, samples):
    scores = np.zeros((source.action_size, 2))
    for axis in range(source.action_size):
        for polarity, sign in enumerate((-1, 1)):
            order = list(range(source.action_size))
            signs = [1] * source.action_size
            order[port], order[axis] = order[axis], order[port]
            signs[port] = sign
            loss = optical_error(source, (order, signs), samples)
            xyz, pitch = loss['xyz_error_m'], loss['pitch_error_rad']
            scores[axis, polarity] = np.inf if xyz is None else (
                xyz if pitch is None else float(np.hypot(xyz, .12 * pitch)))
    if not np.isfinite(scores).any():
        raise ValueError('the sparse RGB query did not observe its movement')
    return scores


def best_keys(orders, signs, search_costs, queries):
    sparse_costs = np.zeros(len(orders))
    for port, losses in queries:
        sparse_costs += losses[orders[:, port], (signs[:, port] > 0).astype(int)]
    assisted = search_costs / SEARCH_SCALE + sparse_costs / QUERY_SCALE
    # Among equally supported assignments, retain as much of the original map as possible.
    change = np.count_nonzero(orders != np.arange(orders.shape[1]), axis=1)
    negative = np.count_nonzero(signs < 0, axis=1)
    close = np.flatnonzero(sparse_costs <= np.min(sparse_costs) + .001)
    control = min(close, key=lambda i: (change[i], negative[i], i))
    choose = int(np.argmin(assisted))
    return ([orders[choose].tolist(), signs[choose].tolist()],
            [orders[control].tolist(), signs[control].tolist()], assisted)
