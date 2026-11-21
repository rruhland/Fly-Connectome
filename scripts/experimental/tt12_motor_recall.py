"""Identify the motor port correspondence of a previously camera-calibrated arm."""
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import lfilter

from .tt6_body_control import POSE_WEIGHT


def rewire(source, order, signs):
    """New port i delivers signs[i] times old port order[i]."""
    n = source.action_size
    order, signs = np.asarray(order), np.asarray(signs)
    if sorted(order.tolist()) != list(range(n)) or signs.shape != (n,) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('expected a signed permutation of the source motor ports')
    coeff = source.coefficients.copy()
    coeff[1:1+n] = source.coefficients[1+order] * signs[:, None]
    left, right = np.triu_indices(n)
    lookup = np.empty((n, n), dtype=int)
    lookup[left, right] = lookup[right, left] = np.arange(len(left))
    coeff[n+1:] = source.coefficients[n+1+lookup[order[left], order[right]]] * (signs[left]*signs[right])[:, None]
    return replace(source, coefficients=coeff, kernels=source.kernels[order].copy(), lags=source.lags[order].copy())


def predicted(model, commands, indices):
    effective = np.cumsum(commands, axis=1)
    for i, kernel in enumerate(model.kernels):
        effective[:, :, i] += lfilter(kernel - 1., [1.], commands[:, :, i], axis=1)
    return model.pose(effective[indices])


def loss(model, samples):
    commands, indices, observed = samples
    diff = np.nan_to_num((predicted(model, commands, indices) - observed) * POSE_WEIGHT)
    return float(np.mean(np.minimum(np.linalg.norm(diff, axis=1), .25)))


def align(source, samples):
    """Initialize from local directions, then test swaps and signs against RGB motion."""
    commands, indices, observed = samples
    approx = np.cumsum(commands, axis=1) + lfilter(source.kernels.mean(0)-1., [1.], commands, axis=1)
    design = approx[indices]
    centered = design - design.mean(0)
    target = np.nan_to_num(observed - np.nanmedian(observed, axis=0)) * POSE_WEIGHT
    measured = np.linalg.solve(centered.T @ centered + .1*np.eye(source.action_size), centered.T @ target)
    template = source.jacobian(np.zeros(source.action_size)).T * POSE_WEIGHT
    pos = np.linalg.norm(measured[:,None,:] - template[None,:,:], axis=2)
    neg = np.linalg.norm(measured[:,None,:] + template[None,:,:], axis=2)
    rows, order = linear_sum_assignment(np.minimum(pos,neg))
    signs = np.where(pos[rows,order] <= neg[rows,order], 1, -1)
    best = loss(rewire(source,order,signs),samples)
    for _ in range(3):
        previous = best
        new_order, new_signs = order.copy(), signs.copy()
        for i in range(source.action_size):
            for j in range(i,source.action_size):
                perm = order.copy()
                perm[[i,j]] = order[[j,i]]
                for s in (-1,1):
                    for t in (-1,1) if i != j else (s,):
                        sign = signs.copy()
                        sign[i], sign[j] = s,t
                        error = loss(rewire(source,perm,sign),samples)
                        if error < best - 1e-10:
                            best, new_order, new_signs = error,perm,sign
        order, signs = new_order, new_signs
        if best == previous:
            break
    return rewire(source,order,signs), dict(error_m=best, order=order.tolist(), signs=signs.tolist())
