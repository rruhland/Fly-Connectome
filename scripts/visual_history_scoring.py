"""Frozen spatially shared readout and hidden-state evaluation measures."""

import torch


@torch.no_grad()
def fit_visible_readout(examples):
    positive = negative = None
    positive_count = negative_count = 0
    for state, mask in examples:
        flat = state.flatten(1)
        sites = mask.flatten()
        if positive is None:
            positive = torch.zeros(state.shape[0])
            negative = torch.zeros_like(positive)
        positive += flat[:, sites].sum(1)
        negative += flat[:, ~sites].sum(1)
        positive_count += int(sites.sum())
        negative_count += int((~sites).sum())
    weight = positive / max(positive_count, 1)
    weight -= negative / max(negative_count, 1)
    return lambda state: (weight[:, None, None] * state).sum(0)


@torch.no_grad()
def hidden_rank(field, hidden, *, k=32):
    if not bool(hidden.any()):
        raise ValueError('hidden target is empty')
    indices = field.flatten().topk(k).indices
    return float(hidden.flatten()[indices].any())


@torch.no_grad()
def pair_separation(left, right):
    distance = (left-right).norm()
    scale = (left.norm().square()+right.norm().square()).sqrt()
    return float(distance/scale.clamp(min=1e-8))
