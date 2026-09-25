"""Frozen diagnostic for mass-calibrated sparse local-event emission."""

import torch


@torch.no_grad()
def select_by_mass(forecast, *, gain):
    flat = forecast.flatten()
    count = min(flat.numel(), max(0, int(gain*float(flat.sum())+.5)))
    selected = torch.zeros_like(flat, dtype=torch.bool)
    if count:
        selected[flat.topk(count).indices] = True
    return selected.view_as(forecast)
