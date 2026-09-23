"""Opt-in frame-horizon local credit restricted to measured T5 target edges."""

import torch

from frame_prediction import FramePrediction


class T5FramePrediction(FramePrediction):
    """Same local update equation, with only T5 target forecasts retained."""

    def __init__(self, network, config, *, target_mask, **kwargs):
        super().__init__(network, config, **kwargs)
        if target_mask.shape != (network.n,) or target_mask.dtype != torch.bool:
            raise ValueError('one boolean target flag per neuron required')
        self.learnable_edges = ((network.pathways == 1)
                                & target_mask.to(network.device)[network.post])
        if not self.learnable_edges.any():
            raise ValueError('no measured predictive edges into targets')

    def _capture_forecast(self):
        super()._capture_forecast()
        keys, eligibility, prediction = self.forecast
        keep = self.learnable_edges[keys.remainder(self.network.e)]
        self.forecast = keys[keep], eligibility[keep], prediction[keep]
