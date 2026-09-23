"""Opt-in frame-horizon local credit restricted to measured T5 target edges."""

import torch

from frame_prediction import FramePrediction
from fly_connectome.plasticity import Plasticity


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


class T5BalancedFramePrediction(T5FramePrediction):
    """Cell-local event-frequency gain on the same frame-horizon update."""

    def __init__(self, network, config, *, target_mask, **kwargs):
        super().__init__(network, config, target_mask=target_mask, **kwargs)
        self.target_nodes = target_mask.to(network.device).nonzero().flatten()
        self.event_count = torch.ones(network.n, device=network.device)
        self.quiet_count = torch.ones(network.n, device=network.device)
        self.last_event_gain = None
        self.event_gain_sum = 0.
        self.event_gain_samples = 0

    @torch.no_grad()
    def observe(self, activity, reward):
        n, cfg = self.network, self.config
        boundary = self.tick % 8 == 0
        self.last_visual_update = None
        self.last_event_gain = None
        if boundary:
            observed = cfg.observation(activity, n.config.threshold,
                                       self.sensory_mask, self.sensory_gain)
            if self.forecast is not None:
                keys, eligibility, prediction = self.forecast
                env, edges = keys.div(n.e, rounding_mode='floor'), keys.remainder(n.e)
                post = n.post[edges]
                target = observed[env, post]
                event = target.abs() > .01
                gain = torch.where(event,
                    (self.quiet_count[post]/self.event_count[post]).clamp(1, 4),
                    1.)
                delta = cfg.eta_prediction*(target-prediction)*eligibility*gain
                Plasticity._accumulate(self, edges, delta)
                self.last_visual_update = edges, target, delta
                if event.any():
                    self.last_event_gain = float(gain[event].mean())
                    self.event_gain_sum += float(gain[event].sum())
                    self.event_gain_samples += int(event.sum())
        Plasticity.observe(self, activity, reward)
        self._update_forecast_trace(activity)
        if boundary:
            local_event = observed[0, self.target_nodes].abs() > .01
            self.event_count[self.target_nodes] = (
                .98*self.event_count[self.target_nodes]+local_event.float())
            self.quiet_count[self.target_nodes] = (
                .98*self.quiet_count[self.target_nodes]+(~local_event).float())
            self._capture_forecast()
        self.tick += 1
