"""Opt-in graded Mi4 release on existing inhibitory feedforward T4 edges."""
import math

import torch

from full_context_efficacy import MultiContextEfficacyNetwork


def blank_limited_release(network, cap=.05):
    """Tonic release that bounds the p99 steady added T4 current on blank."""
    if cap <= 0:
        raise ValueError('blank current cap must be positive')
    totals = torch.zeros(network.n, device=network.device)
    totals.index_add_(0, network.graded_posts,
                      network.magnitudes[network.graded_edges])
    active = totals[torch.unique(network.graded_posts)]
    leak = network.current_decay[torch.unique(network.graded_posts)]
    steady_for_unit_release = active/(1-leak)
    return float(cap/torch.quantile(steady_for_unit_release, .99))


class GradedMi4Network(MultiContextEfficacyNetwork):
    """Experimental release channel; production Network remains unchanged."""

    def __init__(self, *args, cell_types, base_release, voltage_scale,
                 release_fraction, shuffle=False, **kwargs):
        super().__init__(*args, cell_types=cell_types, **kwargs)
        if base_release <= 0 or voltage_scale <= 0 or not 0 < release_fraction <= 1:
            raise ValueError('positive bounded graded-release calibration required')
        mi4 = torch.tensor([name == 'Mi4' for name in cell_types], device=self.device)
        t4 = torch.tensor([name.startswith('T4') for name in cell_types], device=self.device)
        self.graded_edge_mask = (mi4[self.pre] & t4[self.post]
                                 & (self.pathways == 0))
        self.graded_edges = self.graded_edge_mask.nonzero().flatten()
        if not len(self.graded_edges) or not (self.signs[self.graded_edges] == -1).all():
            raise ValueError('graded Mi4->T4 edges must exist and retain inhibitory sign')
        self.mi4_nodes = mi4.nonzero().flatten()
        lookup = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        lookup[self.mi4_nodes] = torch.arange(len(self.mi4_nodes), device=self.device)
        self.graded_sources = lookup[self.pre[self.graded_edges]]
        self.graded_posts = self.post[self.graded_edges]
        self.graded_delays = self.delays[self.graded_edges]
        self.graded_history = torch.full((self.history_length, len(self.mi4_nodes)),
                                         base_release, device=self.device)
        self.graded_baseline = torch.zeros(len(self.mi4_nodes), device=self.device)
        self.graded_current = torch.zeros_like(self.voltage)
        self.last_graded_impulse = torch.zeros_like(self.voltage)
        self.base_release = float(base_release)
        self.release_slope = float(release_fraction*base_release/voltage_scale)
        self.baseline_decay = math.exp(-self.config.dt/.250)
        self.graded_enabled = False
        if shuffle:
            generator = torch.Generator(device=self.device).manual_seed(1211)
            self.release_permutation = torch.randperm(len(self.mi4_nodes),
                                                      generator=generator, device=self.device)
        else:
            self.release_permutation = None

    def configure_release(self, base_release, voltage_scale, release_fraction):
        if self.graded_enabled or base_release <= 0 or voltage_scale <= 0 or not 0 < release_fraction <= 1:
            raise ValueError('release must be configured before graded dynamics start')
        self.base_release = float(base_release)
        self.release_slope = float(release_fraction*base_release/voltage_scale)
        self.graded_history.fill_(base_release)

    def enable_graded(self):
        self.graded_baseline.copy_(self.voltage[0, self.mi4_nodes])
        self.graded_history.fill_(self.base_release)
        self.graded_current.zero_()
        self.last_graded_impulse.zero_()
        self.graded_enabled = True

    def _arrivals(self):
        environments, edges = super()._arrivals()
        if not self.graded_enabled:
            return environments, edges
        keep = ~self.graded_edge_mask[edges]
        return environments[keep], edges[keep]

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        tick = self.step_index
        self.last_graded_impulse.zero_()
        if self.graded_enabled:
            due = self.graded_history[(tick-self.graded_delays).remainder(
                self.history_length), self.graded_sources]
            impulse = self.magnitudes[self.graded_edges]*self.signs[self.graded_edges]*due
            self.last_graded_impulse[0].index_add_(0, self.graded_posts, impulse)
            leak = self.current_decay[self.graded_posts]
            self.feedforward_current[0].index_add_(0, self.graded_posts, impulse/leak)
            self.graded_current.mul_(self.current_decay)
            self.graded_current.add_(self.last_graded_impulse)
        activity = super().step(sensory_current, capture_increments=capture_increments)
        if self.graded_enabled:
            voltage = self.voltage[0, self.mi4_nodes]
            deviation = voltage-self.graded_baseline
            if self.release_permutation is not None:
                deviation = deviation[self.release_permutation]
            release = (self.base_release+self.release_slope*deviation).clamp(
                0., 2*self.base_release)
            self.graded_history[tick % self.history_length].copy_(release)
            self.graded_baseline.mul_(self.baseline_decay).add_(
                voltage, alpha=1-self.baseline_decay)
        return activity
