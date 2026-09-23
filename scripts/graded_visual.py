"""Opt-in graded release on selected visual neurons' measured outgoing edges."""

import math

import torch

from full_context_efficacy import MultiContextEfficacyNetwork


def graded_release(voltage, baseline, floor, *, scale, cap):
    if scale <= 0 or not 0 < cap <= 1:
        raise ValueError('positive voltage scale and bounded release cap required')
    return ((voltage-baseline-floor)/scale).clamp(0, cap)


class GradedVisualNetwork(MultiContextEfficacyNetwork):
    """Experimental T2 output; all measured outgoing edges keep signs/delays."""

    def __init__(self, *args, cell_types, release_cap, voltage_scale, **kwargs):
        super().__init__(*args, cell_types=cell_types, **kwargs)
        if voltage_scale <= 0 or not 0 < release_cap <= 1:
            raise ValueError('positive voltage scale and bounded release cap required')
        source_mask = torch.tensor([name == 'T2' for name in cell_types],
                                   device=self.device)
        self.graded_nodes = source_mask.nonzero().flatten()
        self.graded_edge_mask = source_mask[self.pre]
        self.graded_edges = self.graded_edge_mask.nonzero().flatten()
        if not len(self.graded_edges):
            raise ValueError('no measured T2 output edges')
        lookup = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        lookup[self.graded_nodes] = torch.arange(len(self.graded_nodes),
                                                device=self.device)
        self.graded_sources = lookup[self.pre[self.graded_edges]]
        self.graded_posts = self.post[self.graded_edges]
        self.graded_delays = self.delays[self.graded_edges]
        self.release_history = torch.zeros(self.history_length,
                                           len(self.graded_nodes),
                                           device=self.device)
        self.graded_baseline = torch.zeros(len(self.graded_nodes),
                                           device=self.device)
        self.blank_residual = torch.zeros(128, len(self.graded_nodes),
                                          device=self.device)
        self.blank_count = 0
        self.graded_floor = torch.zeros(len(self.graded_nodes), device=self.device)
        self.last_graded_impulse = torch.zeros_like(self.voltage)
        self.release_cap = float(release_cap)
        self.voltage_scale = float(voltage_scale)
        self.baseline_decay = math.exp(-self.config.dt/.250)
        self.graded_enabled = False

    def enable_graded(self):
        if self.blank_count < len(self.blank_residual):
            raise ValueError('calibrate local release floor on 128 blank ticks')
        self.graded_floor.copy_(torch.quantile(self.blank_residual, .99, dim=0))
        self.release_history.zero_()
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
            edges, posts = self.graded_edges, self.graded_posts
            due = self.release_history[(tick-self.graded_delays).remainder(
                self.history_length), self.graded_sources]
            pathways = self.pathways[edges]
            edge_impulse = torch.where(pathways == 1,
                self.visual_impulse(edges), self.signs[edges])
            impulse = self.magnitudes[edges]*edge_impulse*due
            leak = self.current_decay[posts]
            for path, current in ((0, self.feedforward_current),
                                  (2, self.behavioral_current)):
                chosen = pathways == path
                current[0].index_add_(0, posts[chosen],
                                      impulse[chosen]/leak[chosen])
            predictive = pathways == 1
            positions = self.incoming_lookup[edges]
            ordinary = predictive & (positions < 0)
            for sign_mask, store, decay in (
                    (impulse > 0, self.excitatory_prediction,
                     self.excitatory_decay),
                    (impulse < 0, self.inhibitory_prediction,
                     self.inhibitory_decay)):
                chosen = ordinary & sign_mask
                store[0].index_add_(0, posts[chosen], impulse[chosen]/decay)
            contextual = predictive & (positions >= 0)
            if contextual.any():
                target_positions = self.target_lookup[posts[contextual]]
                edge_positions = positions[contextual]
                signed_release = self.visual_impulse(edges[contextual])*due[contextual]
                for context in (0, 1):
                    component_impulse = (self.components[context, edge_positions]
                                         * signed_release)
                    for positive, store, decay in (
                            (True, self.context_exc, self.excitatory_decay),
                            (False, self.context_inh, self.inhibitory_decay)):
                        chosen = component_impulse > 0 if positive else component_impulse < 0
                        store[context, 0].index_add_(
                            0, target_positions[chosen],
                            component_impulse[chosen]/decay)
                next_state = (self.sensory_state[:, self.targets]
                              * math.exp(-self.config.dt/self.config.tau_sensory)
                              + sensory_current[:, self.targets])
                selected_context = (next_state[0, target_positions] > 0).long()
                impulse[contextual] = (self.components[selected_context, edge_positions]
                                       * signed_release)
            self.last_graded_impulse[0].index_add_(0, posts, impulse)
        activity = super().step(sensory_current,
                                capture_increments=capture_increments)
        voltage = self.voltage[0, self.graded_nodes]
        residual = (voltage-self.graded_baseline).clamp(min=0)
        if not self.graded_enabled:
            self.blank_residual[tick % len(self.blank_residual)].copy_(residual)
            self.blank_count += 1
        else:
            self.release_history[tick % self.history_length].copy_(
                graded_release(voltage, self.graded_baseline,
                               self.graded_floor, scale=self.voltage_scale,
                               cap=self.release_cap))
        self.graded_baseline.mul_(self.baseline_decay).add_(
            voltage, alpha=1-self.baseline_decay)
        return activity
