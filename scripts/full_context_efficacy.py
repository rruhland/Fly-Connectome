"""Opt-in vectorized context efficacy for directly observed visual neurons."""
import math

import torch

from signed_kinetics import AreaMatchedKineticsNetwork


class MultiContextEfficacyNetwork(AreaMatchedKineticsNetwork):
    """Two local magnitudes only on predictive edges entering selected cells."""

    def __init__(self, *args, target_mask, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch != 1 or self.config.tau_sensory <= 0:
            raise ValueError('full context experiment requires B=1 and sensory decay')
        if target_mask.shape != (self.n,) or target_mask.dtype != torch.bool:
            raise ValueError('one boolean target flag per neuron required')
        predictive = self.pathways == 1
        connected = torch.bincount(self.post[predictive], minlength=self.n) > 0
        self.targets = (target_mask.to(self.device) & connected).nonzero().flatten()
        if not len(self.targets):
            raise ValueError('no directly observed predictive targets')
        target_lookup = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        target_lookup[self.targets] = torch.arange(len(self.targets), device=self.device)
        self.incoming = (predictive & (target_lookup[self.post] >= 0)).nonzero().flatten()
        self.incoming_lookup = torch.full((self.e,), -1, dtype=torch.long, device=self.device)
        self.incoming_lookup[self.incoming] = torch.arange(len(self.incoming), device=self.device)
        self.edge_targets = target_lookup[self.post[self.incoming]]
        self.component_positions = torch.arange(len(self.incoming), device=self.device)
        self.components = self.magnitudes[self.incoming].repeat(2, 1)
        self.context_exc = torch.zeros(2, self.batch, len(self.targets), device=self.device)
        self.context_inh = torch.zeros_like(self.context_exc)
        self.current_context = torch.zeros(self.batch, len(self.targets), dtype=torch.bool, device=self.device)
        self.target_lookup = target_lookup

    def set_weights(self, weights, components=None):
        self.magnitudes.copy_(weights)
        self.components.copy_(weights[self.incoming].repeat(2, 1) if components is None else components)
        self.magnitudes[self.incoming] = self.components[0]

    def _decay_prediction(self, leak):
        super()._decay_prediction(leak)
        self.context_exc.mul_(self.excitatory_decay)
        self.context_inh.mul_(self.inhibitory_decay)
        first = self.context_exc[0]+self.context_inh[0]
        second = self.context_exc[1]+self.context_inh[1]
        self.predictive_current[:, self.targets] = torch.where(self.current_context, second, first)

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        next_state = (self.sensory_state[:, self.targets]
                      * math.exp(-self.config.dt/self.config.tau_sensory)
                      + sensory_current[:, self.targets])
        self.current_context.copy_(next_state > 0)
        chosen = self.current_context[0, self.edge_targets].long()
        self.magnitudes[self.incoming] = self.components[chosen, self.component_positions]
        activity = super().step(sensory_current, capture_increments=capture_increments)
        self.magnitudes[self.incoming] = self.components[0]

        env, edges = activity.arrival_environments, activity.arrival_edges
        positions = self.incoming_lookup[edges]
        keep = positions >= 0
        env, edges, positions = env[keep], edges[keep], positions[keep]
        flat_targets = env*len(self.targets)+self.edge_targets[positions]
        impulses = self.visual_arrival_impulse(env, edges)
        for context in (0, 1):
            current = self.components[context, positions]*impulses
            excitatory = current > 0
            self.context_exc[context].view(-1).index_add_(
                0, flat_targets[excitatory], current[excitatory])
            self.context_inh[context].view(-1).index_add_(
                0, flat_targets[~excitatory], current[~excitatory])
        self.excitatory_prediction[:, self.targets] = torch.where(
            self.current_context, self.context_exc[1], self.context_exc[0])
        self.inhibitory_prediction[:, self.targets] = torch.where(
            self.current_context, self.context_inh[1], self.context_inh[0])
        return activity
