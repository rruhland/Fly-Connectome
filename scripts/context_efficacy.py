"""Experimental local context efficacy on existing visual synapses."""
import math

import torch

from signed_kinetics import AreaMatchedKineticsNetwork
from timed_local_prediction import TimedLocalPrediction


class ContextEfficacyNetwork(AreaMatchedKineticsNetwork):
    """Two local magnitudes per target edge, with exact summed synaptic currents."""

    def __init__(self, *args, target, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch != 1 or self.config.tau_sensory <= 0:
            raise ValueError('context efficacy requires one sensory-decay environment')
        self.target = int(target)
        self.incoming = ((self.post == target) & (self.pathways == 1)).nonzero().flatten()
        self.incoming_lookup = torch.full((self.e,), -1, dtype=torch.long, device=self.device)
        self.incoming_lookup[self.incoming] = torch.arange(len(self.incoming), device=self.device)
        self.components = self.magnitudes[self.incoming].repeat(2, 1)
        self.context_exc = torch.zeros(2, 1, device=self.device)
        self.context_inh = torch.zeros(2, 1, device=self.device)
        self.current_context = 0

    def set_weights(self, weights, components=None):
        self.magnitudes.copy_(weights)
        self.components.copy_(weights[self.incoming].repeat(2, 1) if components is None else components)
        self.magnitudes[self.incoming] = self.components[0]

    def _decay_prediction(self, leak):
        super()._decay_prediction(leak)
        self.context_exc.mul_(self.excitatory_decay)
        self.context_inh.mul_(self.inhibitory_decay)
        self.predictive_current[0, self.target] = (self.context_exc[self.current_context, 0]
                                                    + self.context_inh[self.current_context, 0])

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        next_state = (self.sensory_state[0, self.target]
                      * math.exp(-self.config.dt/self.config.tau_sensory)
                      + sensory_current[0, self.target])
        self.current_context = int(next_state > 0)
        self.magnitudes[self.incoming] = self.components[self.current_context]
        activity = super().step(sensory_current, capture_increments=capture_increments)
        self.magnitudes[self.incoming] = self.components[0]

        env, edges = activity.arrival_environments, activity.arrival_edges
        positions = self.incoming_lookup[edges]
        keep = positions >= 0
        env, edges, positions = env[keep], edges[keep], positions[keep]
        impulses = self.visual_arrival_impulse(env, edges)
        for context in (0, 1):
            current = self.components[context, positions]*impulses
            excitatory = current > 0
            self.context_exc[context].index_add_(0, env[excitatory], current[excitatory])
            self.context_inh[context].index_add_(0, env[~excitatory], current[~excitatory])
        self.excitatory_prediction[0, self.target] = self.context_exc[self.current_context, 0]
        self.inhibitory_prediction[0, self.target] = self.context_inh[self.current_context, 0]
        return activity


class ContextTimedPrediction(TimedLocalPrediction):
    """Assign eight-tick local error to only the magnitude used at issue."""

    def __init__(self, network, config, *, target, **kwargs):
        if not isinstance(network, ContextEfficacyNetwork) or target != network.target:
            raise ValueError('context rule requires the matching context network target')
        super().__init__(network, config, target=target, **kwargs)
        self.context_proposals = torch.zeros_like(network.components)

    @torch.no_grad()
    def observe(self, activity, reward):
        prior_context = self.last_issue['context'] if self.last_issue is not None else None
        super().observe(activity, reward)
        if self.last_confirmation is not None:
            edges = self.last_confirmation['edges']
            positions = self.network.incoming_lookup[edges]
            self.context_proposals[prior_context].index_add_(
                0, positions, self.last_confirmation['delta'])
            self.proposals[edges] = 0
            self.last_confirmation['context'] = prior_context
        if self.tick % 8 == 1:
            self.last_issue['context'] = self.network.current_context

    @torch.no_grad()
    def synchronize(self):
        super().synchronize()
        self.network.components.add_(self.context_proposals).clamp_(0, self.config.maximum_weight)
        self.network.magnitudes[self.network.incoming] = self.network.components[0]
        self.context_proposals.zero_()
