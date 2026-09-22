"""Opt-in vectorized context efficacy for directly observed visual neurons."""
from dataclasses import replace
import math

import torch

from frame_prediction import FramePrediction
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


class MultiContextTimedPrediction(FramePrediction):
    """Frame-horizon local credit gated and selected independently per target."""

    def __init__(self, network, config, **kwargs):
        if not isinstance(network, MultiContextEfficacyNetwork):
            raise ValueError('full context rule requires matching neural dynamics')
        super().__init__(network, config, **kwargs)
        if not self.sensory_mask[network.targets].all():
            raise ValueError('all context targets need direct local sensory input')
        self.reference = torch.zeros(len(network.targets), device=network.device)
        self.previous_state = torch.zeros_like(self.reference)
        self.gate = torch.zeros(len(network.targets), dtype=torch.bool, device=network.device)
        self.quiet_count = torch.ones_like(self.reference)
        self.event_count = torch.ones_like(self.reference)
        self.margin = math.exp(-4*network.config.dt/network.config.tau_sensory)
        self.context_proposals = torch.zeros_like(network.components)
        self.last_issue = None
        self.last_confirmation = None

    def _capture_forecast(self):
        super()._capture_forecast()
        n = self.network
        keys, eligibility, prediction = self.forecast
        edges = keys.remainder(n.e)
        positions = n.target_lookup[n.post[edges]]
        keep = positions >= 0
        positions = positions[keep]
        gate = self.gate[positions].float()
        self.forecast = keys[keep], eligibility[keep]*gate, prediction[keep]*gate

    @torch.no_grad()
    def observe(self, activity, reward):
        boundary = self.tick % 8 == 0
        self.last_confirmation = None
        if boundary:
            n = self.network
            observed = self.config.observation(activity, n.config.threshold,
                                               self.sensory_mask, self.sensory_gain)
            event = observed[0, n.targets] != 0
            due = self.forecast is not None
            if due:
                gain = torch.where(event,
                    (self.quiet_count/self.event_count).clamp(1, 8), 1.)
                keys, eligibility, prediction = self.forecast
                positions = n.target_lookup[n.post[keys.remainder(n.e)]]
                self.forecast = keys, eligibility*gain[positions], prediction
                prior_context = self.last_issue['context']
                self.last_confirmation = dict(gain=gain.clone())
            current_state = n.sensory_state[0, n.targets]
            self.reference[event] = self.previous_state[event].abs()
            value = current_state.abs()
            self.gate = ((self.reference > 0)
                         & (self.reference*self.margin <= value)
                         & (value <= self.reference/self.margin))
            self.previous_state.copy_(current_state)
        # All other predictive and behavioral magnitudes are frozen in this
        # M1A bridge, so their arrivals need no eligibility or pair state.
        arrivals = self.network.incoming_lookup[activity.arrival_edges] >= 0
        learning_activity = replace(activity,
            arrival_environments=activity.arrival_environments[arrivals],
            arrival_edges=activity.arrival_edges[arrivals])
        super().observe(learning_activity, reward)
        if boundary:
            n = self.network
            if self.last_confirmation is not None:
                edges, targets, delta = self.last_visual_update
                positions = n.incoming_lookup[edges]
                local_targets = n.edge_targets[positions]
                contexts = prior_context[local_targets].long()
                flat = contexts*len(n.incoming)+positions
                self.context_proposals.view(-1).index_add_(0, flat, delta)
                self.proposals[edges] = 0
                self.last_confirmation.update(edges=edges.clone(), targets=targets.clone(),
                                              delta=delta.clone(), context=contexts.bool().clone())
            raw = self.config.encode(activity.predicted[0, n.targets], n.config.threshold)
            keys, eligibility, _ = self.forecast
            self.last_issue = dict(tick=self.tick-1, context=n.current_context[0].clone(),
                gate=self.gate.clone(), reference=self.reference.clone(),
                sensory_state=self.previous_state.clone(), raw_prediction=raw.clone(),
                prediction=raw*self.gate, edges=keys.remainder(n.e).clone(),
                eligibility=eligibility.clone())
            self.event_count.add_(event.float())
            self.quiet_count.add_((~event).float())

    @torch.no_grad()
    def synchronize(self):
        super().synchronize()
        n = self.network
        n.components.add_(self.context_proposals).clamp_(0, self.config.maximum_weight)
        n.magnitudes[n.incoming] = n.components[0]
        self.context_proposals.zero_()


class AlwaysOpenContextPrediction(MultiContextTimedPrediction):
    """Use the physical forecast and local eligibility without a timing window."""

    def _capture_forecast(self):
        FramePrediction._capture_forecast(self)
        n = self.network
        keys, eligibility, prediction = self.forecast
        edges = keys.remainder(n.e)
        keep = n.target_lookup[n.post[edges]] >= 0
        self.forecast = keys[keep], eligibility[keep], prediction[keep]

    @torch.no_grad()
    def observe(self, activity, reward):
        super().observe(activity, reward)
        if self.tick % 8 == 1:
            self.last_issue['prediction'] = self.last_issue['raw_prediction'].clone()
