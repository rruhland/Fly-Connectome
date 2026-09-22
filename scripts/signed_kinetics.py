"""Bounded slow-excitation-v1 experiment, reference runner only (no checkpoint)."""
import math
import torch

from fly_connectome.dynamics import Network


class SignedKineticsNetwork(Network):
    """Existing signed predictive edges: 20 ms excitatory, 5 ms inhibitory.

    Each arrival still contributes its fixed sign times its current magnitude.
    Only predictive decay changes. Two local current stores permit distinct
    decays without new edges or access to future input. Native engines and
    production checkpoint serialization do not support this experimental state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excitatory_prediction = torch.zeros_like(self.voltage)
        self.inhibitory_prediction = torch.zeros_like(self.voltage)
        self.excitatory_decay = math.exp(-self.config.dt/.020)
        self.inhibitory_decay = math.exp(-self.config.dt/.005)

    def visual_decay(self, edges):
        return torch.where(self.signs[edges] > 0, self.excitatory_decay, self.inhibitory_decay)

    def _decay_prediction(self, leak):
        self.excitatory_prediction.mul_(self.excitatory_decay)
        self.inhibitory_prediction.mul_(self.inhibitory_decay)
        self.predictive_current.copy_(self.excitatory_prediction+self.inhibitory_prediction)

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        activity = super().step(sensory_current, capture_increments=capture_increments)
        env, edges = activity.arrival_environments, activity.arrival_edges
        targets = env*self.n+self.post[edges]
        weights = self.magnitudes[edges]*self.visual_arrival_impulse(env,edges)
        visual = self.pathways[edges] == 1
        for current, mask in ((self.excitatory_prediction, visual & (weights > 0)),
                              (self.inhibitory_prediction, visual & (weights < 0))):
            current.view(-1).index_add_(0, targets[mask], weights[mask])
        return activity


class AreaMatchedKineticsNetwork(SignedKineticsNetwork):
    """20 ms excitation with the discrete impulse area of a unit 5 ms synapse.

    Sum of an impulse A*r**k is A/(1-r). This factor matches that sum at
    the actual dt. Magnitudes remain plastic; the fixed waveform gain is not.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excitatory_gain = (1-self.excitatory_decay)/(1-self.inhibitory_decay)

    def visual_impulse(self, edges):
        signs = self.signs[edges]
        return signs*torch.where(signs > 0, self.excitatory_gain, 1.)
