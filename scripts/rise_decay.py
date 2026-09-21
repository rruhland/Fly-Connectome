"""Experimental positive rise/decay kernel; reference runner only."""
import torch

from fly_connectome.dynamics import Network
from signed_kinetics import SignedKineticsNetwork
from frame_prediction import FramePrediction


class RiseDecayNetwork(SignedKineticsNetwork):
    """Excitatory kernel A*(r20**k-r5**k), area matched to a 5 ms impulse.

    The two positive stores are components of one nonnegative excitatory
    response, not an additional inhibitory edge. Arrival weights are retained
    in the stores; subsequent plasticity does not reweight existing current.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excitatory_rise = torch.zeros_like(self.voltage)
        self.kernel_gain = (1/(1-self.inhibitory_decay))/(1/(1-self.excitatory_decay)-1/(1-self.inhibitory_decay))

    def visual_impulse(self, edges):
        # Excitatory current starts at zero and rises on subsequent ticks.
        return self.signs[edges].clamp(max=0)

    def _decay_prediction(self, leak):
        self.excitatory_prediction.mul_(self.excitatory_decay)
        self.excitatory_rise.mul_(self.inhibitory_decay)
        self.inhibitory_prediction.mul_(self.inhibitory_decay)
        self.predictive_current.copy_(self.excitatory_prediction-self.excitatory_rise+self.inhibitory_prediction)

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        activity = Network.step(self, sensory_current, capture_increments=capture_increments)
        env, edges = activity.arrival_environments, activity.arrival_edges
        targets = env*self.n+self.post[edges]
        visual = self.pathways[edges] == 1
        positive = visual & (self.signs[edges] > 0)
        negative = visual & (self.signs[edges] < 0)
        for current in (self.excitatory_prediction, self.excitatory_rise):
            current.view(-1).index_add_(0, targets[positive], self.kernel_gain*self.magnitudes[edges[positive]])
        self.inhibitory_prediction.view(-1).index_add_(0, targets[negative], -self.magnitudes[edges[negative]])
        return activity


class RiseDecayPrediction(FramePrediction):
    """Local two-component eligibility, captured by the existing +8 tick rule."""

    def __init__(self, network, config, **kwargs):
        if network.batch != 1:
            raise ValueError('rise-decay experiment supports one environment')
        super().__init__(network, config, **kwargs)
        self.slow_trace = torch.zeros(network.e, device=network.device)
        self.fast_trace = torch.zeros_like(self.slow_trace)

    def _update_forecast_trace(self, activity):
        n = self.network
        self.slow_trace.mul_(n.excitatory_decay)
        self.fast_trace.mul_(n.inhibitory_decay)
        edges = activity.arrival_edges
        edges = edges[(n.pathways[edges] == 1) & (n.signs[edges] > 0)]
        arrival = torch.full_like(edges, n.kernel_gain, dtype=self.slow_trace.dtype)
        self.slow_trace.index_add_(0, edges, arrival)
        self.fast_trace.index_add_(0, edges, arrival)
        positive = (n.pathways[self.keys] == 1) & (n.signs[self.keys] > 0)
        self.values[positive] = self.slow_trace[self.keys[positive]]-self.fast_trace[self.keys[positive]]
