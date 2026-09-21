"""Experimental frame-horizon-v1 local supervision; production rule unchanged."""
import torch

from fly_connectome.plasticity import Plasticity


class FramePrediction(Plasticity):
    """Issue at ticks 0,8,... and confirm eight ticks later, including quiet frames.

    Snapshot only local visual-edge eligibility and the postsynaptic prediction.
    Live traces, behavioral updates and homeostasis still advance every tick.
    Unconfirmed forecasts at sequence end are discarded, not treated as failures.
    This experimental state is not supported by production checkpoint/resume.
    """

    def __init__(self, network, config, **kwargs):
        if config.visual_eligibility != 'forecast-causal-v1':
            raise ValueError('frame horizon requires causal visual eligibility')
        super().__init__(network, config, **kwargs)
        self.tick = 0
        self.forecast = None
        self.last_visual_update = None

    def _accumulate(self, edges, proposals):
        # Suppress the base rule's one-tick visual targets only.
        behavioral = self.network.pathways[edges] == 2
        super()._accumulate(edges[behavioral], proposals[behavioral])

    @torch.no_grad()
    def observe(self, activity, reward):
        n, cfg = self.network, self.config
        boundary = self.tick % 8 == 0
        self.last_visual_update = None
        if boundary and self.forecast is not None:
            keys, eligibility, prediction = self.forecast
            env, edges = keys.div(n.e, rounding_mode='floor'), keys.remainder(n.e)
            observed = cfg.observation(activity, n.config.threshold, self.sensory_mask, self.sensory_gain)
            target = observed[env, n.post[edges]]
            delta = cfg.eta_prediction * (target-prediction) * eligibility
            super()._accumulate(edges, delta)
            self.last_visual_update = edges, target, delta
        super().observe(activity, reward)
        if boundary:
            edges = self.keys.remainder(n.e)
            visual = n.pathways[edges] == 1
            keys = self.keys[visual].clone()
            env, edges = keys.div(n.e, rounding_mode='floor'), keys.remainder(n.e)
            self.forecast = (keys, self.values[visual].clone(), self.expected[env, n.post[edges]].clone())
        self.tick += 1
