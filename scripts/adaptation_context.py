"""Approved local adaptation expression experiment; no physical-state changes."""
from dataclasses import replace
import torch

from signed_kinetics import AreaMatchedKineticsNetwork
from frame_prediction import FramePrediction


def context_gains(adaptation,threshold):
    u=adaptation/(threshold+adaptation)
    return 2*(1-u),2*u


class AdaptationContextNetwork(AreaMatchedKineticsNetwork):
    """A: larger adaptation favors inhibitory forecast expression only."""
    def forecast_gains(self,adaptation):
        return context_gains(adaptation,self.config.threshold)

    @torch.no_grad()
    def step(self,sensory_current,*,capture_increments=False):
        activity=super().step(sensory_current,capture_increments=capture_increments)
        ge,gi=self.forecast_gains(self.adaptation)
        # Algebraically g_E*E + g_I*I; preserve baseline accumulation roundoff
        # exactly when both gains are one (the required identity control).
        prediction=activity.predicted+(ge-1)*self.excitatory_prediction+(gi-1)*self.inhibitory_prediction
        return replace(activity,predicted=prediction)


class AdaptationExcitatoryNetwork(AdaptationContextNetwork):
    """B: same fixed gains with excitatory/inhibitory assignments reversed."""
    def forecast_gains(self,adaptation):
        ge,gi=super().forecast_gains(adaptation)
        return gi,ge


class AdaptationPrediction(FramePrediction):
    """Match sparse causal eligibility to each edge's issue-time local gain."""
    def _capture_forecast(self):
        super()._capture_forecast()
        keys,credit,prediction=self.forecast
        n=self.network
        env,edges=keys.div(n.e,rounding_mode='floor'),keys.remainder(n.e)
        ge,gi=n.forecast_gains(n.adaptation[env,n.post[edges]])
        self.forecast=(keys,credit*torch.where(n.signs[edges]>0,ge,gi),prediction)
