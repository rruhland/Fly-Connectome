"""Separate local memory current from an interval-change event forecast."""
from dataclasses import replace
import torch

from signed_kinetics import AreaMatchedKineticsNetwork
from frame_prediction import FramePrediction


class EventContrastNetwork(AreaMatchedKineticsNetwork):
    """Physical 20/5 ms area-matched dynamics; forecast is I[t]-I[t-8].

    No contrast signal is fed back into the membrane. The current buffer is
    local to each neuron and advances during normal warmup as well as stimuli.
    Experimental reference runner only, no production checkpoint/native support.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.forecast_history=torch.zeros(8,self.batch,self.n,device=self.device)

    @torch.no_grad()
    def step(self,sensory_current,*,capture_increments=False):
        slot=self.step_index%8
        activity=super().step(sensory_current,capture_increments=capture_increments)
        contrast=activity.predicted-self.forecast_history[slot]
        self.forecast_history[slot].copy_(activity.predicted)
        return replace(activity,predicted=contrast)


class EventContrastPrediction(FramePrediction):
    """Credit the difference of the two local issue-time memory traces."""

    def __init__(self,network,config,**kwargs):
        if network.batch!=1:
            raise ValueError('event-contrast experiment supports one environment')
        super().__init__(network,config,**kwargs)
        self.previous_frame_trace=torch.zeros(network.e,device=network.device)

    def _capture_forecast(self):
        n=self.network
        live=torch.zeros_like(self.previous_frame_trace)
        live[self.keys]=self.values
        contrast=live-self.previous_frame_trace
        visual=(n.pathways==1)&(contrast.abs()>self.config.prune_epsilon)
        keys=visual.nonzero().flatten()
        self.forecast=(keys,contrast[keys].clone(),self.expected[0,n.post[keys]].clone())
        self.previous_frame_trace.copy_(live)
