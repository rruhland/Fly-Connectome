import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from event_contrast import EventContrastNetwork, EventContrastPrediction
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.graph import Graph
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.plasticity import LearningConfig


def networks():
    graph=Graph.from_contacts([10,20],[10],[20],[1],[1,1],.5)
    cfg=NeuronConfig(dt=1/960,threshold=100.)
    return [cls(graph,[1],['predictive'],config=cfg) for cls in
            (EventContrastNetwork,AreaMatchedKineticsNetwork)]


def test_forecast_is_eight_tick_local_contrast_without_changing_physics():
    event,memory=networks();history=[]
    for tick in range(32):
        for net in (event,memory):
            if tick==1: net.history[0,0,0]=True
        a,b=[net.step(torch.zeros(1,2),capture_increments=True) for net in (event,memory)]
        history.append(b.predicted.clone())
        expected=b.predicted-(history[tick-8] if tick>=8 else 0)
        torch.testing.assert_close(a.predicted,expected,rtol=0,atol=0)
        for name in ('voltage','predictive_current','adaptation','history'):
            torch.testing.assert_close(getattr(event,name),getattr(memory,name),rtol=0,atol=0)
    assert history[-1][0,1]>0  # Memory remains positive even as forecast contrast turns negative.
    assert a.predicted[0,1]<0


def test_matching_contrast_credit_and_confirmation_uses_issue_snapshot():
    net,_=networks()
    rule=EventContrastPrediction(net,LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',eta_prediction=.1,homeostasis_rate=0.))
    previous=torch.zeros(net.e)
    for tick in range(25):
        if tick==1: net.history[0,0,0]=True
        activity=net.step(torch.zeros(1,2),capture_increments=True)
        forecast=rule.forecast
        before=rule.proposals.clone()
        rule.observe(activity,torch.zeros(1))
        if tick%8==0:
            if forecast is not None:
                keys,eligibility,prediction=forecast
                expected=torch.zeros(net.e)
                expected[keys]=-.1*prediction*eligibility
                torch.testing.assert_close(rule.proposals-before,expected,atol=1e-9,rtol=1e-5)
            live=torch.zeros(net.e);live[rule.keys]=rule.values
            keys,eligibility,_=rule.forecast
            captured=torch.zeros(net.e);captured[keys]=eligibility
            torch.testing.assert_close(captured,live-previous,atol=1e-8,rtol=0)
            previous=live.clone()
        else:
            torch.testing.assert_close(rule.proposals,before,rtol=0,atol=0)
