import math
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from rise_decay import RiseDecayNetwork, RiseDecayPrediction
from fly_connectome.graph import Graph
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.plasticity import LearningConfig


def test_delayed_peak_equal_area_and_matching_local_eligibility():
    graph = Graph.from_contacts([10,20,30], [10,20], [30,30], [1,1], [1,-1,1], .5)
    dt=1/960
    net=RiseDecayNetwork(graph,[1,1],['predictive']*2,config=NeuronConfig(dt=dt,threshold=100.))
    rule=RiseDecayPrediction(net,LearningConfig(visual_target='input-arrivals-v1',
                            visual_eligibility='forecast-causal-v1',homeostasis_rate=0.))
    slow,fast=math.exp(-dt/.020),math.exp(-dt/.005)
    gain=(1/(1-fast))/(1/(1-slow)-1/(1-fast))
    net.step(torch.zeros(1,3));net.history[0,0,:2]=True
    impulses=[]
    for tick in range(400):
        activity=net.step(torch.zeros(1,3),capture_increments=True)
        positive=gain*(slow**tick-fast**tick)
        expected=torch.tensor([positive,-fast**tick])
        if tick<30:
            rule.observe(activity,torch.zeros(1))
            torch.testing.assert_close(rule.values,expected,atol=2e-7,rtol=2e-6)
            if tick % 8 == 0:
                torch.testing.assert_close(rule.forecast[1],expected,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(activity.predicted[0,2],.5*expected.sum(),atol=2e-7,rtol=2e-6)
        impulses.append(float(net.excitatory_prediction[0,2]-net.excitatory_rise[0,2]))
    assert impulses[0]==0
    assert min(impulses)>=-1e-7
    assert 7<=max(range(len(impulses)),key=impulses.__getitem__)<=10
    assert abs(sum(impulses)*dt/(.5*dt/(1-fast))-1)<2e-6


def test_rise_decay_rejects_unmatched_one_tick_learning():
    from controlled_visual import run_sequence
    graph=Graph.from_contacts([10,20],[10],[20],[1],[1,1],.5)
    net=RiseDecayNetwork(graph,[1],['predictive'])
    with pytest.raises(ValueError,match='requires frame-horizon'):
        run_sequence(net,None,{},[],[],learning=True)
