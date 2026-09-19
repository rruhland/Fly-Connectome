import math

import pytest
import torch

from fly_connectome.dynamics import Activity, Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig, Plasticity


def test_input_increments_do_not_change_forward_dynamics_or_include_decay_tail():
    graph = Graph.from_contacts([10,20], [10], [20], [1], [1,1], .4)
    cfg = NeuronConfig(tau_sensory=.02)
    a, b = Network(graph,[1],['feedforward'],config=cfg), Network(graph,[1],['feedforward'],config=cfg)
    rule = LearningConfig(prediction_encoding='signed-current-v1', visual_target='input-arrivals-v1')
    for tick in range(30):
        current = torch.tensor([[-30. if tick == 0 else 0., 0.]])
        x, y = a.step(current), b.step(current, capture_increments=True)
        torch.testing.assert_close(a.voltage, b.voltage, atol=0, rtol=0)
        assert torch.equal(x.spikes, y.spikes)
        target = rule.observation(y, 1., torch.tensor([True,False]), 30.)
        assert target[0,0] == (-1 if tick == 0 else 0)
    assert y.observed[0,0] < -1 and not target.any()


def test_feedforward_arrival_target_excludes_existing_synaptic_current():
    graph = Graph.from_contacts([10,20],[10],[20],[1],[1,1],.4)
    net = Network(graph,[1],['feedforward'])
    cfg = LearningConfig(prediction_encoding='signed-current-v1',visual_target='input-arrivals-v1')
    net.step(torch.tensor([[30.,0.]]),capture_increments=True)
    arrival = net.step(torch.zeros(1,2),capture_increments=True)
    tail = net.step(torch.zeros(1,2),capture_increments=True)
    mask = torch.tensor([True,False])
    assert cfg.observation(arrival,1.,mask,30.)[0,1] == pytest.approx(.4)
    assert cfg.observation(tail,1.,mask,30.)[0,1] == 0
    assert tail.observed[0,1] > 0


@pytest.mark.parametrize('sign', [-1,1])
@pytest.mark.parametrize('observed,factor', [(0.,-1),(.4,0),(.8,1)])
def test_causal_error_uses_previous_forecast_trace(sign, observed, factor):
    graph = Graph.from_contacts([10,20],[10],[20],[1],[sign,1],.5)
    n = Network(graph,[1],['predictive'])
    p = Plasticity(n, LearningConfig(prediction_encoding='signed-current-v1',
        visual_eligibility='forecast-causal-v1', eta_prediction=.1,homeostasis_rate=0.))
    def sample(value, arrived):
        return Activity(torch.zeros(1,2,dtype=torch.bool), torch.tensor([[0.,sign*value]]),
            torch.tensor([[0.,sign*.4]]), torch.tensor([0] if arrived else [],dtype=torch.long),
            torch.tensor([0] if arrived else [],dtype=torch.long))
    p.observe(sample(.8, True),torch.ones(1))
    p.synchronize()
    assert n.magnitudes[0] == .5  # this arrival did not produce the previous forecast
    p.observe(sample(observed,False),torch.ones(1))
    p.synchronize()
    assert n.magnitudes[0].item() == pytest.approx(.5+.04*factor)
    assert p.values.item() == pytest.approx(sign*math.exp(-n.config.dt/n.config.tau_current))


@pytest.mark.parametrize('sign', [-1,1])
def test_local_rule_learns_delayed_association_and_beats_frozen_and_zero(sign):
    # Hand-controlled arrivals isolate local learning from emergent graph activity.
    # One measured fixture edge; the teacher is a fixed delayed input sequence.
    graph = Graph.from_contacts([10,20],[10],[20],[1],[sign,1],.8)
    net = Network(graph,[1],['predictive'])
    rule = Plasticity(net,LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',
        eta_prediction=.03,homeostasis_rate=0.))
    arrivals = torch.rand(2000,generator=torch.Generator().manual_seed(3)) < .08
    trace = 0.
    errors = torch.zeros(3)
    for tick, arrived in enumerate(arrivals):
        target = sign*.2*trace
        if tick >= 1700:
            errors += torch.tensor([(target-rule.expected[0,1].item())**2,
                                    (target-sign*.8*trace)**2,target**2])
        trace = trace*math.exp(-net.config.dt/net.config.tau_current)+int(arrived)
        teacher = torch.tensor([[0.,target]])
        activity = Activity(torch.zeros(1,2,dtype=torch.bool), teacher,
            torch.tensor([[0.,sign*net.magnitudes[0].item()*trace]]),
            torch.tensor([0] if arrived else [],dtype=torch.long),
            torch.tensor([0] if arrived else [],dtype=torch.long),
            feedforward_arrivals=teacher,sensory_input=torch.zeros(1,2))
        rule.observe(activity,torch.zeros(1))
        rule.synchronize()
    assert net.magnitudes[0].item() == pytest.approx(.2, abs=.001)
    assert errors[0] < .01*errors[1] and errors[0] < .01*errors[2]


def test_causal_visual_setting_leaves_behavioral_pair_learning_unchanged():
    graph = Graph.from_contacts([10,20],[10],[20],[1],[1,1],.4)
    nets = [Network(graph,[1],['behavioral']) for _ in range(2)]
    rules = [Plasticity(net,LearningConfig(visual_eligibility=version)) for net,version in
             zip(nets,('legacy-v1','forecast-causal-v1'))]
    for tick in range(100):
        sensory = torch.tensor([[30. if tick%5 == 0 else 0.,30. if tick%7 == 0 else 0.]])
        for net,rule in zip(nets,rules):
            rule.observe(net.step(sensory),torch.tensor([.1 if tick%2 else -.2]))
            rule.synchronize()
    for name in ('values','arrival_trace','post_trace','rates'):
        torch.testing.assert_close(getattr(rules[0],name),getattr(rules[1],name),atol=0,rtol=0)
    torch.testing.assert_close(nets[0].magnitudes,nets[1].magnitudes,atol=0,rtol=0)


def test_event_rule_exact_resume_preserves_separate_target_metrics(tmp_path):
    from test_training import trainer
    from fly_connectome.sensor import Retina
    from fly_connectome.training import Trainer, load_checkpoint
    base = trainer()
    cfg = LearningConfig(prediction_encoding='signed-current-v1',visual_target='input-arrivals-v1',
                         visual_eligibility='forecast-causal-v1')
    a = Trainer(base.network.graph,[1]*3,['predictive','behavioral','behavioral'],
        Retina(**dict(base.retina.spec,injection={'L2':'contrast'})),[30],[40],[1],
        config=base.config,neurons=NeuronConfig(dt=base.network.config.dt,tau_sensory=.02),learning=cfg)
    a.run(5)
    path = tmp_path/'event.pt'
    a.save(path)
    b = load_checkpoint(path)
    a.run(8); b.run(8)
    assert a.metrics == b.metrics
    assert a.metrics['learning_squared_error'] != a.metrics['prediction_squared_error']
    assert a.previous_learning_observed[0,0] == 0
    for x,y,keys in ((a,b,('previous_predicted','previous_learning_observed','learning_statistics')),
                     (a.plasticity,b.plasticity,('keys','values','expected','proposals')),
                     (a.network,b.network,('voltage','magnitudes','sensory_state'))):
        for key in keys:
            torch.testing.assert_close(getattr(x,key),getattr(y,key),atol=0,rtol=0)


def run_batched_rule(amplitudes, device='cpu'):
    graph = Graph.from_contacts([10,20],[10],[20],[1],[-1,1],.4)
    net = Network(graph,[1],['predictive'],batch=len(amplitudes),device=device)
    cfg = LearningConfig(prediction_encoding='signed-current-v1',visual_target='input-arrivals-v1',
                         visual_eligibility='forecast-causal-v1',eta_prediction=.01,homeostasis_rate=0.)
    rule = Plasticity(net,cfg)
    values = torch.tensor(amplitudes,device=device)
    for tick in range(50):
        target = torch.stack((torch.zeros_like(values),-values if tick%3 == 1 else torch.zeros_like(values)),1)
        arrived = tick%3 == 0
        rule.observe(Activity(torch.zeros(len(values),2,dtype=torch.bool,device=device),target,
            torch.stack((torch.zeros_like(values),-values*.5),1),
            torch.arange(len(values),device=device) if arrived else torch.empty(0,dtype=torch.long,device=device),
            torch.zeros(len(values) if arrived else 0,dtype=torch.long,device=device),
            feedforward_arrivals=target,sensory_input=torch.zeros_like(target)),torch.zeros(len(values),device=device))
        if tick%2:
            rule.synchronize()
    return net.magnitudes.cpu()


def test_event_rule_duplicate_and_permuted_batch_equivalence():
    torch.testing.assert_close(run_batched_rule([.4]),run_batched_rule([.4,.4]),atol=0,rtol=0)
    torch.testing.assert_close(run_batched_rule([.2,.8]),run_batched_rule([.8,.2]),atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_event_rule_cuda_matches_cpu():
    torch.testing.assert_close(run_batched_rule([.2,.8]),run_batched_rule([.2,.8],'cuda'),atol=1e-6,rtol=1e-5)
