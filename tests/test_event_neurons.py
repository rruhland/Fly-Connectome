"""Experimental regrouped neurons: test scheduling, not bitwise equivalence."""
import copy
import importlib.util
from pathlib import Path
import shutil

import pytest
import torch
from fly_connectome.dynamics import Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.native_cpu import NativeCPU, build_library


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    path = Path(__file__).parents[1]/'scripts/benchmark_event_neurons.py'
    assert path.exists(), 'event neuron experiment is not implemented'
    spec = importlib.util.spec_from_file_location('event_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not shutil.which('g++'):
        pytest.skip('optional native experiment requires g++')
    folder = tmp_path_factory.mktemp('event')
    native, event = folder/'native.dll', folder/'event.dll'
    build_library(native)
    module.build_event(event)
    return module, native, event


@pytest.mark.parametrize('horizon', [1, 32, 128])
@pytest.mark.parametrize('filtered', [False, True])
def test_autonomous_delayed_signed_and_refractory_trajectory(experiment, horizon, filtered):
    module, native, event = experiment
    graph = Graph.from_contacts([10,20,30,40], [10,20,30], [20,30,40],
                                [1,1,1], [1,-1,1,1], .5)
    config = NeuronConfig(tau_sensory=.02 if filtered else 0.,
        class_parameters={'auto': {'rest_current': 1.5},
                          'equal': {'tau_membrane': .02, 'tau_current': .02}})
    dense = Network(graph, [3,2,4], ['feedforward','predictive','behavioral'],
                    config=config, cell_types=['auto','equal','',''])
    lazy = copy.deepcopy(dense)
    reference, candidate = NativeCPU(native), module.EventCPU(native,event,horizon=horizon)
    spikes = 0
    for tick in range(400):
        injection = torch.zeros(1,4)
        if tick in (30,70,110):
            injection[0,1] = 40.
        a = reference.step(dense,injection,capture_increments=True)
        b = candidate.step(lazy,injection,capture_increments=True)
        assert torch.equal(a.spikes,b.spikes), f'first changed spike at {tick}'
        torch.testing.assert_close(a.observed,b.observed,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(a.predicted,b.predicted,atol=2e-5,rtol=2e-5)
        assert torch.equal(a.feedforward_arrivals,b.feedforward_arrivals)
        spikes += int(a.spikes.sum())
    candidate.materialize(lazy)
    for name in ('voltage','feedforward_current','predictive_current','behavioral_current',
                 'sensory_state','adaptation','refractory'):
        torch.testing.assert_close(getattr(dense,name),getattr(lazy,name),atol=2e-5,rtol=2e-5)
    assert spikes > 0
    if horizon > 1:
        assert candidate.cache[lazy]['counters'][1] > 0


def test_materialize_then_external_edit_and_invalidate(experiment):
    module, native, event = experiment
    graph = Graph.from_contacts([10], [], [], [], [1], .5)
    net = Network(graph, [], [])
    kernel = module.EventCPU(native,event,horizon=128)
    for _ in range(12):
        kernel.step(net,torch.zeros(1,1))
    kernel.materialize(net)
    assert kernel.cache[net]['last'].item() == net.step_index-1
    net.voltage.fill_(2.)
    net.silenced.fill_(True)
    kernel.invalidate(net)
    assert not kernel.step(net,torch.zeros(1,1)).spikes.any()
    kernel.materialize(net)
    assert net.voltage.item() == 0.


def test_nonmonotone_signed_currents_and_adaptation(experiment):
    module, native, event = experiment
    graph = Graph.from_contacts([10,20], [], [], [], [1,1], .5)
    config = NeuronConfig(tau_current=.04,tau_sensory=.003)
    dense = Network(graph, [], [],config=config)
    dense.feedforward_current[:] = torch.tensor([[3.,-3.]])
    dense.sensory_state[:] = torch.tensor([[-7.,7.]])
    dense.adaptation[:] = .2
    lazy = copy.deepcopy(dense)
    a, b = NativeCPU(native), module.EventCPU(native,event,horizon=128)
    count = 0
    for tick in range(250):
        x,y = a.step(dense,torch.zeros(1,2)),b.step(lazy,torch.zeros(1,2))
        assert torch.equal(x.spikes,y.spikes), tick
        count += int(x.spikes.sum())
    assert count > 0
    b.materialize(lazy)
    torch.testing.assert_close(dense.voltage,lazy.voltage,atol=2e-5,rtol=2e-5)


def test_snapshot_checkpoint_resume_materializes_current_state(experiment,tmp_path):
    from test_training import trainer
    from fly_connectome.training import load_checkpoint
    module, native, event = experiment
    model = trainer()
    kernel = module.EventCPU(native,event,horizon=128)
    kernel.enable(model)
    model.run(3)
    model.snapshot()
    assert (kernel.cache[model.network]['last'] == model.network.step_index-1).all()
    model.run(2)
    path = tmp_path/'event.pt'
    model.save(path)
    resumed = load_checkpoint(path)
    assert resumed.manifest['execution_experiment'] == 'event-neurons-v1'
    other = module.EventCPU(native,event,horizon=128)
    other.enable(resumed)
    model.run(5)
    resumed.run(5)
    kernel.materialize(model.network)
    other.materialize(resumed.network)
    for owner in ('network','plasticity'):
        for name,value in vars(getattr(model,owner)).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,getattr(getattr(resumed,owner),name),atol=0,rtol=0)


def test_backend_switch_flushes_and_removes_old_materializer(experiment):
    from test_training import trainer
    module, native, event = experiment
    model = trainer()
    kernel = module.EventCPU(native,event)
    kernel.enable(model)
    model.run(3)
    dense = NativeCPU(native)
    dense.enable(model)
    assert not hasattr(model,'_materialize')
    assert (kernel.cache[model.network]['last']==model.network.step_index-1).all()
    model.run(3)
    before = model.network.voltage.clone()
    model.snapshot()
    assert torch.equal(before,model.network.voltage)


def test_overflow_cannot_be_certified_as_a_quiet_interval(experiment):
    module, native, event = experiment
    graph = Graph.from_contacts([10], [], [], [], [1], .5)
    net = Network(graph, [], [])
    # First tick is refractory, masking voltage overflow but leaving huge currents.
    net.refractory.fill_(1)
    net.feedforward_current.fill_(2.5e38)
    net.sensory_state.fill_(2.5e38)
    from dataclasses import replace
    net.config = replace(net.config,tau_sensory=100.,tau_current=100.,threshold=1e38)
    net.predictive_current.fill_(-2.5e38)
    net.behavioral_current.fill_(-2.5e38)
    other = copy.deepcopy(net)
    dense, lazy = NativeCPU(native),module.EventCPU(native,event,horizon=128)
    for tick in range(3):
        a,b = dense.step(net,torch.zeros(1,1)),lazy.step(other,torch.zeros(1,1))
        assert torch.equal(a.spikes,b.spikes), tick
