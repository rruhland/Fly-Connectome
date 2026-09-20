"""Experimental sleeping kernel must preserve every state bit, including outputs."""
import copy
import importlib.util
from pathlib import Path
import shutil

import pytest
import torch

from fly_connectome.dynamics import Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.native_cpu import NativeCPU, build_library

spec = importlib.util.spec_from_file_location('benchmark_sleeping',
    Path(__file__).parents[1] / 'scripts' / 'benchmark_sleeping.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('optional sleeping experiment requires g++')
    folder = tmp_path_factory.mktemp('sleeping')
    dense, sleeping = folder/'dense.dll', folder/'sleeping.dll'
    build_library(dense)
    experiment.build_sleeping(sleeping)
    return dense, sleeping


def assert_bits(a, b):
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    assert torch.equal(a, b)


@pytest.mark.parametrize('filtered', [False, True])
def test_exact_trajectory_autonomous_spikes_delays_inhibition_and_wake(kernels, filtered):
    dense_path, sleeping_path = kernels
    graph = Graph.from_contacts([10,20,30,40,50], [10,20,30], [20,30,40],
                                [1,1,1], [1,-1,1,1,1], .5)
    cfg = NeuronConfig(tau_sensory=.02 if filtered else 0.,
        class_parameters={'auto': {'rest_current': 1.5}, 'quiet': {'rest_current': .5}})
    net = Network(graph, [3,2,4], ['feedforward','predictive','behavioral'],
                  config=cfg, cell_types=['auto','','','quiet',''])
    net.voltage[0,3] = .5  # Exactly representable nonzero equilibrium.
    other = copy.deepcopy(net)
    dense = NativeCPU(dense_path)
    sleeping = experiment.SleepingCPU(dense_path, sleeping_path)
    saw_auto = saw_inhibition = saw_refractory = saw_delayed = False
    for tick in range(180):
        injection = torch.zeros(1,5)
        if tick == 30:
            injection[0,4] = 40.
        if tick == 70:
            injection[0,1] = 100.
        if tick == 100:
            injection[0,4] = -0.
        a = dense.step(net, injection, capture_increments=True)
        b = sleeping.step(other, injection, capture_increments=True)
        for name, value in vars(net).items():
            if isinstance(value, torch.Tensor):
                assert_bits(value, getattr(other, name))
        for name, value in vars(a).items():
            if value is not None:
                assert_bits(value, getattr(b, name))
        if tick == 1:
            assert sleeping.flags[other][3] == 1
            assert sleeping.flags[other][4] == 1
        if tick == 30:
            assert sleeping.flags[other][4] == 0
        saw_auto |= bool(a.spikes[0,0])
        saw_inhibition |= bool((net.predictive_current < 0).any())
        saw_refractory |= bool(net.refractory.any())
        saw_delayed |= bool(len(a.arrival_edges))
    assert saw_auto and saw_inhibition and saw_refractory and saw_delayed


def test_external_edits_require_explicit_cache_invalidation(kernels):
    dense_path, sleeping_path = kernels
    graph = Graph.from_contacts([10], [], [], [], [1], .5)
    net = Network(graph, [], [])
    other = copy.deepcopy(net)
    dense = NativeCPU(dense_path)
    sleeping = experiment.SleepingCPU(dense_path, sleeping_path)
    zero = torch.zeros(1,1)
    dense.step(net, zero)
    sleeping.step(other, zero)
    assert sleeping.flags[other][0] == 1
    for target in (net, other):
        target.voltage.fill_(2.)
        target.silenced.fill_(True)
    sleeping.invalidate(other)
    a, b = dense.step(net, zero), sleeping.step(other, zero)
    assert_bits(net.voltage, other.voltage)
    assert_bits(a.spikes, b.spikes)
    assert not b.spikes.any()


def test_signed_zero_input_and_subnormal_fixed_state_preserve_bits(kernels):
    dense_path, sleeping_path = kernels
    graph = Graph.from_contacts([10,20], [], [], [], [1,1], .5)
    net = Network(graph, [], [])
    # Multiplication can leave the smallest subnormal unchanged: do not zero it.
    net.predictive_current[0,1] = torch.tensor([1], dtype=torch.int32).view(torch.float32)[0]
    other = copy.deepcopy(net)
    dense = NativeCPU(dense_path)
    sleeping = experiment.SleepingCPU(dense_path, sleeping_path)
    for tick in range(5):
        injection = torch.zeros(1,2)
        if tick == 2:
            injection[0,0] = -0.
        a = dense.step(net, injection, capture_increments=True)
        b = sleeping.step(other, injection, capture_increments=True)
        for name in ('voltage','feedforward_current','predictive_current','behavioral_current',
                     'sensory_state','adaptation','refractory'):
            assert_bits(getattr(net,name), getattr(other,name))
        assert_bits(a.observed,b.observed)
        assert_bits(a.predicted,b.predicted)
        if tick == 1:
            assert (sleeping.flags[other] == 1).all()
        if tick == 2:
            assert sleeping.flags[other][0] == 0
    assert other.predictive_current.view(torch.int32)[0,1] == 1
