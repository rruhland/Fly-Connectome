import torch
import pytest

from fly_connectome.graph import Graph
from fly_connectome.dynamics import Network, NeuronConfig


def chain(device='cpu', batch=1, delays=(2,)):
    graph = Graph.from_contacts([10, 20], [10], [20], [1], [1, 1], 2.)
    config = NeuronConfig(dt=1., tau_membrane=1., tau_current=1., threshold=1.,
                          refractory_steps=2, adaptation_jump=.2)
    return Network(graph, delays, ['feedforward'], batch=batch, config=config, device=device)


def test_fixed_delay_and_discrete_spikes():
    net = chain()
    outputs = [net.step(torch.tensor([[2., 0.]])).spikes.tolist()]
    outputs += [net.step(torch.zeros(1, 2)).spikes.tolist() for _ in range(2)]
    assert outputs == [[[True, False]], [[False, False]], [[False, True]]]
    assert net.adaptation[0, 0] > 0


def test_refractory_and_quiet_no_forced_activity():
    net = chain()
    assert [net.step(torch.tensor([[2., 0.]])).spikes[0, 0].item() for _ in range(4)] == [True, False, False, True]
    quiet = chain()
    for _ in range(100):
        assert not quiet.step(torch.zeros(1, 2)).spikes.any()


def test_environment_private_state_and_no_autograd():
    net = chain(batch=2)
    net.step(torch.tensor([[2., 0.], [0., 0.]], requires_grad=True))
    net.step(torch.zeros(2, 2))
    out = net.step(torch.zeros(2, 2))
    assert out.spikes.tolist() == [[False, True], [False, False]]
    for value in vars(net).values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.grad_fn is None


def test_feedforward_and_prediction_currents_keep_anatomy_partition():
    graph = Graph.from_contacts([10, 20, 30], [10, 20], [30, 30], [1, 1], [1]*3, 2.)
    net = Network(graph, [1, 1], ['feedforward', 'predictive'], config=NeuronConfig(dt=1., tau_membrane=1.))
    net.step(torch.tensor([[2., 2., 0.]]))
    out = net.step(torch.zeros(1, 3))
    assert out.observed[0, 2] > 0
    assert out.predicted[0, 2] > 0
    assert out.arrival_edges.tolist() == [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_cpu_cuda_trace_parity():
    cpu, cuda = chain(), chain('cuda')
    for step in range(30):
        x = torch.tensor([[2. if step % 5 == 0 else 0., 0.]])
        a, b = cpu.step(x), cuda.step(x.cuda())
        assert torch.equal(a.spikes, b.spikes.cpu())
        torch.testing.assert_close(cpu.voltage, cuda.voltage.cpu(), atol=1e-5, rtol=1e-5)


def test_resting_current_disinhibition_uses_only_existing_edge():
    graph = Graph.from_contacts([10, 20, 30], [10], [20], [1], [-1, 1, 1], 4.)
    cfg = NeuronConfig(dt=.001, adaptation_jump=0., class_parameters={
        'L1': {'rest_current': 1.5}, 'Mi1': {'rest_current': 1.2}})
    net = Network(graph, [1], ['feedforward'], config=cfg, cell_types=['L1', 'Mi1', 'quiet'])
    zeros = torch.zeros(1, 3)
    baseline = torch.zeros(3, dtype=torch.int64)
    for tick in range(1000):
        a = net.step(zeros)
        if tick >= 500:
            baseline += a.spikes[0]
    released = torch.zeros(3, dtype=torch.int64)
    for tick in range(500):
        a = net.step(torch.tensor([[-3., 0., 0.]]))
        released += a.spikes[0]
        assert a.observed[0, 1] <= 0  # intrinsic bias is not an observation
    assert baseline[0] > 0 and released[0] == 0
    assert released[1] > baseline[1]
    assert baseline[2] == released[2] == 0
    assert net.e == 1 and net.signs.tolist() == [-1.]


def test_signed_sensory_current_decays_without_becoming_prediction():
    from dataclasses import replace
    net = chain()
    net.config = replace(net.config, tau_sensory=2.)
    first = net.step(torch.tensor([[-2., 0.]]))
    second = net.step(torch.zeros(1, 2))
    assert first.observed[0, 0] == -2
    torch.testing.assert_close(second.observed[0, 0], torch.tensor(-2 * __import__('math').exp(-.5)))
    assert not second.predicted.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_cpu_cuda_class_dynamics_and_signed_currents():
    graph = Graph.from_contacts([10, 20], [10], [20], [1], [-1, 1], 4.)
    config = NeuronConfig(tau_sensory=.02, class_parameters={
        'L1': {'rest_current': 1.5, 'tau_membrane': .03}, 'Mi1': {'rest_current': 1.2}})
    cpu = Network(graph, [1], ['feedforward'], config=config, cell_types=['L1', 'Mi1'])
    cuda = Network(graph, [1], ['feedforward'], config=config, cell_types=['L1', 'Mi1'], device='cuda')
    for tick in range(200):
        current = torch.tensor([[-3. if tick in (50, 100) else 0., 0.]])
        a, b = cpu.step(current), cuda.step(current.cuda())
        assert torch.equal(a.spikes, b.spikes.cpu())
        for name in ('voltage', 'sensory_state', 'adaptation'):
            torch.testing.assert_close(getattr(cpu, name), getattr(cuda, name).cpu(), atol=1e-5, rtol=1e-5)


def test_subthreshold_motion_operating_point_requires_existing_afferent_spikes():
    graph = Graph.from_contacts([10, 20, 30, 40], [10, 10], [20, 30], [1, 1], [1]*4, .4)
    cfg = NeuronConfig(class_parameters={t: {'rest_current': .95} for t in ('T4a', 'T5a')})
    net = Network(graph, [1, 1], ['feedforward']*2, config=cfg, cell_types=['Mi1','T4a','T5a','T4a'])
    for _ in range(500):
        assert not net.step(torch.zeros(1, 4)).spikes.any()
    spikes = torch.zeros(1, 4, dtype=torch.int64)
    for _ in range(100):
        spikes += net.step(torch.tensor([[30., 0., 0., 0.]])).spikes
    assert (spikes[0, 1:3] > 0).all()
    assert spikes[0, 3] == 0  # identical intrinsic current, but no measured input edge
    torch.testing.assert_close(net.magnitudes, torch.tensor([.4, .4]))
