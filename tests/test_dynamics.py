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
