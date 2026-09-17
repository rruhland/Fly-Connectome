import torch

from fly_connectome.graph import Graph
from fly_connectome.sensor import Retina
from fly_connectome.training import Trainer, RunConfig, load_checkpoint, warm_start


def trainer(batch=1):
    g = Graph.from_contacts([10, 20, 30, 40], [10, 20, 20], [20, 30, 40], [5, 5, 5], [1]*4, .1)
    retina = Retina(32, 64, [[0, 0]], [0, -1, -1, -1], ['L2', 'visual', 'DNa02', 'DNa02'], {'L2': 'off'})
    return Trainer(g, [1]*3, ['feedforward', 'behavioral', 'behavioral'], retina,
                   [30], [40], list(range(1, batch + 1)),
                   config=RunConfig(neural_steps=2, sync_steps=2),
                   manifest={'dataset': 'test-fixture', 'graph_sha256': g.identity()})


def test_exact_checkpoint_resume_and_frozen_evaluation(tmp_path):
    a = trainer()
    a.run(5)
    path = tmp_path / 'checkpoint.pt'
    a.save(path)
    before = path.read_bytes()
    b = load_checkpoint(path)
    a.run(8); b.run(8)
    for key in ('voltage', 'magnitudes', 'history', 'adaptation'):
        torch.testing.assert_close(getattr(a.network, key), getattr(b.network, key), rtol=0, atol=0)
    torch.testing.assert_close(a.environment.ball, b.environment.ball, rtol=0, atol=0)
    frozen = load_checkpoint(path, evaluation=True)
    weights = frozen.network.magnitudes.clone()
    frozen.run(5)
    torch.testing.assert_close(weights, frozen.network.magnitudes, rtol=0, atol=0)
    assert path.read_bytes() == before


def test_environment_reset_does_not_reset_neural_state():
    a = trainer()
    a.network.voltage.fill_(.3)
    a.reset_environment()
    assert torch.all(a.network.voltage == .3)


def test_warm_start_preserves_only_matching_measured_edges():
    a = trainer()
    a.network.magnitudes[:] = torch.tensor([.2, .3, .4])
    g = Graph.from_contacts([10, 20, 30, 40, 50], [10, 20, 20, 40], [20, 30, 40, 50], [5]*4, [1]*5, .1)
    from fly_connectome.dynamics import Network
    b = Network(g, [1]*4, ['feedforward']*4)
    warm_start(a.network, b)
    torch.testing.assert_close(b.magnitudes, torch.tensor([.2, .3, .4, .5]))


def test_detailed_activity_is_evaluation_only(tmp_path):
    a = trainer()
    a.step()
    assert 'events' not in a.snapshot() and 'spiking_neurons' not in a.snapshot()
    path = tmp_path/'model.pt'
    a.save(path)
    frozen = load_checkpoint(path, evaluation=True, seeds=[100])
    frozen.step()
    snapshot = frozen.snapshot()
    assert snapshot['events']['width'] == 64
    assert 'spiking_neurons' in snapshot
    assert 'population_rates_hz' in snapshot
