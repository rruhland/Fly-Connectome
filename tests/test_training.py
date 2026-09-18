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


def test_class_dynamics_and_sensory_state_resume_exactly(tmp_path):
    from fly_connectome.dynamics import NeuronConfig
    base = trainer()
    a = Trainer(base.network.graph, [1]*3, ['feedforward', 'behavioral', 'behavioral'],
        Retina(**dict(base.retina.spec, injection={'L2': 'contrast'})), [30], [40], [1],
        config=base.config, neurons=NeuronConfig(dt=base.network.config.dt, tau_sensory=.02,
            class_parameters={'L2': {'rest_current': 1.4, 'tau_membrane': .01}}))
    a.run(3)
    path = tmp_path / 'new.pt'
    a.save(path)
    assert torch.load(path, weights_only=True)['schema_version'] == 2
    b = load_checkpoint(path)
    a.run(7); b.run(7)
    for key in ('voltage', 'sensory_state', 'magnitudes', 'history'):
        torch.testing.assert_close(getattr(a.network, key), getattr(b.network, key), rtol=0, atol=0)


def test_frozen_warmup_does_not_run_game_or_learning(tmp_path):
    from dataclasses import replace
    a = trainer()
    a.config = replace(a.config, warmup_steps=10)
    path = tmp_path / 'warm.pt'
    a.save(path)
    b = load_checkpoint(path, evaluation=True, seeds=[100])
    assert b.network.step_index == 10 and b.step_index == 0
    assert b.metrics['spikes'] == 0 and b.plasticity is None
    c = load_checkpoint(path)
    assert c.network.step_index == 0


def test_legacy_checkpoint_keeps_zero_background_and_impulse_semantics(tmp_path):
    a = trainer()
    a.run(3)
    path = tmp_path / 'legacy.pt'
    a.save(path)
    payload = torch.load(path, weights_only=True)
    payload['schema_version'] = 1
    for key in ('class_parameters', 'tau_sensory'):
        payload['metadata']['neurons'].pop(key)
    payload['metadata']['config'].pop('warmup_steps')
    for key in ('rest_current', 'membrane_decay', 'current_decay', 'sensory_state'):
        payload['state']['network'].pop(key)
    torch.save(payload, path)
    b = load_checkpoint(path)
    assert not b.network.rest_current.any() and b.network.config.tau_sensory == 0
    a.run(5); b.run(5)
    for key in ('voltage', 'magnitudes', 'history'):
        torch.testing.assert_close(getattr(a.network, key), getattr(b.network, key), rtol=0, atol=0)
