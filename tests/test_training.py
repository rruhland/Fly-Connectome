import torch
import pytest

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


def test_checkpoint_replace_retries_windows_sharing_conflicts(tmp_path, monkeypatch):
    import fly_connectome.training as training
    model = trainer()
    path = tmp_path/'shared.pt'
    model.save(path)
    original_replace = training.os.replace
    attempts = []
    def temporarily_locked(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError('file is open by another reader')
            error.winerror = 5
            raise error
        original_replace(source,destination)
    monkeypatch.setattr(training.os,'replace',temporarily_locked)
    monkeypatch.setattr('time.sleep',lambda _: None)
    model.run(2)
    model.save(path)
    assert len(attempts) == 3
    assert load_checkpoint(path).step_index == 2
    assert not list(tmp_path.glob('*.tmp'))


def test_checkpoint_replace_does_not_retry_unrelated_permission_error(tmp_path, monkeypatch):
    import fly_connectome.training as training
    attempts = []
    def denied(*_):
        attempts.append(1)
        raise PermissionError('directory is not writable')
    monkeypatch.setattr(training.os,'replace',denied)
    with pytest.raises(PermissionError):
        trainer().save(tmp_path/'denied.pt')
    assert len(attempts) == 1
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_windows_lock_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import fly_connectome.training as training
    model = trainer()
    path = tmp_path/'locked.pt'
    model.save(path)
    previous = path.read_bytes()
    attempts = []
    def locked(*_):
        attempts.append(1)
        error = PermissionError('persistent sharing conflict')
        error.winerror = 32
        raise error
    monkeypatch.setattr(training.os,'replace',locked)
    monkeypatch.setattr('time.sleep',lambda _: None)
    model.run(2)
    with pytest.raises(PermissionError):
        model.save(path)
    assert 1 < len(attempts) < 100
    assert path.read_bytes() == previous
    assert load_checkpoint(path).step_index == 0
    assert not list(tmp_path.glob('*.tmp'))


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
    payload['metadata']['learning'].pop('prediction_encoding')
    payload['metadata']['learning'].pop('visual_target')
    payload['metadata']['learning'].pop('visual_eligibility')
    for key in ('rest_current', 'membrane_decay', 'current_decay', 'sensory_state'):
        payload['state']['network'].pop(key)
    payload['state']['plasticity'].pop('sensory_mask')
    for key in ('learning_statistics', 'previous_learning_observed'):
        payload['state']['trainer'].pop(key)
    torch.save(payload, path)
    b = load_checkpoint(path)
    assert not b.network.rest_current.any() and b.network.config.tau_sensory == 0
    torch.testing.assert_close(b.plasticity.sensory_mask, b.retina.injected)
    assert not b.learning_statistics.any() and not b.previous_learning_observed.any()
    a.run(5); b.run(5)
    for key in ('voltage', 'magnitudes', 'history'):
        torch.testing.assert_close(getattr(a.network, key), getattr(b.network, key), rtol=0, atol=0)


def test_signed_prediction_metrics_and_pending_state_resume(tmp_path):
    from fly_connectome.dynamics import NeuronConfig
    from fly_connectome.plasticity import LearningConfig
    base = trainer()
    a = Trainer(base.network.graph, [1]*3, ['predictive', 'behavioral', 'behavioral'],
        Retina(**dict(base.retina.spec, injection={'L2': 'contrast'})), [30], [40], [1],
        config=base.config, neurons=NeuronConfig(dt=base.network.config.dt, tau_sensory=.02),
        learning=LearningConfig(prediction_encoding='signed-current-v1'))
    a.step()
    assert a.previous_observed[0, 0] < 0
    torch.testing.assert_close(a.previous_predicted, a.plasticity.expected)
    path = tmp_path / 'signed.pt'
    a.save(path)
    b = load_checkpoint(path)
    assert b.learning_config.prediction_encoding == 'signed-current-v1'
    a.run(5); b.run(5)
    for obj_a, obj_b, keys in ((a, b, ('statistics', 'previous_observed', 'previous_predicted')),
                             (a.plasticity, b.plasticity, ('expected', 'values', 'proposals')),
                             (a.network, b.network, ('voltage', 'magnitudes'))):
        for key in keys:
            torch.testing.assert_close(getattr(obj_a, key), getattr(obj_b, key), rtol=0, atol=0)


def test_lightweight_metrics_preserve_every_model_update_and_event_score():
    import copy
    from dataclasses import replace
    full=trainer()
    light=copy.deepcopy(full)
    light.config=replace(light.config,metrics_mode='events')
    for _ in range(20):
        a,_,_=full.step();b,_,_=light.step()
        torch.testing.assert_close(a.spikes,b.spikes,rtol=0,atol=0)
    for obj in ('network','plasticity'):
        for name,value in vars(getattr(full,obj)).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,getattr(getattr(light,obj),name),rtol=0,atol=0)
    for name in ('previous_observed','previous_predicted','previous_learning_observed','motor_rates'):
        torch.testing.assert_close(getattr(full,name),getattr(light,name),rtol=0,atol=0)
    torch.testing.assert_close(full.statistics[:4],light.statistics[:4],rtol=0,atol=0)
    torch.testing.assert_close(full.statistics[7:],light.statistics[7:],rtol=0,atol=0)
    assert light.statistics[6]==0 and full.statistics[6]>0
    assert light.learning_statistics[2]==0
    assert light.snapshot()['metrics_mode']=='events'
    light.config=replace(light.config,metrics_mode='full')
    full.step();light.step()
    torch.testing.assert_close(full.network.magnitudes,light.network.magnitudes,rtol=0,atol=0)
