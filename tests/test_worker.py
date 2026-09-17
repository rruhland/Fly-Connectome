import pytest
import torch

from test_training import trainer
from fly_connectome.worker import apply_command


def test_pause_reset_save_and_confirmed_restart(tmp_path):
    model = trainer()
    state = {'paused': False, 'stop': False, 'single_step': False}
    apply_command(model, state, {'action': 'pause'}, tmp_path)
    assert state['paused']
    apply_command(model, state, {'action': 'resume'}, tmp_path)
    assert not state['paused']
    apply_command(model, state, {'action': 'save'}, tmp_path)
    assert (tmp_path / 'checkpoint-latest.pt').exists()
    model.network.magnitudes.fill_(.1)
    with pytest.raises(ValueError, match='confirmation'):
        apply_command(model, state, {'action': 'restart'}, tmp_path)
    apply_command(model, state, {'action': 'restart', 'confirmed': True}, tmp_path)
    assert torch.all(model.network.magnitudes == .5)


def test_evaluation_cannot_restart_or_save(tmp_path):
    from fly_connectome.training import load_checkpoint
    path = tmp_path / 'source.pt'
    trainer().save(path)
    model = load_checkpoint(path, evaluation=True)
    for action in ('save', 'restart'):
        with pytest.raises(ValueError, match='evaluation'):
            apply_command(model, {}, {'action': action, 'confirmed': True}, tmp_path)


def test_worker_publishes_stopped_state_and_saves_on_completion(tmp_path):
    from fly_connectome.worker import run_worker
    from fly_connectome.telemetry import Telemetry, latest_snapshot
    path = tmp_path / 'source.pt'
    trainer().save(path)
    run = tmp_path / 'run'
    telemetry = Telemetry(run)
    telemetry.subscribe()
    run_worker(path, run, steps=2)
    assert latest_snapshot(run)['stopped'] is True
    assert (run / 'checkpoint-latest.pt').exists()
    telemetry.close()
