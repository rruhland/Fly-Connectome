import torch
import pytest

from test_training import trainer
from fly_connectome.evaluation import evaluate, compare
from fly_connectome.diagnostics import Probe, run_probe


def test_evaluation_uses_held_out_seeds_and_never_trains(tmp_path):
    path = tmp_path / 'model.pt'
    trainer().save(path)
    before = path.read_bytes()
    a = evaluate(path, [100], 10)
    b = evaluate(path, [100], 10)
    assert a == b
    assert a['seeds'] == [100] and a['hit_shaping'] == 0
    with pytest.raises(ValueError, match='overlap'):
        evaluate(path, [1], 2)
    assert path.read_bytes() == before
    result = compare(path, path, [100], 5)
    assert set(result) == {'learned', 'frozen', 'random', 'scripted'}


def test_probes_are_deterministic_silenced_and_separate_from_checkpoint(tmp_path):
    path = tmp_path / 'model.pt'
    trainer().save(path)
    before = path.read_bytes()
    probe = Probe(kind='flash', steps=5, start=1, duration=2, x=32, y=16, radius=10)
    a = run_probe(path, probe, seed=100)
    b = run_probe(path, probe, seed=100)
    assert torch.equal(a['spikes'], b['spikes'])
    silent = run_probe(path, probe, seed=100, silenced_types=['L2'])
    assert not silent['spikes'][:, 0].any()
    assert a['checkpoint_sha256'] == silent['checkpoint_sha256']
    assert path.read_bytes() == before


def test_off_target_and_moving_edges_are_binary():
    probe = Probe(kind='moving_edge', steps=4, polarity='off', speed=2.)
    frames = [probe.frame(i, 8, 16) for i in range(4)]
    assert frames[0].dtype == torch.bool
    assert not torch.equal(frames[0], frames[3])
