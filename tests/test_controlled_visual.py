"""Controlled experiment boundaries: measured crop, visible input, causal scoring."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from controlled_visual import crop_payload, trajectory, score_forecasts, run_sequence, make_network, passed


def test_crop_preserves_induced_edges_delays_signs_and_weights():
    from test_training import trainer
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'source.pt'
        trainer().save(path)
        payload = torch.load(path, weights_only=True)
    before = copy.deepcopy(payload['metadata'])
    crop, nodes, edges = crop_payload(payload, [0, 1, 3])
    assert nodes.tolist() == [0, 1, 3]
    assert edges.tolist() == [0, 2]
    assert crop['graph'].body_ids.tolist() == [10, 20, 40]
    assert crop['graph'].pre.tolist() == [0, 1]
    assert crop['graph'].post.tolist() == [1, 2]
    assert crop['delays'] == [1, 1]
    assert crop['pathways'] == ['feedforward', 'behavioral']
    torch.testing.assert_close(crop['weights'], payload['state']['network']['magnitudes'][[0, 2]])
    assert payload['metadata'] == before


def test_dot_produces_on_then_off_without_silent_camera_reset():
    from fly_connectome.sensor import EventCamera
    frames = trajectory(2, 5, [1, 2, 3], 0, blank=2)
    camera = EventCamera(1, 2, 5)
    events = [camera.observe(frame) for frame in frames]
    assert [e.pixels.tolist() for e in events] == [[], [], [1], [1, 2], [2, 3], [3]]
    assert events[3].on.tolist() == [False, True]
    assert not frames[-1].any()


def test_forecast_scoring_requires_positive_lead_and_excludes_unobserved_tail():
    # Perfect reactive predictions must not score as anticipatory predictions.
    y = np.array([0., 0., -1., 0., 0.])
    reactive = y.copy()
    scores = score_forecasts(y, reactive, np.array([0, 2, 4]), lead=2)
    assert scores['samples'] == 2
    assert scores['event_mse'] == 1.
    assert scores['quiet_mse'] == 1.
    assert scores['mse'] == 1.
    anticipatory = np.array([-1., 0., 0., 0., 0.])
    assert score_forecasts(y, anticipatory, np.array([0, 2, 4]), lead=2)['mse'] == 0.


def test_visual_runner_is_causal_and_frozen_weights_stay_fixed(tmp_path):
    from test_training import trainer
    model = trainer()
    path = tmp_path / 'source.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    crop, _, _ = crop_payload(payload, [0, 1, 2, 3])
    frames = trajectory(32, 64, [1, 2, 3], 0, blank=2)
    changed = frames.clone()
    changed[3:] = False
    a, b = (make_network(crop, payload['metadata']) for _ in range(2))
    weights = a.magnitudes.clone()
    first = run_sequence(a, crop, payload['metadata'], frames, [0], learning=False)
    second = run_sequence(b, crop, payload['metadata'], changed, [0], learning=False)
    np.testing.assert_array_equal(first['prediction'][:6], second['prediction'][:6])
    assert first['prediction'].shape == (12, 1)
    assert first['spikes'].shape == (12, 4)
    assert 'eligibility' not in first
    assert first['stability']['finite']
    torch.testing.assert_close(a.magnitudes, weights, atol=0, rtol=0)


def test_training_updates_real_network_and_records_local_proposals(tmp_path):
    from test_training import trainer
    model = trainer()
    path = tmp_path / 'source.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    payload['metadata']['pathways'] = ['predictive'] * 3
    crop, _, _ = crop_payload(payload, [0, 1, 2, 3])
    net = make_network(crop, payload['metadata'])
    weights = net.magnitudes.clone()
    frames = trajectory(32, 64, list(range(10)), 0, blank=2)
    result = run_sequence(net, crop, payload['metadata'], frames, [1], learning=True)
    assert result['spikes'].any()
    assert result['arrival_counts'].sum() > 0
    assert not torch.equal(net.magnitudes, weights)
    assert np.abs(result['local_update_sums']).sum() > 0


def test_silencing_does_not_count_as_prediction_learning():
    zero = score_forecasts(np.array([0., -1., 1., 0.]), np.zeros(4), np.arange(3), 1)
    frozen = dict(zero, mse=zero['mse'] * 2)
    assert not passed(zero, frozen, zero, zero)


def test_raw_nonfinite_state_cannot_be_hidden_by_prediction_clipping(tmp_path):
    from test_training import trainer
    model = trainer()
    path = tmp_path / 'source.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    crop, _, _ = crop_payload(payload, [0, 1, 2, 3])
    net = make_network(crop, payload['metadata'])
    net.predictive_current[0, 0] = float('inf')
    frames = trajectory(32, 64, [1], 0, blank=1)
    with pytest.raises(ValueError, match='nonfinite'):
        run_sequence(net, crop, payload['metadata'], frames, [0], learning=False)
