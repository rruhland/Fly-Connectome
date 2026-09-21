import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from frame_prediction import FramePrediction
from fly_connectome.dynamics import Activity, Network
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig, Plasticity


def setup(cls=FramePrediction):
    graph = Graph.from_contacts([10, 20, 30], [10, 10], [20, 30], [1, 1], [1]*3, .5)
    net = Network(graph, [1, 1], ['predictive', 'behavioral'])
    cfg = LearningConfig(prediction_encoding='signed-current-v1',
                         visual_target='input-arrivals-v1',
                         visual_eligibility='forecast-causal-v1',
                         eta_prediction=.1, eta_reward=.1, homeostasis_rate=0.)
    return net, cls(net, cfg)


def activity(observed=0., predicted=0., arrived=False):
    obs = torch.tensor([[0., observed, 0.]])
    pred = torch.tensor([[0., predicted, 0.]])
    return Activity(torch.ones(1, 3, dtype=torch.bool), obs, pred,
                    torch.tensor([0, 0] if arrived else [], dtype=torch.long),
                    torch.tensor([0, 1] if arrived else [], dtype=torch.long),
                    feedforward_arrivals=obs, sensory_input=torch.zeros_like(obs))


@pytest.mark.parametrize('confirmation,expected', [(1., .08), (0., -.02), (-1., -.12)])
def test_only_eighth_tick_confirms_issue_time_forecast_and_eligibility(confirmation, expected):
    net, rule = setup()
    rule.observe(activity(predicted=.2, arrived=True), torch.zeros(1))  # issue t=0
    assert rule.proposals[0] == 0
    for _ in range(7):
        # Strong contradictory intervening observations/arrivals must not get credit.
        rule.observe(activity(observed=-1., predicted=.9, arrived=True), torch.zeros(1))
        assert rule.proposals[0] == 0
    rule.observe(activity(observed=confirmation, predicted=.9, arrived=True), torch.zeros(1))
    assert rule.proposals[0].item() == pytest.approx(expected)
    rule.synchronize()
    assert net.magnitudes[0].item() == pytest.approx(.5+expected)


def test_behavioral_learning_and_live_traces_match_original_each_tick():
    _, frame = setup()
    _, original = setup(Plasticity)
    for tick in range(25):
        sample = activity(observed=.7, predicted=.3, arrived=tick % 3 == 0)
        for rule in (frame, original):
            rule.observe(sample, torch.tensor([.4]))
        torch.testing.assert_close(frame.keys, original.keys, rtol=0, atol=0)
        torch.testing.assert_close(frame.values, original.values, rtol=0, atol=0)
        torch.testing.assert_close(frame.proposals[1], original.proposals[1], rtol=0, atol=0)
        torch.testing.assert_close(frame.rates, original.rates, rtol=0, atol=0)


def test_runner_frame_update_diagnostics_reconstruct_weight_changes(tmp_path):
    import numpy as np
    from test_training import trainer
    from controlled_visual import crop_payload, make_network, run_sequence, trajectory
    model = trainer()
    path = tmp_path / 'source.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    metadata = payload['metadata']
    metadata['config']['neural_steps'] = 8
    metadata['learning'].update(visual_target='input-arrivals-v1',
                                visual_eligibility='forecast-causal-v1',
                                prediction_encoding='signed-current-v1',
                                homeostasis_rate=0., eta_prediction=1e-5)
    metadata['pathways'] = ['predictive'] * 3
    crop, _, _ = crop_payload(payload, [0, 1, 2, 3])
    net = make_network(crop, metadata)
    before = net.magnitudes.clone()
    frames = trajectory(32, 64, list(range(10)), 0, blank=2)
    result = run_sequence(net, crop, metadata, frames, [1], learning=True,
                          visual_schedule='frame-horizon-v1')
    changes = (net.magnitudes-before).numpy()[result['incoming_edges']]
    assert np.abs(result['local_update_sums']).sum() > 0
    np.testing.assert_allclose(changes, result['local_update_sums'].sum(axis=0), atol=1e-7)
