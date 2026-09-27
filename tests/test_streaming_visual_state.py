import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from streaming_visual_state import (StreamingVisualState, EndpointCalibration,
                                    ContrastNormalizedVisualState, NoiseCalibratedVisualState,
                                    ConsensusVisualState)
from local_metric_association import LocalMetricAssociation
from contextual_motion_dynamics import ContextualMotionDynamics


def sample(x):
    visible = torch.zeros(16, 32)
    visible[8, x] = 1.
    event = torch.zeros(2, 16, 32)
    event[1, 8, x] = 1.
    return event, visible


def test_streaming_missing_sample_is_a_hypothesis_and_cannot_train_dynamics():
    model = StreamingVisualState(height=16, width=32)
    for x in range(3, 11):
        result = model.step(*sample(x), learn=True)
    identity = result['entities'][0]['id']
    weights = {h: m.values.clone() for h, m in model.dynamics.items()}
    result = model.step(torch.zeros(2, 16, 32), None, learn=True)
    entity = next(e for e in result['entities'] if e['id'] == identity)
    assert entity['observed'] is False
    assert entity['observation_age'] == 1
    assert result['forecasts']
    assert all(f['evidence_age'] > 0 for f in result['forecasts'])
    assert all(torch.equal(weights[h], m.values) for h, m in model.dynamics.items())


def test_calibration_censors_missing_camera_and_learns_balanced_observable_outcomes():
    model = EndpointCalibration()
    for _ in range(20):
        model.observe(True, 2.)
        model.observe(False, None)
    result = model.summary()
    assert result['reidentification_probability'] == .5
    assert result['conditional_radius_90'] == 2.
    assert result['samples'] == 40


def test_expired_tracks_are_removed_without_reusing_public_identities():
    model = StreamingVisualState(height=16, width=32)
    identities = []
    for _ in range(3):
        for x in (3, 4, 5):
            result = model.step(*sample(x))
        identities.append(result['entities'][0]['id'])
        for _ in range(66):
            model.step(torch.zeros(2, 16, 32), None)
        assert not model.state.tracker.slots
        assert not model.histories
        assert not model.pending
    assert len(set(identities)) == 3


def test_checkpoint_preserves_learning_and_frozen_streaming_predictions(tmp_path):
    model = StreamingVisualState(height=16, width=32)
    for x in range(3, 20):
        model.step(*sample(x), learn=True)
    path = tmp_path/'candidate.pt'
    model.save(path)
    loaded = StreamingVisualState.load(path)
    model.reset_state()
    for x in range(3, 12):
        expected, actual = model.step(*sample(x)), loaded.step(*sample(x))
        assert torch.equal(expected['support_field'], actual['support_field'])
        for a, b in zip(expected['forecasts'], actual['forecasts']):
            assert torch.equal(a['position'], b['position'])
            assert a['reidentification_probability'] == b['reidentification_probability']
    assert all(torch.equal(model.dynamics[h].values, loaded.dynamics[h].values)
               for h in model.dynamics)


def test_contrast_gain_control_is_affine_invariant_and_constant_image_is_finite():
    model = ContrastNormalizedVisualState(height=16, width=32)
    _, visible = sample(5)
    assert torch.allclose(model.frame_contrast(visible), model.frame_contrast(.3+.2*visible))
    assert torch.equal(model.frame_contrast(torch.ones_like(visible)), torch.zeros_like(visible))


def test_cross_sensor_consensus_does_not_turn_blank_sensor_noise_into_entities():
    model = ConsensusVisualState(height=64, width=64)
    generator = torch.Generator().manual_seed(98001)
    for _ in range(12):
        visible = .4+.02*torch.randn(64, 64, generator=generator)
        result = model.step(torch.zeros(2, 64, 64), visible)
        assert not result['entities']


def test_consensus_uses_raw_event_evidence_even_when_learned_filter_is_uncertain():
    model = ConsensusVisualState(height=16, width=32)
    model.observer.bias.fill_(-10.)
    for x in (3, 6, 9, 12):
        result = model.step(*sample(x))
    assert len(result['entities']) == 1
    assert result['entities'][0]['observed']
    assert result['entities'][0]['position'][1] == 12


def test_checkpoint_keeps_injected_learning_rules_and_horizon_set(tmp_path):
    model = StreamingVisualState(memory=LocalMetricAssociation(),
                                 dynamics={1: ContextualMotionDynamics()})
    path = tmp_path/'components.pt'
    model.save(path)
    loaded = StreamingVisualState.load(path)
    assert type(loaded.state.memory) is LocalMetricAssociation
    assert tuple(loaded.dynamics) == (1,)
    assert type(loaded.dynamics[1]) is ContextualMotionDynamics


def test_confirmed_visual_continuity_survives_event_sensor_dropout():
    model = ConsensusVisualState(height=16, width=32)
    for x in (3, 4, 5):
        result = model.step(*sample(x))
    identity = result['entities'][0]['id']
    for x in (6, 7, 8):
        event, image = sample(x)
        result = model.step(torch.zeros_like(event), image)
        assert any(e['id'] == identity and e['observed'] for e in result['entities'])
