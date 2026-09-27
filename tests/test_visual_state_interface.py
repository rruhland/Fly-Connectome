import copy

import pytest
import torch

from fly_connectome.vision.interface import VisualStateEncoder


def state(sample=0):
    return dict(sample=sample, entities=[
        dict(id=4, position=torch.tensor([4., 8.]), observed=True, observation_age=0, association_strength=1.),
        dict(id=7, position=torch.tensor([8., 12.]), observed=True, observation_age=0, association_strength=1.)],
        forecasts=[dict(id=4, horizon_samples=4, origin_sample=sample, evidence_age=0,
            position=torch.tensor([6., 9.]), mixture_centers=torch.tensor([[4., 12.], [8., 6.]]),
            mixture_weights=torch.tensor([.25, .75]), component_sigma_pixels=1.,
            marginal_interval_90=torch.tensor([[2., 3.], [10., 15.]]))])


def test_transport_is_invariant_to_order_and_identity_names():
    a, b = VisualStateEncoder(16, 32, .02), VisualStateEncoder(16, 32, .02)
    for t in range(2):
        original = state(t)
        for e in original['entities']:
            e['position'] += t
        renamed = copy.deepcopy(original)
        renamed['entities'].reverse()
        for e in renamed['entities']:
            e['id'] += 100
        for f in renamed['forecasts']:
            f['id'] += 100
        left, right = a.encode(original, torch.zeros(16, 32)), b.encode(renamed, torch.zeros(16, 32))
        assert torch.equal(left['indices'], right['indices'])
        assert torch.equal(left['values'], right['values'])


def test_full_multimodal_forecast_and_aspect_ratio_are_preserved():
    encoder = VisualStateEncoder(16, 32, .02)
    original = state()
    out = encoder.encode(original, torch.zeros(16, 32))
    forecast = out['forecasts'][0]
    assert torch.equal(forecast['mixture_centers'], original['forecasts'][0]['mixture_centers'])
    assert torch.equal(forecast['mixture_weights'], original['forecasts'][0]['mixture_weights'])
    restored = forecast['normalized_centers']*32+torch.tensor([7.5, 15.5])
    assert torch.equal(restored, forecast['mixture_centers'])
    assert forecast['horizon_seconds'] == .08


def test_missing_evidence_does_not_invent_observed_velocity():
    encoder = VisualStateEncoder(16, 32, .02)
    encoder.encode(state(0), torch.zeros(16, 32))
    observed = state(1)
    observed['entities'][0]['position'] += torch.tensor([0., 1.])
    out = encoder.encode(observed, torch.zeros(16, 32))
    velocity = next(e['observed_velocity'] for e in out['entities'] if e['id'] == 4)
    missing = state(2)
    missing['entities'][0].update(observed=False, observation_age=1, position=torch.tensor([10., 25.]))
    out = encoder.encode(missing, None)
    entity = next(e for e in out['entities'] if e['id'] == 4)
    assert torch.equal(entity['observed_velocity'], velocity)
    assert entity['velocity_age_seconds'] == .02
    assert out['context_age_seconds'] == .02
    with pytest.raises(ValueError, match='reset_state'):
        encoder.encode(state(0), None)
    encoder.reset_state()
    out = encoder.encode(state(0), None)
    assert not any(e['velocity_known'] for e in out['entities'])


def test_large_static_context_survives_without_detected_entities():
    encoder = VisualStateEncoder(16, 32, .02)
    image = torch.zeros(16, 32)
    image[3:10, 4:20] = .8
    out = encoder.encode(dict(sample=0, entities=[], forecasts=[]), image)
    assert out['coarse_context'].abs().sum() > 0
    assert torch.isfinite(out['values']).all()


def test_off_image_forecasts_are_reported_even_inside_aspect_padding():
    encoder = VisualStateEncoder(16, 32, .02)
    original = state()
    original['forecasts'][0]['mixture_centers'] = torch.tensor([[20., 15.], [4., 12.]])
    original['entities'][0]['position'] = torch.tensor([20., 15.])
    out = encoder.encode(original, None)
    dense = torch.zeros(out['dimension'])
    dense[out['indices']] = out['values']
    assert dense[slice(*out['layout']['future_outside'])].sum() == .25
    assert dense[slice(*out['layout']['future'])].sum() == .75
    assert dense[slice(*out['layout']['metadata'])][-2] == 1


def test_transport_preserves_all_mixture_mass_and_does_not_alias_source():
    encoder = VisualStateEncoder(16, 32, .02)
    original = state()
    original['forecasts'].append(copy.deepcopy(original['forecasts'][0]))
    original['forecasts'][1]['mixture_centers'] += 40
    out = encoder.encode(original, None)
    dense = torch.zeros(out['dimension'])
    dense[out['indices']] = out['values']
    assert dense[slice(*out['layout']['future'])].sum()+dense[slice(*out['layout']['future_outside'])].sum() == 2
    out['forecasts'][0]['mixture_centers'].zero_()
    assert original['forecasts'][0]['mixture_centers'].abs().sum() > 0
