"""Exact reference checks for lower-cost promoted visual-state execution."""

import torch

from fly_connectome.vision import VisualStateEncoder, load_default
from fly_connectome.vision.dynamics import ContextBelief, SpatialBelief, mixture_quantile
from fly_connectome.vision.memory import ConsensusLocalMetricAssociation
from fly_connectome.vision.state import StreamingVisualState


def assert_identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_identical(left, right)
    else:
        assert actual == expected


def scalar_reference_step(model, events, image, *, learn=True):
    state = StreamingVisualState.step(model, events, image, learn=learn)
    for forecast in state['forecasts']:
        if forecast['evidence_age']:
            continue
        history = torch.stack(model.observed_history(forecast['id']))
        belief = model.dynamics[forecast['horizon_samples']]
        centers, weights = belief.distribution(history[1:]-history[:-1])
        low, high = belief.calibration.bounds()
        forecast.update(mixture_centers=centers+history[-1], mixture_weights=weights,
            component_sigma_pixels=1., point_semantics='mixture_mean',
            marginal_interval_90=torch.stack((mixture_quantile(centers, weights, low),
                                               mixture_quantile(centers, weights, high)))+history[-1],
            interval_calibration_samples=len(belief.calibration.ranks))
    return state


def test_batched_intervals_match_scalar_search_for_varied_mixtures():
    from fly_connectome.vision.dynamics import batched_mixture_quantiles

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(92731)
    for batch, components in ((1, 1), (3, 5), (24, 32)):
        centers = torch.randn(batch, components, 2, generator=generator)
        weights = torch.softmax(torch.randn(batch, components, generator=generator), 1)
        probabilities = torch.rand(batch, 2, 2, generator=generator)
        probabilities[0, 0] = 0
        probabilities[0, 1] = 1
        expected = torch.stack([
            torch.stack([mixture_quantile(c, w, p) for p in bounds])
            for c, w, bounds in zip(centers, weights, probabilities)
        ])
        actual = batched_mixture_quantiles(centers, weights, probabilities)
        assert torch.equal(actual, expected)


def test_online_multientity_outputs_and_learning_match_scalar_reference(tmp_path):
    torch.set_num_threads(1)
    actual, reference = load_default(), load_default()
    actual_encoder, reference_encoder = (VisualStateEncoder(64, 64, .02),
                                         VisualStateEncoder(64, 64, .02))
    previous = torch.zeros(64, 64)
    for sample in range(38):
        image = torch.zeros(64, 64)
        for index in range(8):
            phase = (sample+index*3) % 80
            x = 5+(phase if phase < 40 else 79-phase)
            y = 4+index*7
            image[y:y+3, x:x+3] = 1
        events = torch.stack(((previous-image).clamp(min=0),
                              (image-previous).clamp(min=0)))
        previous = image
        available_image = None if sample in (24, 25, 26) else image
        left = actual.step(events, available_image, learn=True)
        right = scalar_reference_step(reference, events, available_image)
        assert_identical(left, right)
        assert_identical(actual_encoder.encode(left, available_image),
                         reference_encoder.encode(right, available_image))
    actual.save(tmp_path/'actual.pt')
    reference.save(tmp_path/'reference.pt')
    assert_identical(torch.load(tmp_path/'actual.pt', weights_only=True),
                     torch.load(tmp_path/'reference.pt', weights_only=True))


def test_each_new_forecast_uses_one_mixture_lookup():
    torch.set_num_threads(1)
    model = load_default()
    calls = {h: 0 for h in model.dynamics}
    for horizon, belief in model.dynamics.items():
        original = belief.distribution
        def counted(history, *, horizon=horizon, original=original):
            calls[horizon] += 1
            return original(history)
        belief.distribution = counted
    previous = torch.zeros(64, 64)
    issued = 0
    for sample in range(14):
        image = torch.zeros(64, 64)
        image[12:15, 8+sample:11+sample] = 1
        events = torch.stack(((previous-image).clamp(min=0),
                              (image-previous).clamp(min=0)))
        previous = image
        result = model.step(events, image, learn=True)
        issued += sum(f['evidence_age'] == 0 for f in result['forecasts'])
    assert issued > 0
    assert sum(calls.values()) == issued


def test_context_variance_is_reused_until_keys_change(monkeypatch):
    memory = ConsensusLocalMetricAssociation(dimensions=4)
    for index in range(48):
        memory.observe(torch.tensor([float(index % 7), float(index // 7),
                                     float(index % 3), float(index % 5)]),
                       torch.tensor([float(index % 3), float(index % 5)]))
    original_var = torch.Tensor.var
    calls = 0
    def counted_var(tensor, *args, **kwargs):
        nonlocal calls
        if tensor is memory.keys:
            calls += 1
        return original_var(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'var', counted_var)
    key = torch.tensor([1., 2., 0., 1.])
    for _ in range(6):
        memory.predict(key)
    assert calls == 1
    memory.observe(torch.tensor([3., 4., 0., 2.]), torch.tensor([1., 3.]))
    memory.predict(key)
    assert calls == 2
    memory.keys[0, 0].add_(.5)
    memory.predict(key)
    assert calls == 3


def test_transport_clones_standard_tensors_without_generic_deepcopy(monkeypatch):
    encoder = VisualStateEncoder(16, 32, .02)
    forecast = dict(id=4, horizon_samples=4, origin_sample=0, evidence_age=0,
                    position=torch.tensor([6., 9.]),
                    mixture_centers=torch.tensor([[4., 12.], [8., 6.]]),
                    mixture_weights=torch.tensor([.25, .75]),
                    marginal_interval_90=torch.tensor([[2., 3.], [10., 15.]]),
                    component_sigma_pixels=1.)
    state = dict(sample=0, entities=[dict(id=4, position=torch.tensor([4., 8.]),
        observed=True, observation_age=0, association_strength=1.)],
        forecasts=[forecast, dict(forecast)])
    def forbidden_deepcopy(tensor, memo):
        raise AssertionError('generic tensor deepcopy used for standard records')
    monkeypatch.setattr(torch.Tensor, '__deepcopy__', forbidden_deepcopy)
    output = encoder.encode(state, None)
    assert torch.equal(output['forecasts'][0]['mixture_centers'], forecast['mixture_centers'])
    assert output['forecasts'][0]['mixture_centers'] is not forecast['mixture_centers']
    assert output['forecasts'][0]['mixture_centers'] is output['forecasts'][1]['mixture_centers']
    assert output['entities'][0]['position'] is not state['entities'][0]['position']


def test_component_order_matches_scalar_canonical_sort():
    from fly_connectome.vision.interface import _ordered_components

    forecasts = [
        dict(horizon_samples=4,
             normalized_centers=torch.tensor([[.25, -.5], [.25, -.5]], dtype=torch.float64),
             mixture_weights=torch.tensor([.3, .7], dtype=torch.float64)),
        dict(horizon_samples=1,
             normalized_centers=torch.tensor([[.1, .2], [.10000000001, .2]], dtype=torch.float64),
             mixture_weights=torch.tensor([.4, .6], dtype=torch.float64)),
    ]
    expected = torch.tensor(sorted((f['horizon_samples']-1, *p.tolist(), float(w))
                                   for f in forecasts
                                   for p, w in zip(f['normalized_centers'], f['mixture_weights'])))
    assert torch.equal(_ordered_components(forecasts), expected)


def test_frozen_evaluation_matches_scalar_reference_without_learning(tmp_path):
    torch.set_num_threads(1)
    actual, reference = load_default(), load_default()
    actual.save(tmp_path/'before.pt')
    previous = torch.zeros(64, 64)
    for sample in range(24):
        image = torch.zeros(64, 64)
        image[12:15, 5+sample:8+sample] = 1
        image[35:38, 40-sample:43-sample] = 1
        events = torch.stack(((previous-image).clamp(min=0),
                              (image-previous).clamp(min=0)))
        previous = image
        available_image = None if sample in (16, 17) else image
        assert_identical(actual.step(events, available_image, learn=False),
                         scalar_reference_step(reference, events, available_image, learn=False))
    actual.save(tmp_path/'after.pt')
    assert_identical(torch.load(tmp_path/'after.pt', weights_only=True),
                     torch.load(tmp_path/'before.pt', weights_only=True))


def test_batched_online_credit_matches_ordered_sequential_updates():
    sequential = ContextBelief(SpatialBelief())
    batched = ContextBelief(SpatialBelief())
    generator = torch.Generator().manual_seed(92732)
    updates = []
    for index in range(24):
        history = torch.randn(4 if index % 3 == 0 else 8, 2, generator=generator)
        displacement = torch.randn(2, generator=generator)
        centers = torch.randn(3, 2, generator=generator)
        weights = torch.softmax(torch.randn(3, generator=generator), 0)
        updates.append((history, displacement,
                        None if index % 5 == 0 else (centers, weights)))
    for history, displacement, credit in updates:
        sequential.observe(history, displacement, credit=credit)
    batched.observe_many(updates)
    for name in ('short', 'long'):
        left, right = getattr(sequential, name), getattr(batched, name)
        assert_identical((left.keys, left.values, left.seen, left.generator.get_state()),
                         (right.keys, right.values, right.seen, right.generator.get_state()))
    assert_identical(list(sequential.calibration.ranks), list(batched.calibration.ranks))
    assert_identical(sequential.distribution(updates[-1][0]),
                     batched.distribution(updates[-1][0]))
