"""Exact reference checks for lower-cost promoted visual-state execution."""

import torch

from fly_connectome.vision import VisualStateEncoder, load_default
from fly_connectome.vision.dynamics import mixture_quantile
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


def scalar_reference_step(model, events, image):
    state = StreamingVisualState.step(model, events, image, learn=True)
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
