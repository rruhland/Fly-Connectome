import sys
import copy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from long_context_comparison import ContextBelief as ExperimentalContext
from continual_vision_audit import CausalDynamicsStream
from fly_connectome.vision import load_default, load_legacy_default
from fly_connectome.vision.dynamics import ContextBelief, SpatialBelief
from fly_connectome.vision.state import ProbabilisticVisualState


def test_default_remains_legacy_until_active_recovery_passes():
    current, legacy = load_default(), load_legacy_default()
    for h, model in current.dynamics.items():
        assert type(model) is SpatialBelief
        assert torch.equal(model.keys, legacy.dynamics[h].keys)


def test_context_matches_approved_experiment():
    actual = ContextBelief(SpatialBelief())
    expected = ExperimentalContext(SpatialBelief(), 8)
    generator = torch.Generator().manual_seed(927)
    for t in range(100):
        history = torch.randn(8 if t % 3 else 4, 2, generator=generator)
        target = torch.randn(2, generator=generator)
        for a, b in zip(actual.distribution(history), expected.distribution(history)):
            assert torch.equal(a, b)
        credit = actual.predict_with_credit(history)[1]
        actual.observe(history, target, credit=credit)
        expected.observe(history, target, credit=credit)
    for name in ('short', 'long'):
        assert torch.equal(getattr(actual, name).keys, getattr(expected, name).keys)
        assert torch.equal(getattr(actual, name).values, getattr(expected, name).values)
    assert torch.equal(torch.stack(list(actual.calibration.ranks)),
                       torch.stack(list(expected.calibration.ranks)))


def test_upgrade_and_checkpoint_preserve_banks_in_fresh_scene(tmp_path):
    model = load_legacy_default()
    prior = model.dynamics[4].keys.clone()
    model.upgrade_temporal_context()
    assert torch.equal(model.dynamics[4].short.keys, prior)
    assert model.dynamics[4].long.keys.shape == (0, 16)
    history = torch.tensor([[0., 1.]] * 8)
    for _ in range(40):
        for bank in model.dynamics.values():
            _, credit = bank.predict_with_credit(history)
            bank.observe(history, torch.tensor([1., 2.]), credit=credit)
    path = tmp_path / 'context.pt'
    model.save(path)
    assert torch.load(path, weights_only=True)['version'] == 2
    restored = ProbabilisticVisualState.load(path)
    assert restored.state.tracker.frame == -1
    assert not restored.pending
    for h, bank in model.dynamics.items():
        other = restored.dynamics[h]
        for name in ('short', 'long'):
            assert torch.equal(getattr(bank, name).keys, getattr(other, name).keys)
            assert torch.equal(getattr(bank, name).values, getattr(other, name).values)
        assert torch.equal(bank.predict(history), other.predict(history))
        assert torch.equal(torch.stack(list(bank.calibration.ranks)),
                           torch.stack(list(other.calibration.ranks)))


def test_active_camera_credit_outages_and_reload_match_experiment(tmp_path):
    torch.set_num_threads(1)
    live = load_legacy_default().upgrade_temporal_context()
    reference = CausalDynamicsStream({h: ExperimentalContext(m, 8)
                                     for h, m in load_legacy_default().dynamics.items()})
    previous = torch.zeros(64, 64)
    generator = torch.Generator().manual_seed(928)
    cached = {}
    active = 0
    cached_checks = 0
    for t in range(110):
        frame = torch.zeros(64, 64)
        x = 5 + (t % 70 if t % 70 < 35 else 69-t % 70)
        frame[15:18, x:x+3] = 1
        frame[40:43, 55-x:58-x] = 1
        events = torch.stack(((previous-frame).clamp(min=0), (frame-previous).clamp(min=0)))
        previous = frame
        unavailable = 70 <= t < 74
        image = (.2+.5*frame+.01*torch.randn(frame.shape, generator=generator)).clamp(0, 1)
        state = live.step(events, None if unavailable else image, learn=True)
        positions = {e['id']: e['position'] for e in state['entities'] if e['observed']}
        expected = reference.step(positions, available=not unavailable, learn=True)
        if unavailable:
            assert state['entities']
            assert all(any(f['id'] == e['id'] and f['evidence_age'] > 0
                           for f in state['forecasts']) for e in state['entities'])
        for forecast in state['forecasts']:
            key = (forecast['id'], forecast['origin_sample'],
                   state['sample']+forecast['horizon_samples'])
            if forecast['evidence_age']:
                assert torch.equal(forecast['mixture_centers'], cached[key])
                cached_checks += 1
                continue
            cached[key] = forecast['mixture_centers'].clone()
            other = next(f for f in expected if f['id'] == forecast['id'] and
                         f['horizon'] == forecast['horizon_samples'])
            assert torch.equal(forecast['mixture_centers'], other['centers'])
            assert torch.equal(forecast['mixture_weights'], other['weights'])
            model = live.dynamics[forecast['horizon_samples']]
            active += model.uses_context(torch.empty(len(live.observed_history(forecast['id']))-1, 2))
        if t == 85:
            path = tmp_path / 'active.pt'
            live.save(path)
            restored = ProbabilisticVisualState.load(path)
            replay = copy.deepcopy(live)
            replay.reset_state()
            live.reset_state()
            reference = CausalDynamicsStream(reference.dynamics)
            assert not live.histories and not live.pending and not live.forecast_cache
            # Restart both cameras from the same fresh scene and learned parameters.
            for index in range(12):
                a = replay.step(events, image, learn=True)
                b = restored.step(events, image, learn=True)
                assert torch.equal(a['support_field'], b['support_field'])
                for first, second in zip(a['forecasts'], b['forecasts']):
                    assert torch.equal(first['mixture_centers'], second['mixture_centers'])
                    assert torch.equal(first['marginal_interval_90'], second['marginal_interval_90'])
            for h, bank in replay.dynamics.items():
                assert torch.equal(bank.long.keys, restored.dynamics[h].long.keys)
                assert torch.equal(bank.short.values, restored.dynamics[h].short.values)
    assert active > 100
    assert cached_checks > 0
    for h, model in live.dynamics.items():
        other = reference.dynamics[h]
        for name in ('short', 'long'):
            assert torch.equal(getattr(model, name).keys, getattr(other, name).keys)
            assert torch.equal(getattr(model, name).values, getattr(other, name).values)
        assert torch.equal(torch.stack(list(model.calibration.ranks)),
                           torch.stack(list(other.calibration.ranks)))
