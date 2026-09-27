import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from long_context_comparison import ContextBelief
from continual_vision_audit import CausalDynamicsStream
from fly_connectome.vision.dynamics import SpatialBelief, mixture_pit


def test_long_context_learns_distinct_futures_with_identical_last_four_steps():
    history_a = torch.tensor([[0., 1.]]*4+[[1., 0.]]*4)
    history_b = torch.tensor([[0., -1.]]*4+[[1., 0.]]*4)
    model = ContextBelief(SpatialBelief(), 8)
    for _ in range(32):
        model.observe(history_a, torch.tensor([10., 0.]))
        model.observe(history_b, torch.tensor([-10., 0.]))
    assert torch.allclose(model.predict(history_a), torch.tensor([10., 0.]), atol=1e-4)
    assert torch.allclose(model.predict(history_b), torch.tensor([-10., 0.]), atol=1e-4)
    assert torch.equal(history_a[-4:], history_b[-4:])


def test_fallback_and_calibration_use_available_history_and_issued_credit():
    model = ContextBelief(SpatialBelief(), 8)
    history = torch.tensor([[0., 1.]]*8)
    credit = (torch.tensor([[4., 2.]]), torch.ones(1))
    target = torch.tensor([2., 1.])
    for _ in range(32):
        model.observe(history, target, credit=credit)
    assert torch.equal(model.calibration.ranks[-1], mixture_pit(*credit, target))
    assert model.uses_context(history)
    assert not model.uses_context(history[-4:])
    assert torch.equal(model.predict(history[-4:]), model.short.predict(history[-4:]))


def test_extended_stream_resets_contiguous_history_after_missing_evidence():
    model = ContextBelief(SpatialBelief(), 8)
    stream = CausalDynamicsStream({1: model})
    for t in range(12):
        stream.step({1: torch.tensor([0., float(t)])}, learn=True)
    assert len(model.long.keys) == 3
    stream.step({}, available=False, learn=True)
    for t in range(13, 18):
        stream.step({1: torch.tensor([0., float(t)])}, learn=True)
    assert len(model.long.keys) == 3
    assert len(stream.histories[1]) <= 9
