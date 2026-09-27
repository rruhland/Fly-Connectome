import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from memory_context_comparison import MemoryComparison, ambiguity, history_groups
from fly_connectome.vision.dynamics import SpatialBelief


def test_recent_bank_preserves_stable_admission_and_issued_credit():
    source = SpatialBelief(capacity=4)
    recent = MemoryComparison(source, 'recent_stable')
    history = torch.tensor([[0., 1.]]*4)
    for i in range(520):
        target = torch.tensor([float(i % 3), 1.])
        credit = (torch.zeros(1, 2), torch.ones(1))
        source.observe(history, target, credit=credit)
        recent.observe(history, target, credit=credit)
    assert torch.equal(source.keys, recent.keys)
    assert torch.equal(source.values, recent.values)
    assert torch.equal(torch.stack(list(source.calibration.ranks)), torch.stack(list(recent.calibration.ranks)))
    assert len(recent.recent_keys) == 512
    assert recent.seen == source.seen


def test_append_retains_all_examples_and_normalized_distribution():
    model = MemoryComparison(SpatialBelief(capacity=2), 'append')
    history = torch.tensor([[0., 1.]]*4)
    for i in range(40):
        model.observe(history, torch.tensor([float(i), 1.]))
    assert len(model.keys) == 40
    centers, weights = model.distribution(history)
    assert centers.shape == (32, 2)
    assert torch.allclose(weights.sum(), torch.tensor(1.))


def test_history_aliasing_detects_distinct_futures_without_truth():
    def scene(last):
        positions = [{1: torch.tensor([0., float(t)])} for t in range(9)]
        positions.extend([{1: torch.tensor([0., float(last)])}])
        return dict(observed=positions)
    left = history_groups([scene(9)], 'raw4', 1)
    right = history_groups([scene(14)], 'raw4', 1)
    result = ambiguity(left, right)
    assert result['shared_keys'] == result['conflicting_keys'] == 1
    assert result['mean_gap'] == 5.
