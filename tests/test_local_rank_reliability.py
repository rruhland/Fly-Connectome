"""Direct local visual prediction error calibrates two readout sources."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_rank_reliability import LocalRankReliability, fit_reliability


def example(*, target_file=True, swap=False):
    target = torch.zeros((2, 1, 4))
    fast = torch.zeros_like(target)
    files = torch.zeros_like(target)
    fast[1, 0, 3 if swap else 0] = 1.
    files[1, 0, 0 if swap else 3] = 1.
    target[1, 0, (0 if swap else 3) if target_file else
           (3 if swap else 0)] = 1.
    return target, fast, files


def test_aligned_credit_prefers_a_reliably_correct_local_source():
    samples = [[example() for _ in range(20)]]
    learned = fit_reliability(samples, shuffled=False)
    _, fast, files = example()
    forecast = learned.forecast(fast, files)
    assert forecast[1, 0, 3] > forecast[1, 0, 0]


def test_shuffled_credit_breaks_source_specific_reliability():
    samples = [[example(), example(swap=True)]]
    aligned = fit_reliability(samples, shuffled=False)
    shuffled = fit_reliability(samples, shuffled=True)
    assert aligned.weight[1, 1, 0] > shuffled.weight[1, 1, 0]


def test_zero_source_activity_cannot_create_a_forecast():
    model = LocalRankReliability()
    empty = torch.zeros((2, 1, 4))
    model.observe(*example()[1:], example()[0])
    assert not bool(model.forecast(empty, empty).any())
