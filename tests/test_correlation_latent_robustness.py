"""Scene-generator checks for frozen generic visual robustness."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correlation_latent_robustness import make_robust_cases, scene_sequence


def test_two_independent_movers_create_more_events_than_one():
    first = dict(shape='diamond', center=(16, 20), before=(-1, 0),
                 after=(-1, 0))
    second = dict(shape='ring', center=(16, 44), before=(1, 0),
                  after=(1, 0))
    single = scene_sequence([first], background=True)
    pair = scene_sequence([first, second], background=True)
    assert len(pair) == 18
    assert all(frame.shape == (2, 32, 64) for frame in pair)
    assert int(pair[6].sum()) > int(single[6].sum())


def test_robust_suite_contains_each_generic_family():
    cases = make_robust_cases()
    families = {family for family, *_ in cases}
    assert families == {'independent', 'crossing', 'occlusion',
                        'speed_change', 'noise'}
    assert all(len(events) == 18 and torch.isfinite(torch.stack(events)).all()
               for _, _, events in cases)
