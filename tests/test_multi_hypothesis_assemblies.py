"""Delayed local competition preserves alternatives and distinct movers."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multi_hypothesis_assemblies import MultiHypothesisAssemblies


def candidate(y, x):
    return (y, x, torch.tensor([1., 0.]))


def test_ambiguous_candidate_retains_multiple_causal_predecessors():
    model = MultiHypothesisAssemblies(beam=3)
    model.step(0, [candidate(4, 4), candidate(4, 6)])
    model.step(1, [candidate(4, 5)])
    histories = [hypothesis['history'] for hypothesis in model.recent[-1]]
    assert any(history[0][2] == 4 for history in histories)
    assert any(history[0][2] == 6 for history in histories)


def test_smooth_rightward_path_survives_an_ambiguous_middle_frame():
    model = MultiHypothesisAssemblies(beam=3)
    model.step(0, [candidate(4, 4)])
    model.step(1, [candidate(4, 4), candidate(4, 5)])
    model.step(2, [candidate(4, 6)])
    paths = [row['history'] for row in model.tracks()]
    assert [(0, 4, 4), (1, 4, 5), (2, 4, 6)] in paths


def test_two_distant_paths_remain_separate_across_quiet_frames():
    model = MultiHypothesisAssemblies(beam=3, max_gap=4)
    model.step(0, [candidate(2, 2), candidate(7, 7)])
    model.step(1, [])
    model.step(2, [candidate(2, 3), candidate(7, 6)])
    paths = [row['history'] for row in model.tracks()]
    assert any(path[0][1:] == (2, 2) and path[-1][1:] == (2, 3)
               for path in paths)
    assert any(path[0][1:] == (7, 7) and path[-1][1:] == (7, 6)
               for path in paths)
