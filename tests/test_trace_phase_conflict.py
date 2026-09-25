"""Gap-exit, quiet, visible, and post-gap targets are distinct frames."""

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trace_phase_conflict import phase_indices, score_ranking


def test_phase_indices_cover_causal_forecast_targets():
    assert phase_indices((7, 8, 9)) == dict(
        exit=9, quiet=8, visible=5, post=10)
    assert phase_indices((6, 7)) == dict(
        exit=7, quiet=6, visible=4, post=8)
    assert phase_indices((8, 9, 10, 11)) == dict(
        exit=11, quiet=10, visible=6, post=12)


def test_ranking_distinguishes_future_event_from_quiet_pixels():
    event = torch.zeros((2, 32, 64))
    event[0, 0, 0] = 1
    matrix = sparse.csr_matrix(([1.], ([0], [0])), shape=(2048, 1))
    quiet = sparse.csr_matrix(([.1], ([0], [0])), shape=(2048, 1))
    weights = np.array([[1., 0.]])
    result = score_ranking(matrix, [event], quiet, weights)
    assert result['event_vs_quiet_auc'] == 1
