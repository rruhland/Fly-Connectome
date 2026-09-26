"""Transfer scoring keeps visual targets separate from observed noise."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_zero_shot_visual_transfer import (collect_corrupted_samples,
                                           corrupt_events, file_activity)
from sparse_recurrent_field import SparseRecurrentField


def event(y, x):
    result = torch.zeros((2, 32, 64))
    result[1, y, x] = 1.
    result[1, y+1, x+1] = 1.
    return result


def test_corruption_changes_only_observed_event_stream():
    original = [event(5, 5)]
    clean = corrupt_events(original, seed=7, dropout=0., false_rate=0.)
    dropped = corrupt_events(original, seed=7, dropout=1., false_rate=0.)
    torch.testing.assert_close(clean[0], original[0])
    assert not bool(dropped[0].any())
    assert int(original[0].sum()) == 2


def test_corrupted_collection_scores_clean_future_event():
    clean = [event(5, 5), event(5, 6)]
    observed = [event(5, 5), torch.zeros_like(clean[1])]
    samples = collect_corrupted_samples(
        SparseRecurrentField(units=1), [(clean, observed)], {})
    assert len(samples[0]) == 1
    torch.testing.assert_close(samples[0][0][0], clean[1])


def test_file_activity_counts_autonomous_selected_hypotheses():
    links = {(1., 1): True}
    result = file_activity([[event(5, 5)]], links)
    assert result['active_frames'] == 1
    assert result['mean_live_files'] == 1.
    assert result['max_live_files'] == 1
