"""Evidence-file arms share clean targets and expose autonomous live state."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_predictive_file_evidence import evaluate_episodes
from sparse_recurrent_field import SparseRecurrentField


def event(y, x):
    result = torch.zeros((2, 32, 64))
    result[1, y, x:x+2] = 1.
    return result


def test_all_file_arms_score_the_same_next_event():
    episodes = []
    for y in (5, 15):
        events = [event(y, 5), event(y, 6)]
        episodes.append((events, events))
    scores, activity = evaluate_episodes(
        SparseRecurrentField(units=1), episodes, {})
    assert all(row['events'] == 2 and row['targets'] == 4
               for row in scores.values())
    assert activity['base']['active_frames'] == 4


def test_unconfirmed_burst_does_not_create_a_live_file():
    events = [event(5, 5), torch.zeros((2, 32, 64))]
    _, activity = evaluate_episodes(
        SparseRecurrentField(units=1), [(events, events),
                                        (events, events)], {})
    assert activity['base']['mean_live_files'] == 1.
    assert activity['confirmation']['mean_live_files'] == 0.
