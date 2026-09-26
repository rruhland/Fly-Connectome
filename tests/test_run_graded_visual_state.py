"""Graded visual state arms share the same clean next-event target."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_graded_visual_state import evaluate_streams


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 20))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_all_arms_score_the_same_future_camera_event():
    first = event(on=((5, 5), (5, 6)))
    second = event(on=((5, 7),), off=((5, 5),))
    shuffled = [event(on=((5, 15), (5, 16))),
                event(on=((5, 17),), off=((5, 15),))]
    scores, state = evaluate_streams(
        [([first, second], [first, second], shuffled)],
        links={}, expected=[1])
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in scores.values())
    assert all(row['active_frames'] == 2 for row in state.values())
