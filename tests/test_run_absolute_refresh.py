"""Absolute refresh evaluation uses camera contrast, not entity labels."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_absolute_refresh import contrast_frames, evaluate_streams


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 20))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_absolute_contrast_follows_clean_camera_events():
    frames = contrast_frames([event(on=((5, 5), (5, 6))),
                              event(off=((5, 5),))])
    assert frames[0][5, 5] == 1
    assert frames[1][5, 5] == 0
    assert frames[1][5, 6] == 1


def test_refresh_arms_score_identical_clean_next_event():
    first = event(on=((5, 5), (5, 6)))
    second = event(on=((5, 7),), off=((5, 5),))
    scores, _ = evaluate_streams([([first, second], [first, second])],
                                 links={}, expected=[1])
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in scores.values())
