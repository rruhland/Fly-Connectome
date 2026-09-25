"""Forecast scoring must retain polarity and count quiet-frame alarms."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_observation_horizon_forecast import (empty_score, score_event,
                                              sensory_sequence)


def test_wrong_sign_is_both_false_positive_and_miss():
    row = empty_score()
    prediction = torch.zeros((2, 32, 64))
    target = torch.zeros_like(prediction)
    prediction[0, 8, 9] = 1
    target[1, 8, 9] = 1
    score_event(row, prediction, target, target)
    assert (row['tp'], row['fp'], row['fn']) == (0, 1, 1)
    assert row['novel_recalled'] == 0


def test_quiet_target_counts_predicted_alarm():
    row = empty_score()
    prediction = torch.zeros((2, 32, 64))
    prediction[1, 8, 9] = 1
    score_event(row, prediction, torch.zeros_like(prediction),
                torch.zeros_like(prediction))
    assert row['quiet_frames'] == 1
    assert row['quiet_alarm_pixels'] == 1


def test_recent_sensory_sequence_uses_only_past_events():
    on = torch.zeros((2, 32, 64))
    on[1, 8, 9] = 1
    off = torch.zeros_like(on)
    off[0, 8, 9] = 1
    frames = sensory_sequence([on, torch.zeros_like(on), off], recent=True)
    assert frames[0][2].shape == (4, 32, 64)
    assert frames[0][2][2, 8, 9] == 0
    assert 0 < frames[1][2][3, 8, 9] < 1
    assert frames[2][2][0, 8, 9] == 1
