"""Check causal command transport and camera-only timing inference."""

import numpy as np
import pytest

from scripts.experimental.tt4_transfer_trial import _write
from scripts.experimental.tt20_latency import CommandPipe, blade_motion, identify_delay, scene_motion
from scripts.experimental.tt20_latency_trial import CHECK_SEED, FRESH_RECEIPT, RECEIPT, schedule_starts


def test_fifo_applies_a_command_exactly_four_control_frames_later():
    requests = np.linspace(-1., 1., 90).reshape(15, 6)
    pipe = CommandPipe(4)
    applied = np.asarray([pipe.apply(row) for row in requests])
    np.testing.assert_array_equal(applied[:4], np.zeros((4, 6)))
    np.testing.assert_array_equal(applied[4:], requests[:-4])


def test_delay_estimate_recovers_a_shift_with_different_motion_scale():
    expected = np.array([0., 0., .8, .7, .5, .3, .1, .2, .05, .5, .9, .62, 0., 0.])
    observed = 1.35 * np.r_[np.zeros(4), expected[:-4]]
    recovered = identify_delay(expected, observed)
    assert recovered['delay_frames'] == 4
    assert recovered['margin'] > .4


def test_either_view_supplies_motion_without_coordinates_from_the_plant():
    neutral = np.zeros((100, 120, 3), dtype='uint8')
    first, second = neutral.copy(), neutral.copy()
    first[25:35, 20:28, 0] = 255
    second[25:35, 27:35, 0] = 255
    assert blade_motion([first, second])[0] > 6.
    assert scene_motion([first, second])[0] > 0.


def test_schedule_allows_its_own_saved_receipt_and_rejects_prior_points(tmp_path):
    previous, current = tmp_path / 'previous', tmp_path / 'current'
    previous.mkdir()
    current.mkdir()
    own = current / RECEIPT
    starts = schedule_starts(previous, current, own, CHECK_SEED, 8)
    _write(own, dict(held_starts=starts))
    assert schedule_starts(previous, current, own, CHECK_SEED, 8) == starts
    _write(current / FRESH_RECEIPT, dict(pairs={'left': [dict(seed=starts[0])]}))
    with pytest.raises(ValueError, match='overlap'):
        schedule_starts(previous, current, own, CHECK_SEED, 8)
