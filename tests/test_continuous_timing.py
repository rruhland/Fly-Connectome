import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from continuous_timing import challenge, summarize


def test_schedule_is_continuous_and_omission_changes_only_one_pair():
    frames, times, blocks = challenge(False)
    omitted, other_times, other_blocks = challenge(True)
    np.testing.assert_array_equal(times, other_times)
    np.testing.assert_array_equal(blocks, other_blocks)
    state = frames[:, 0, 30, 39].numpy().astype(int)
    altered = omitted[:, 0, 30, 39].numpy().astype(int)
    np.testing.assert_array_equal(np.flatnonzero(np.diff(state))+1, times)
    np.testing.assert_array_equal(np.flatnonzero(np.diff(altered))+1, np.delete(times, [60, 61]))
    np.testing.assert_array_equal(frames[:times[60]], omitted[:times[60]])
    np.testing.assert_array_equal(frames[times[62]:], omitted[times[62]:])
    assert frames.sum((1, 2, 3)).eq(1).all()
    np.testing.assert_array_equal(np.diff(np.r_[20, times]), np.repeat([2, 3, 6, 4], 24))


def test_report_retains_missing_polarities_as_null_and_scores_quiet():
    result = summarize(np.array([0., 0.]), np.array([.2, -.1]))
    assert result['on_anticipation'] is None
    assert result['off_anticipation'] is None
    assert result['quiet_alarm_fraction'] == 1.
    assert summarize(np.array([-1., 1.]), np.array([-.7, .8]))['quiet_alarm_fraction'] is None
