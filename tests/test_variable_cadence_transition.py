"""Generic fractional-speed visual experience stays label-free to learners."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_variable_cadence_transition import continuous_events


def test_fractional_moving_dot_has_sparse_event_frames():
    slow = continuous_events('dot', (16, 32), (0, 1), .25, False)
    fast = continuous_events('dot', (16, 32), (0, 1), 2, False)
    assert len(slow) == len(fast) == 18
    assert all(event.shape == (2, 32, 64) for event in slow)
    assert sum(event.sum() > 0 for event in slow) < sum(
        event.sum() > 0 for event in fast)
