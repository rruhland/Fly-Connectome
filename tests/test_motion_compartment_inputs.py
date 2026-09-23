import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_compartment_inputs import direction_index, frames


def test_opposite_motion_has_matched_occupancy_on_both_axes():
    for axis in ('horizontal', 'vertical'):
        right, left = frames(18, axis, 1), frames(18, axis, -1)
        assert all(a.equal(b) for a, b in zip(right[2:15], reversed(left[2:15])))
        assert [int((a != b).sum()) for a, b in zip(right[1:], right[:-1])] == [
            int((a != b).sum()) for a, b in zip(left[1:], left[:-1])]


def test_direction_index_is_zero_without_activity():
    assert direction_index(0, 0) == 0
    assert direction_index(3, 1) == .5
