import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_t5_axis_aligned import moving_bar


def test_axis_aligned_off_bars_move_on_expected_axis():
    horizontal = moving_bar('horizontal', 18, 1, 1)
    vertical = moving_bar('vertical', 10, -1, 1)
    assert len(horizontal) == len(vertical) == 18
    assert horizontal[0].all() and vertical[15].all()
    assert not horizontal[8][0, 16, 18]
    assert horizontal[8][0, 16, 28]
    assert not horizontal[14][0, 16, 24]
    assert not vertical[8][0, 10, 32]
    assert vertical[8][0, 10, 42]
    assert not vertical[14][0, 4, 32]
