import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_tm9_graded import moving_bar_frames


def test_doubled_speed_bar_keeps_thirteen_frames_and_off_polarity():
    right = moving_bar_frames(32, 1, 2)
    left = moving_bar_frames(32, -1, 2)
    assert len(right) == len(left) == 18
    assert not right[2][0, 16, 20]
    assert not left[2][0, 16, 44]
    assert not right[8][0, 16, 32]
    assert right[8][0, 16, 40]
    assert right[0].all() and right[15].all()
