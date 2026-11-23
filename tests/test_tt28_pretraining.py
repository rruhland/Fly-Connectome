import numpy as np

from scripts.experimental.tt28_when_to_switch import stable
from scripts.experimental.tt28_toss_prior import toss


def test_short_or_recently_changed_tool_estimates_do_not_end_practice():
    axis = np.array([0.0, 0.0, 1.0])
    assert not stable([axis] * 5)[0]
    assert stable([axis] * 7)[0]
    changed = np.array([np.sin(0.2), 0.0, np.cos(0.2)])
    assert not stable([axis] * 12 + [changed] * 3)[0]


def test_directional_toss_changes_the_approach_without_changing_the_seeded_drop_height():
    slow = toss(593814, "practice", "standard", (0.5, 0.5))
    fast = toss(593814, "practice", "standard", (3.5, 3.5))
    assert slow.pose == fast.pose and slow.offset[2] == fast.offset[2]
    time = (slow.velocity[2] + np.sqrt(slow.velocity[2] ** 2 + 2 * 9.81 * slow.offset[2])) / 9.81
    assert np.allclose(
        np.array(slow.offset) + time * np.array(slow.velocity),
        np.array(fast.offset) + time * np.array(fast.velocity),
    )
