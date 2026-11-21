from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from tt3_active_stroke import ActiveStrokeMemory, Pulse


def test_momentum_waits_for_a_repeat_approach_and_preserves_transverse_goal():
    stationary = np.array((0., .13, -.07))

    class CarriedGoal:
        def goal(self, *_args, **_kwargs):
            return stationary.copy()

    memory = ActiveStrokeMemory(CarriedGoal(), Pulse(.2))
    normal = np.array((1., 0., 0.))
    ready = np.zeros(3)
    trajectory = ((.19, -.02), (.11, -.02), (.06, -.02), (.01, -.02),
                  (.025, .02), (.11, .02), (.17, .02), (.23, .02), (.21, -.02),
                  (.16, -.02), (.12, -.02), (.045, -.02))
    for frame, (gap, speed) in enumerate(trajectory):
        memory.frame = frame
        goal = memory.goal(0, ready, gap * normal, speed * normal, ready, normal=normal)
        np.testing.assert_array_equal(goal[1:], stationary[1:])
        if frame < 8:
            np.testing.assert_array_equal(goal, stationary)
    assert memory.recurrence.repeat_at == 8
    assert len(memory.active) > 0
    assert goal[0] < 0.
