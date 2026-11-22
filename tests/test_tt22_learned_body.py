import numpy as np

from scripts.experimental.tt20_motor_carry import Wiring
from scripts.experimental.tt22_learned_body import ActuatorBridge, recode


class Recorder:
    def step(self, action, *, render):
        return action, render


def test_body_translation_is_applied_to_the_actual_step():
    wiring = Wiring((0, 2, 3, 4, 1), (-1, -1, 1, -1, 1))
    command = np.array((.41, -.34, .28, .17, -.08, 0.))
    arena = Recorder()
    learned = ActuatorBridge(arena, wiring, wiring)
    result, render = arena.step(command, render=False)
    assert not render and np.array_equal(result, command)
    assert learned.public[0] != command.tolist()
    assert recode([command], learned.public, learned.realized)['physical_differs'] == 0

    arena = Recorder()
    unadapted = ActuatorBridge(arena, wiring, None)
    result, _ = arena.step(command, render=False)
    assert np.array_equal(result, wiring.physical(command))
    assert unadapted.public[0] == command.tolist()
    assert recode([command], unadapted.public, unadapted.realized)['physical_differs'] == 1
