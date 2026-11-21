from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt8_service_player import ServicePlayer
from scripts.experimental.tt9_receiving_lift import LiftServicePlayer
from scripts.experimental.tt7_contact import TIMINGS


def test_camera_toss_is_exempt_but_later_incoming_stroke_uses_lift(monkeypatch):
    base = (np.zeros(3), np.zeros(3), .2, 1.)
    monkeypatch.setattr(ServicePlayer, '_stroke_goals', lambda self, estimate: base)
    player = object.__new__(LiftServicePlayer)
    player.serve_plan = (17., (.07, -1.4))
    player.scene = SimpleNamespace(tick=17)
    player.servo = SimpleNamespace(observer=SimpleNamespace(delay=1.))
    player.receives = []
    player.timings = TIMINGS
    player.fixed_lift = 1
    estimate = (np.array([.1, -1.35, .25]), np.array([.01, -.09, .005]))
    assert player._stroke_goals(estimate) is base
    assert not player.receives
    player.scene.tick = 60
    goal, lead, pitch, horizon = player._stroke_goals(estimate)
    assert lead[2] > goal[2] > 0.
    assert (pitch, horizon) == base[2:]
    assert player.receives[0]['profile'] == 1
