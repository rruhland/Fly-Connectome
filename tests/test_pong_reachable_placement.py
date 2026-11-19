"""Test the public contact plane and the effect of pending motor requests."""

from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_rapid_motor_recovery import VisualMotorBelief
from pong_reachable_placement import GuardedScorePlacement, ReachablePlacement, ShotGeometry, motor_bounds
from pong_score_placement import ScorePlacement
from pong_timing_controller import ArrivalContext


def test_reach_horizon_contains_no_new_actuation_before_queue_drains():
    belief = VisualMotorBelief((1, 2, 0))
    belief.initialize(0.)
    timing = SimpleNamespace(belief=belief, requests=deque([0] * 12),
                             delay_commands=12, camera_seconds=.02)
    low, high = motor_bounds(timing, .15, 0.)
    assert low == high < 0.
    low, high = motor_bounds(timing, .6, 0.)
    assert low < high


def test_centered_forecast_uses_observed_contact_plane():
    history = deque((t, .3 * t, -.5 * t) for t in (.5, .52, .54, .56))
    timing = SimpleNamespace(seconds=.56, inherited=SimpleNamespace(history=history, bound_ids=None))
    geometry = ShotGeometry(-.435, .435, .49, .55, 9., 4., (.35, .65))
    controller = ReachablePlacement(timing, geometry, aimed=False)
    context = ArrivalContext(.27, .34, 0., -.4, .4, .27)
    assert controller._offset(context) == pytest.approx(-.009)
    assert controller._offset(None) == 0.


def test_frozen_score_ranking_obeys_reach_and_waits_for_timing_evidence():
    policy = ScorePlacement()
    for _ in range(4):
        policy.observe_score(5, .22, 1.)
    policy.observe_score(1, .22, 1.)
    frozen = policy.frozen_branch()
    belief = VisualMotorBelief((0, 1, 2))
    timing = SimpleNamespace(belief=belief, candidate=12)
    geometry = ShotGeometry(-.435, .435, .49, .55, 9., 4., (.35, .65))
    controller = GuardedScorePlacement(timing, geometry, frozen.episode(1))
    assert controller._select_placement(.22, .6, (.1, .315), .15, .2, None) is None
    assert controller.placement.choices == []
    timing.candidate = None
    assert controller._select_placement(.22, .6, (.1, .315), .15, .2, None) == (.079, None)
    assert frozen.to_state() == policy.to_state()
