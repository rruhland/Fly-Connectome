import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from m1a_pong_state_ceiling import EventBallTracker


def test_event_only_tracker_localizes_ball_and_excludes_paddles():
    pong = Pong([17])
    camera = EventCamera(1, 32, 64)
    tracker = EventBallTracker(batch=1)
    observation = tracker.observe(camera.observe(pong.render()))
    assert torch.isfinite(observation).all()
    assert torch.max((observation - pong.ball).abs()) < .03
    empty = torch.zeros((1, 32, 64), dtype=torch.bool)
    assert torch.isnan(tracker.observe(camera.observe(empty))).all()
