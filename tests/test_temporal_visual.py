import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from temporal_visual import oscillation, recurrent_boundaries, trace_available


def test_two_position_dot_gives_real_recurrent_on_off_events():
    from fly_connectome.sensor import EventCamera
    frames = oscillation(blank=2, cycles=2, dwell=3)
    camera = EventCamera(1, 32, 64)
    event_ticks = []
    for frame, image in enumerate(frames):
        events = camera.observe(image)
        hit = events.pixels == 30*64+39
        if hit.any():
            event_ticks.append((frame, bool(events.on[hit][0])))
    assert event_ticks == [(2, True), (5, False), (8, True), (11, False)]
    assert not frames[-1].any()
    assert torch.equal(frames[2], frames[8])


def test_recurrent_scoring_keeps_quiet_frames_and_excludes_uncued_first_cycle():
    indices = recurrent_boundaries([2, 1], cycles=2, dwell=3)
    # Trial lengths 15 and14; exclude only first six dot frames per trial.
    forecast_targets = (indices+8)//8
    assert 1 in forecast_targets and 8 in forecast_targets and 14 in forecast_targets
    assert not set(range(2, 8)) & set(forecast_targets)
    assert not set(range(16, 22)) & set(forecast_targets)
    assert forecast_targets[-1] == 28


def test_gate_tests_trace_information_without_requiring_a_trained_forecast():
    assert trace_available(-1, np.array([-.3, .2]), np.array([-.1, .2]))
    assert trace_available(1, np.array([-.3, .2]), np.array([-.3, .4]))
    assert not trace_available(1, np.array([-.3, .2]), np.array([0., .2]))
    assert not trace_available(-1, np.array([0., .2]), np.array([0., .4]))
