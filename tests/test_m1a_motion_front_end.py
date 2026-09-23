import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from fly_connectome.sensor import EventCamera
from m1a_motion_front_end import event_centroid, median_displacement, opponent_score
from motion_t5_axis_aligned import moving_bar, static_bar


def test_event_centroid_tracks_both_polarities_from_events_only():
    for on in (False, True):
        for direction in (-1, 1):
            camera = EventCamera(1, 32, 64)
            camera.previous.fill_(not on)
            occupancy = torch.full((32 * 64,), not on, dtype=torch.bool)
            frames = moving_bar('vertical', 16, direction, 2)
            if on:
                frames = [~frame for frame in frames]
            positions = [event_centroid(camera.observe(frame), occupancy,
                                        on=on, width=64) for frame in frames]
            assert median_displacement(positions) == direction * 2


def test_static_events_do_not_imply_motion():
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(True)
    occupancy = torch.ones(32 * 64, dtype=torch.bool)
    positions = [event_centroid(camera.observe(frame), occupancy,
                                on=False, width=64)
                 for frame in static_bar('vertical', 16)]
    assert median_displacement(positions) == 0


def test_opponent_score_normalizes_for_local_cell_count():
    assert opponent_score({'T5c': 5, 'T5d': 10},
                          {'T5c': 10, 'T5d': 10}) == .5
    assert opponent_score({'T5c': 10, 'T5d': 5},
                          {'T5c': 10, 'T5d': 10}) == -.5
