"""Assembly scoring counts missed entities and never shares one track."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_local_motion_assemblies import (case_events, direction_from_history,
                                         score_tracks)


def test_direction_requires_a_linked_displacement():
    assert direction_from_history([(4, 5, 5)]) is None
    assert direction_from_history([(4, 5, 5), (7, 5, 8)]) == 'right'
    assert direction_from_history([(4, 8, 5), (7, 5, 5)]) == 'up'


def test_one_track_cannot_satisfy_two_entities():
    case = dict(split='separated', polarity='dark', objects=[
        dict(shape='square', center=(12, 20), direction='right', speed=1),
        dict(shape='square', center=(20, 44), direction='left', speed=1)])
    token = dict(history=[(t, 12, 20+t-8) for t in range(4, 13)])
    result = score_tracks(case, [token])
    assert result['total'] == 2
    assert result['matched'] == 1
    assert result['correct'] == 1


def test_case_events_keeps_the_full_camera_stream():
    case = dict(split='position', polarity='dark', objects=[
        dict(shape='square', center=(16, 32), direction='right', speed=1)])
    events = case_events(case)
    assert len(events) == 18
    assert events[4].shape == (2, 32, 64)
