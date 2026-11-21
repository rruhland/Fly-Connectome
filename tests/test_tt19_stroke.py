import json
from dataclasses import asdict

import numpy as np

from scripts.experimental.tt19_stroke import RallyStrokePoint, Stroke, StrokePoint, StrokeSearch, optical_strokes


def test_visible_positions_recover_forward_stroke_without_rally_events():
    dt = .033
    blade = [0., -.90, .18]
    positions = [[0., -.90 + .065 * (5 - i), .18] for i in range(6)]
    positions.extend([[0., -.90 + .095 * i, .18 + .013 * i] for i in range(1, 5)])
    frames = [[ball, blade, .21] for ball in positions]
    hits = optical_strokes(frames, dt, server=-1, reward=1.)
    assert len(hits) == 1
    assert hits[0]['tick'] == 5
    np.testing.assert_allclose(hits[0]['outgoing'][1], .095 / dt)


def test_saved_stroke_search_preserves_policy_and_keeps_frozen_attempts_out_of_training():
    first = StrokeSearch(19, source_style={1: {'pitch': .15, 'slope': -.6, 'velocity': [0., .6, 0.]}})
    assert first.choose(1, train=True) == Stroke.empty()
    first.probes = [{'action': [0., .77, 0., 0., 0., 0.], 'velocity': [0., .64, 0.]}]
    frames = [[[0., -.90, .26], [0., -.95, .18], .15]] * 12
    first.finish(Stroke.empty(), 1, -1, frames, .033, 9, train=True)
    recorded = json.loads(json.dumps(first.state()))
    restored = StrokeSearch.restore(recorded)
    assert restored.choose(1, train=True) == first.choose(1, train=True)
    assert restored.choose(1, train=False) == first.choose(1, train=False)
    restored.finish(Stroke.empty(), 1, 1, frames, .033, 9, train=False)
    assert len(restored.archive[1]) == 1


def test_timing_uses_latest_optical_tick_when_a_capture_is_skipped():
    swing = Stroke((0.,) * 5, (0., .65, 0., 0., 0.), .095)
    point = StrokePoint(swing, None, -1, 1. / 30)
    for tick in range(12):
        if tick % 2:
            point.observe(None)
        else:
            point.observations.append([[0., -.31 - tick * .054, .29], [0., -.92, .25], None])
        command = point.command()
        if tick < 9:
            assert command == (0.,) * 6
    assert point.swing_tick == 9
    assert command == (0., .65, 0., 0., 0., 0.)


def test_observed_bounces_on_both_halves_can_retrieve_a_serving_stroke():
    placement = Stroke((0.,) * 5, (0., .65, 0., 0., 0.), .095)
    rows = [
        dict(server=1, stroke=asdict(Stroke.empty()), public_reward=1,
             detail=dict(optical_bounces=[dict(position=(0., -.45, .05))])),
        dict(server=1, stroke=asdict(placement), public_reward=-1,
             detail=dict(optical_bounces=[dict(position=(0., -.45, .05)), dict(position=(0., .45, .05))])),
    ]
    search = StrokeSearch(19)
    search.prefer_placement(rows)
    assert search.choose(1, train=False) == placement


def test_rearmed_stroke_waits_for_observed_outward_flight_then_another_incoming_ball():
    drive = (0., .55, 0., 0., 0.)
    receive = Stroke((0.,) * 5, drive, .085)
    point = RallyStrokePoint(receive, receive, None, -1, 1. / 30.)
    for tick in range(63):
        if tick < 13:
            ball_y = -.25 - .05 * tick
        elif tick < 35:
            ball_y = -.90 + .070 * (tick - 13)
        else:
            ball_y = .64 - .068 * (tick - 35)
        point.observations.append([[0., ball_y, .25], [0., -.90, .20], None])
        point.command()
    assert point.swing_ticks[0] in (10, 11, 12)
    assert len(point.swing_ticks) == 2
    assert point.swing_ticks[1] > 50
