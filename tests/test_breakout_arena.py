"""Verify the native control boundary and evaluation-only Breakout events."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from breakout_arena import (BreakoutCamera, BreakoutLayout, BreakoutServe, brick_bounds,
                            breakout_schedule, play_breakout)
from fly_connectome.control.environment import PongCamera
from fly_connectome.vision import VisualStateEncoder, load_default


def run_to_end(camera, action=1):
    transitions = []
    while not camera.needs_reset:
        transitions.append(camera.step(action))
    assert sum(step.hits for step in transitions) == camera.summary()['own_returns']
    assert sum(step.reward for step in transitions) == camera.summary()['score']
    return transitions


def test_native_control_and_camera_cadence_with_remap_and_delay():
    camera = BreakoutCamera(7, mapping=(1, 2, 0), delay_frames=2, max_episode_ticks=40)
    pong = PongCamera(7, max_episode_ticks=40)
    initial = camera.frame()
    assert initial.shape == (64, 64) and initial.dtype == torch.bool
    assert initial[60, 32] and not initial[32, 60]
    actual_ticks = []
    for i, requested in enumerate((1, 1, 2, 0, 0, 0, 0)):
        physical = (1, 1, 2, 2, 0, 1, 1)[i]
        a, b = camera.step(requested), pong.step(physical)
        assert a.physics_ticks == b.physics_ticks
        assert a.elapsed_seconds == b.elapsed_seconds
        assert torch.equal(camera.body.position, pong.game.body.position)
        assert torch.equal(camera.body.velocity, pong.game.body.velocity)
        actual_ticks.append(a.physics_ticks)
    assert actual_ticks[:5] == [2, 2, 3, 2, 3]


def test_unmodified_frozen_tracker_reads_raw_paddle_ball_and_bricks():
    camera = BreakoutCamera(17)
    tracker, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    previous, current = torch.zeros_like(camera.frame()), camera.frame()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(8):
            events = torch.stack((previous & ~current, ~previous & current)).float()
            entities = encoder.encode_entities(*tracker.track_entities(events, current.float()))
            previous, current = current, camera.step(2).frame
    finally:
        torch.set_num_threads(threads)
    observed = [entity for entity in entities if entity['observed']]
    positions = [entity['normalized_position'].tolist() for entity in observed]
    assert sum(y < -.18 for y, _ in positions) == len(camera.layout.cells)
    assert sum(y > .40 for y, _ in positions) == 1
    assert sum(-.14 < y < .30 for y, _ in positions) == 1


def test_balanced_motion_required_and_reachable_at_native_speed():
    schedule = breakout_schedule(712, 12)
    assert sum(episode.serve_side for episode in schedule) == 0
    for episode in schedule:
        passive = episode.camera(max_episode_ticks=240)
        run_to_end(passive)
        first = passive.summary()['first_incoming']
        assert not first['success']
        # Minimum and maximum landing distances need different braking times.
        # A legal bang-then-neutral trajectory reaches the strip on every tested start.
        controlled = episode.camera(max_episode_ticks=240)
        while not controlled.contacts:
            error = (first['ball_x'] - float(controlled.body.position[0])) * episode.serve_side
            action = (0 if episode.serve_side > 0 else 2) if error > .09 else 1
            controlled.step(action)
        matched = controlled.summary()['first_incoming']
        assert matched['success']
        assert matched['tick'] == first['tick']
        assert matched['ball'] == first['ball']
        assert matched['velocity'] == first['velocity']
        assert passive.summary()['life_lost'] and passive.summary()['own_opportunities'] == 1
        assert passive.summary()['serve_bricks'] == 0
        assert controlled.summary()['score'] == 0


def test_reward_for_last_brick_and_natural_serve_counted_separately():
    layout = BreakoutLayout(((2, 3),))
    x0, x1, y0, y1 = brick_bounds(2, 3)
    x = .5 * (x0 + x1)
    camera = BreakoutCamera(layout=layout, serve=BreakoutServe(x, .36, 0., -.55), max_episode_ticks=120)
    before = camera.frame()
    ending = run_to_end(camera)[-1]
    region = (slice(round(y0 * 64), round(y1 * 64)), slice(round(x0 * 64), round(x1 * 64)))
    assert before[region].all() and not ending.frame[region].any()
    assert ending.terminated and not ending.truncated and ending.reward == 1.
    summary = camera.summary()
    assert summary['cleared'] and not summary['life_lost']
    assert summary['serve_bricks'] == summary['score'] == summary['initial_bricks'] == 1
    assert summary['post_own_bricks'] == summary['remaining_bricks'] == summary['own_returns'] == 0
    with pytest.raises(ValueError, match='reset required'):
        camera.step(1)
    assert camera.reset()[region].all()


def test_passable_column_gap_and_real_ceiling_bounce():
    layout = BreakoutLayout(((0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)))
    x = (brick_bounds(2, 3)[1] + brick_bounds(2, 4)[0]) / 2
    camera = BreakoutCamera(layout=layout, serve=BreakoutServe(x, .36, 0., -.55), max_episode_ticks=300)
    while camera.velocity[1] < 0:
        assert not camera.step(1).reward
    assert camera.config.ball_radius <= camera.ball[1] < .04
    run_to_end(camera)
    summary = camera.summary()
    assert summary['own_returns'] == 1 and summary['score'] == 0
    assert summary['truncated'] and not summary['life_lost']


def test_row_gap_and_swept_brick_side_collision():
    # A fast explicit launch in the inter-row gap stays clear of both rows.
    layout = BreakoutLayout(((0, 3), (1, 3), (0, 6)))
    y = (brick_bounds(0, 3)[3] + brick_bounds(1, 3)[2]) / 2
    gap = BreakoutCamera(layout=layout, serve=BreakoutServe(.37, y, 20., .0001), max_episode_ticks=2)
    assert gap.step(1).reward == 0 and gap.ball[0] > .68
    _, _, y0, y1 = brick_bounds(0, 3)
    side = BreakoutCamera(layout=layout, serve=BreakoutServe(.37, (y0 + y1) / 2, 20., .0001),
                          max_episode_ticks=2)
    assert side.step(1).reward == 1 and side.velocity[0] < 0


def test_edges_reflect_and_paddle_offset_can_score():
    wall = BreakoutCamera(serve=BreakoutServe(.985, .5, .3, .4), max_episode_ticks=2)
    wall.step(1)
    assert wall.velocity == [-.3, .4]
    assert wall.ball[0] == pytest.approx(.9875 - (.985 + .3 * 2 / 120 - .9875))
    camera = BreakoutCamera(layout=BreakoutLayout(((2, 3),)),
                            serve=BreakoutServe(.49, .79, 0., .55), max_episode_ticks=240)
    run_to_end(camera)
    summary = camera.summary()
    assert summary['cleared'] and summary['own_returns'] == 1
    assert summary['post_own_bricks'] == 1 and summary['serve_bricks'] == 0


def test_schedules_have_independent_seeds_distinct_layouts_and_static_neutral():
    practice = breakout_schedule(703, 8, split='practice')
    heldout = breakout_schedule(703, 8, split='heldout')
    assert len({episode.seed for episode in practice + heldout}) == 16
    assert len({episode.layout_seed for episode in practice + heldout}) == 16
    assert set(e.camera().layout.cells for e in practice).isdisjoint(e.camera().layout.cells for e in heldout)
    assert heldout == breakout_schedule(703, 8, split='heldout')
    first = play_breakout(None, 51, serve_side=1)
    repeated = play_breakout(None, 51, serve_side=1)
    remapped = play_breakout(None, 51, serve_side=1, mapping=(2, 0, 1), delay_frames=4)
    assert first == repeated
    assert first['first_incoming'] == remapped['first_incoming']
    assert first['action_sha256'] != remapped['action_sha256']


def test_tick_cap_preserves_completed_miss_and_unfinished_approach():
    camera = BreakoutCamera(14, serve_side=-1, max_episode_ticks=1)
    run_to_end(camera)
    assert camera.summary()['unfinished_incoming_approach']
    assert not camera.summary()['own_opportunities']
    miss = play_breakout(None, 14, serve_side=-1)['first_incoming']
    capped = BreakoutCamera(14, serve_side=-1, max_episode_ticks=miss['tick'])
    run_to_end(capped)
    assert capped.summary()['first_incoming'] == miss
    assert capped.summary()['truncated'] and not capped.summary()['life_lost']
    assert not capped.summary()['unfinished_incoming_approach']
