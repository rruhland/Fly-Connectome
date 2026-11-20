"""Check the simulator and its rendered agent boundary on deterministic rounds."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from air_hockey import AirHockeyCamera, HockeyServe, GOAL_COLOR, HEIGHT, PUCK_RADIUS, WIDTH
from air_hockey_experiment import paired, run_round, schedule
from air_hockey_learning import GoalMemory, HockeyPlayer, read_camera
from arcade_wall_learn import WallMemory


def test_two_axis_requests_move_the_mallet_and_leave_pixels_as_the_observation():
    camera = AirHockeyCamera(3, serve_style='neutral', opponent='still')
    first = camera.frame()
    initial = read_camera(first)
    frame, reward, terminated, truncated = camera.step(8)
    following = read_camera(frame)
    assert first.shape == (96, 128, 3)
    assert first.dtype == np.uint8
    assert np.any(np.all(first == GOAL_COLOR, axis=-1))
    assert following.player[0] > initial.player[0]
    assert following.player[1] > initial.player[1]
    assert following.rival == initial.rival and following.puck == initial.puck
    assert not (reward or terminated or truncated)


def test_off_center_disc_collision_redirects_on_both_axes():
    center = HEIGHT / 2
    camera = AirHockeyCamera(2, serve=HockeyServe(.42, center + .030, -.67, 0.), opponent='still')
    for _ in range(24):
        camera.step(0)
    hit = next(event for event in camera.events if event['kind'] == 'contact')
    assert hit['side'] == 'player'
    assert hit['incoming'] == (-.67, 0.)
    assert hit['velocity'][0] > .25
    assert hit['velocity'][1] > .3


def test_side_rail_damps_only_lateral_speed_and_goal_opening_ends_play():
    camera = AirHockeyCamera(2, serve=HockeyServe(.6, PUCK_RADIUS + .002, .4, -.65),
                            opponent='still', wall_restitution=.45)
    _, reward, terminated, truncated = camera.step(0)
    assert not (reward or terminated or truncated)
    assert camera.events[0]['kind'] == 'rail'
    assert camera.events[0]['velocity'] == pytest.approx((.4, .65 * .45))
    score = AirHockeyCamera(2, serve=HockeyServe(WIDTH - PUCK_RADIUS - .004, HEIGHT / 2, .7, 0.),
                           opponent='still')
    _, reward, terminated, truncated = score.step(0)
    assert (reward, terminated, truncated) == (1, True, False)
    with pytest.raises(RuntimeError):
        score.step(0)
    score.reset()
    assert not score.terminated and not score.events


def test_off_goal_end_rail_returns_puck_and_time_limit_is_public():
    camera = AirHockeyCamera(2, serve=HockeyServe(WIDTH - PUCK_RADIUS - .004, .14, .7, 0.),
                            opponent='still', max_frames=1)
    _, reward, terminated, truncated = camera.step(0)
    assert (reward, terminated, truncated) == (0, False, True)
    assert camera.events[0]['kind'] == 'end_rail'
    assert camera.events[0]['velocity'][0] < 0.


@pytest.mark.parametrize('restitution', [1., .45, .05])
def test_fast_banks_fit_the_speed_cap(restitution):
    for seed in range(100):
        camera = AirHockeyCamera(seed, serve_style='fast-bank', wall_restitution=restitution)
        assert camera.velocity[0] < 0.


def test_repeatable_frozen_replay_and_separate_practice_schedule():
    wall = WallMemory(default=.91)
    goals = GoalMemory()
    initial_goal_state = goals.to_state()
    case = {'seed': 638313, 'style': 'neutral', 'opponent': 'laneguard'}
    row, frames = run_round(case, wall, goals, record=True, max_frames=100)
    replay, _ = run_round(case, WallMemory.from_state(wall.to_state()),
                          GoalMemory.from_state(initial_goal_state), max_frames=100)
    assert replay == row
    assert len(frames) == row['frames'] + 1
    assert row['contacts'] > 0
    assert goals.to_state() == initial_goal_state
    assert wall.count == 0
    starts = schedule(1, 4, 3, 2, 'guard')
    numbers = starts['pong'] + [case['seed'] for case in starts['hockey_practice'] + starts['hockey_heldout']]
    assert len(numbers) == len(set(numbers))
    assert paired([row], [replay], 'last_touch_goal')['mean'] == 0.


def test_goal_memory_uses_a_public_reward_when_the_visual_contact_is_present():
    goals = GoalMemory()
    player = HockeyPlayer(WallMemory(default=1.), goals, learn_goals=True)
    camera = AirHockeyCamera(63, opponent='still', serve_style='neutral')
    player.observe(camera.frame())
    for _ in range(80):
        player.observe(*camera.step(player.action()))
        if player.last_contact is not None or camera.terminated:
            break
    assert player.last_contact is not None
    shot = player.last_contact
    player.observe(camera.frame(), reward=1, terminated=True)
    assert goals.evidence[shot.context, shot.arm] == 1
    assert goals.goals[shot.context, shot.arm] == 1
