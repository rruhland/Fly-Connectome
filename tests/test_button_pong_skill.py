"""Exercise the public image boundary and independently acquired Pong skill."""

import json
from pathlib import Path
import sys

import pytest

from fly_connectome.pong import Pong

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from button_pong_skill import CAMERA_SIZE, DesiredPongPolicy, NaivePongPolicy, PongScreen, native_points


def view_flight(policy, seed, *, rising=False):
    policy.begin(seed)
    directions = []
    for index in range(27):
        ball = (.53 - .016 * index, .62 - .012 * index if rising else .38 + .012 * index)
        directions.append(policy.action(PongScreen(.5, .5, ball), 4 * index))
    policy.finish(0)
    return directions


def test_native_binary_raster_decodes_visible_paddles_and_ball():
    game = Pong([7123])
    game.ball[0] = game.ball.new_tensor([.29, .65])
    game.body.position[0] = .36
    game.opponent[0] = .76
    screen = PongScreen.read(game.render(CAMERA_SIZE, CAMERA_SIZE)[0])
    resolution = 1 / CAMERA_SIZE
    assert screen.paddle_y == pytest.approx(.36, abs=resolution)
    assert screen.opponent_y == pytest.approx(.76, abs=resolution)
    assert screen.ball == pytest.approx((.29, .65), abs=resolution)
    game.ball[0, 0] = .05
    assert PongScreen.read(game.render(CAMERA_SIZE, CAMERA_SIZE)[0]).ball is None


def test_learned_crossings_generalize_to_mirrored_public_trajectories(tmp_path):
    policy = DesiredPongPolicy(aim=False)
    for seed in range(40):
        view_flight(policy, seed)
    assert policy.crossings.count == 40
    assert policy.crossings.lead == 1.
    checkpoint = tmp_path / 'prior.json'
    checkpoint.write_text(json.dumps(policy.to_state()))
    held_out = DesiredPongPolicy.load(checkpoint, learn=False, aim=False)
    cold = NaivePongPolicy()
    view_flight(held_out, 100, rising=True)
    view_flight(cold, 100, rising=True)
    assert max(held_out.crossing_errors) < .001
    assert min(cold.crossing_errors) > .10
    assert held_out.to_state() == policy.to_state()
    assert cold.crossings.count == 0
    resumed = DesiredPongPolicy.load(checkpoint)
    view_flight(resumed, 101, rising=True)
    assert resumed.crossings.count == 41


def test_same_policy_class_with_or_without_prior_can_learn_scoring():
    policy = DesiredPongPolicy(exploration=0.)
    for seed in range(40):
        view_flight(policy, seed)
    before = policy.scores.visits.sum()
    policy.begin(40)
    for index in range(18):
        screen = PongScreen(.5, .5, (.48 - .016 * index, .43 + .003 * index))
        action = policy.action(screen, 4 * index)
        assert policy.action(screen, 4 * index) == action
    choices = policy.choices.copy()
    policy.finish(1)
    assert len(choices) == 1
    assert policy.scores.visits.sum() == before + 1
    assert policy.scores.totals[choices[0][0]].sum() > 0.
    with pytest.raises(ValueError):
        policy.finish(1)


def test_native_short_timeout_finishes_a_policy_once():
    policy = DesiredPongPolicy()
    assert native_points([1091], [policy], max_ticks=4) == [
        dict(seed=1091, score=0, own_hit_win=0, hits=0, opportunities=0, first_hit=0, ticks=4,
             camera_crossing_errors=[], choices=[])]
    assert not policy.active
