"""Check MuJoCo service contacts and the service order used for scoring."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt4_arena import TABLE_HALF_LENGTH, TABLE_HEIGHT
from scripts.experimental.tt6_body_arena import TT6Arena
from scripts.experimental.tt8_service_arena import MIN_TOSS_RISE, ServiceScore, TT8ServiceArena


@pytest.mark.parametrize('body', ('stock', 'rotary', 'gantry', 'offset'))
def test_toss_rises_behind_the_end_line_and_idle_blade_faults(body):
    arena = TT8ServiceArena(body=body)
    arena.reset(seed=3, server=1, render=False)
    bottom = arena.ball[2]
    apex = bottom
    assert bottom > TABLE_HEIGHT
    assert arena.velocity[2] > 0.
    assert arena.events[0]['kind'] == 'serve_toss'
    assert arena.events[0]['step'] == 0
    for _ in range(60):
        _, reward, done, capped, result = arena.step(np.zeros(arena.action_size), render=False)
        apex = max(apex, arena.ball[2])
        assert arena.ball[1] < -TABLE_HALF_LENGTH
        if done or capped:
            break
    assert apex - bottom >= MIN_TOSS_RISE
    assert reward == -1 and done and not capped
    assert result['reason'] == 'serve_missed_toss'
    assert result['contacts'] == 0
    assert not result['legal_serve']


@pytest.mark.parametrize('wrist, let', [(-.465517, False), (.103448, True)])
def test_learner_racket_generates_two_physical_bounces(wrist, let):
    arena = TT8ServiceArena(body='gantry')
    arena.reset(seed=10, server=1, render=False)
    # Recorded normalized rates for a backswing, pause and forward stroke.
    # The more open blade clips the net and still lands on the rival's half.
    hold = (0., 0., 0., 0.)
    actions = [(1., -1., 1., wrist), (.334553, -1., 1., 0.),
               (0., -1., .235294, 0.), (0., -1., 0., 0.), (0., -.105685, 0., 0.)]
    actions += [hold] * 6 + [(0., 1., 0., 0.)] * 5 + [hold] * 45
    for action in actions:
        _, reward, done, capped, result = arena.step(action, render=False)
        if result['legal_serve'] or done or capped:
            break
    service = [event for event in arena.events if event['kind'] in ('serve_strike', 'bounce')]
    assert [(event['kind'], event['side']) for event in service] == [
        ('serve_strike', 1), ('bounce', 1), ('bounce', -1)]
    assert service[0]['incoming'][2] < 0.
    assert service[0]['location'][1] < -TABLE_HALF_LENGTH
    assert service[1]['location'][1] < 0 < service[2]['location'][1]
    assert result['service_contact'] and result['contacts'] == 1
    assert result['legal_returns'] == 0
    if let:
        assert result['serve_let'] and not result['legal_serve']
        assert result['reason'] == 'serve_let'
        assert any(event['kind'] == 'net' for event in arena.events)
        assert capped and not done and reward == 0
        score = ServiceScore()
        assert score.record(result) == (0, 0)
        assert score.server == 1
    else:
        assert result['legal_serve'] and not result['serve_let']
        assert any(event['kind'] == 'legal_serve' for event in arena.events)
        assert not done and not capped and reward == 0


def test_seeded_rival_service_preserves_receiving_physics_after_learner_service():
    old = TT6Arena(body='rotary')
    arena = TT8ServiceArena(body='rotary')
    arena.reset(seed=93, server=1, render=False)
    arena.step(np.zeros(6), render=False)
    old.reset(seed=10, render=False)
    arena.reset(seed=10, server=-1, render=False)
    rng = np.random.default_rng(13)
    for _ in range(60):
        np.testing.assert_array_equal(arena.data.qpos, old.data.qpos)
        np.testing.assert_array_equal(arena.data.qvel, old.data.qvel)
        action = rng.uniform(-1., 1., 6)
        current = arena.step(action, render=False)
        previous = old.step(action, render=False)
        assert current[1:4] == previous[1:4]
        if any(current[2:4]):
            break
    assert arena.legal_serve
    assert [event for event in arena.events if event['kind'] != 'legal_serve'] == old.events


def test_two_point_turns_one_point_after_deuce_and_win_by_two():
    game = ServiceScore(first_server=-1)

    def point(winner):
        return game.record(dict(server=game.server, points_for=int(winner == 1),
                                points_against=int(winner == -1), serve_let=False, reason='floor'))

    for index, server in enumerate([-1, -1, 1, 1] * 5):
        assert game.server == server
        point(1 if index % 2 else -1)
    assert game.score == (10, 10) and game.server == -1
    assert point(1) == (11, 10) and game.server == 1 and not game.finished
    assert point(-1) == (11, 11) and game.server == -1 and not game.finished
    assert point(1) == (12, 11) and game.server == 1 and not game.finished
    assert point(1) == (13, 11) and game.finished and game.winner == 1
    with pytest.raises(RuntimeError, match='finished'):
        point(1)
