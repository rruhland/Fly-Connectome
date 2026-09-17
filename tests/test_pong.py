import torch

from fly_connectome.pong import Pong, Body, Physics, Curriculum


def test_force_accelerates_brakes_reverses_and_decays():
    body = Body(1, Physics())
    for _ in range(20):
        body.step(torch.tensor([1.]))
    assert body.velocity.item() < 0
    old = body.velocity.item()
    for _ in range(100):
        body.step(torch.tensor([-1.]))
    assert body.velocity.item() > old
    assert body.velocity.item() > 0
    for _ in range(500):
        body.step(torch.zeros(1))
    assert abs(body.velocity.item()) < .001


def test_seeded_physics_and_reset_independence():
    a, b = Pong([3, 7]), Pong([3, 7])
    a.reset(torch.tensor([True, False]))
    for _ in range(100):
        a.step(torch.zeros(2))
        b.step(torch.zeros(2))
    torch.testing.assert_close(a.ball[1], b.ball[1], rtol=0, atol=0)
    assert torch.equal(a.render()[1], b.render()[1])


def test_player_hit_and_score_are_separate_and_reset_body():
    game = Pong([1])
    game.ball[0] = torch.tensor([.065, .5])
    game.ball_velocity[0] = torch.tensor([-.5, 0.])
    result = game.step(torch.zeros(1))
    assert result.hits.item() == 1
    assert result.scores.item() == 0
    assert game.ball_velocity[0, 0] > 0
    game.ball[0] = torch.tensor([.001, .05])
    game.ball_velocity[0] = torch.tensor([-.5, 0.])
    game.body.activation.fill_(1.)
    result = game.step(torch.zeros(1))
    assert result.scores.item() == -1
    assert result.hits.item() == 0
    assert game.body.activation.item() == 0
    assert game.ball[0, 0].item() == .5


def test_curriculum_fades_to_score_only_without_duplicate_penalty():
    curriculum = Curriculum(bootstrap=10, fade=10, hit_scale=.25, score_scale=1.)
    assert [curriculum.alpha(k) for k in (0, 10, 15, 20, 30)] == [.25, .25, .125, 0., 0.]
    assert curriculum.reward(0, torch.tensor([-1.]), torch.tensor([0.])).item() == -1
    assert curriculum.reward(20, torch.tensor([0.]), torch.tensor([1.])).item() == 0


def test_auto_reset_produces_visible_events():
    from fly_connectome.sensor import EventCamera
    game = Pong([1])
    camera = EventCamera(1, 32, 64)
    game.ball[0] = torch.tensor([.001, .05])
    game.ball_velocity[0] = torch.tensor([-.5, 0.])
    camera.observe(game.render())
    game.step(torch.zeros(1))
    events = camera.observe(game.render())
    assert events.on.any() and (~events.on).any()
