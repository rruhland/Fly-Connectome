"""Wall response must be measured from visible flights and stay optional in Pong."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arcade_hockey_pong import AdaptiveWall
from arcade_wall_learn import WallFlight, WallMemory, project_crossing
from fly_connectome.pong import Physics, Pong


@pytest.mark.parametrize('coefficient', [.35, .58, 1.])
def test_public_flight_recovers_one_bounce_from_quantized_positions(coefficient):
    wall = WallMemory()
    flight = WallFlight(wall, (.02, .98), position_noise=.0025)
    for frame in range(50):
        x = .65 - .01 * frame
        y = .02 + (.018 * (22 - frame) if frame <= 22 else .018 * coefficient * (frame - 22))
        flight.observe((round(x * 128) / 128, round(y * 128) / 128))
    assert wall.count == 1
    assert wall.coefficient == pytest.approx(coefficient, abs=.04)
    assert WallMemory.from_state(wall.to_state()).to_state() == wall.to_state()


def test_gap_and_longitudinal_reversal_cannot_create_wall_measurement():
    for interrupt in ('gap', 'paddle'):
        wall = WallMemory()
        flight = WallFlight(wall, (.02, .98))
        for frame in range(50):
            x = .65 - .01 * (frame if interrupt == 'gap' else 22 - abs(frame - 22))
            y = .02 + .018 * abs(22 - frame)
            flight.observe(None if interrupt == 'gap' and frame == 22 else (x, y))
        assert wall.count == 0


def test_project_crossing_applies_wall_response_to_remaining_flight():
    position, velocity, bounds = (.75, .20), (-.50, -.40), (.02, .98)
    assert project_crossing(position, velocity, .15, bounds, .60) == pytest.approx(.20)
    assert project_crossing(position, velocity, .15, bounds, 1.) == pytest.approx(.32)
    assert project_crossing(position, velocity, .90, bounds, .60) is None


def test_inherited_alternatives_are_dormant_until_target_evidence():
    anchor = WallMemory()
    source = WallMemory()
    for _ in range(4):
        anchor.observe(1., .06)
        source.observe(.64, .06)
    fresh = AdaptiveWall(anchor)
    inherited = AdaptiveWall(anchor, (source,))
    frozen = AdaptiveWall(anchor, (source,), 'frozen')
    for model in (fresh, inherited, frozen):
        assert model.coefficient == 1.
        model.observe(.98, .09)
        assert model.coefficient == 1.
    for model in (fresh, inherited, frozen):
        model.observe(.43, .13)
    assert fresh.changed and inherited.changed
    assert frozen.coefficient == 1.
    initial_offset = abs(inherited.coefficient - .46)
    for _ in range(4):
        inherited.observe(.46, .05)
    assert abs(inherited.coefficient - .46) < initial_offset


def test_native_pong_wall_response_preserves_default_and_horizontal_motion():
    elastic = Pong([18], Physics())
    explicit = Pong([18], Physics(wall_restitution=1.))
    damped = Pong([18], Physics(wall_restitution=.5))
    for game in (elastic, explicit, damped):
        game.ball[0] = torch.tensor([.5, .013])
        game.ball_velocity[0] = torch.tensor([-.4, -.5])
        game.step(torch.zeros(1))
    torch.testing.assert_close(elastic.ball, explicit.ball, rtol=0., atol=0.)
    torch.testing.assert_close(elastic.ball_velocity, explicit.ball_velocity, rtol=0., atol=0.)
    torch.testing.assert_close(elastic.ball[:, 0], damped.ball[:, 0], rtol=0., atol=0.)
    assert float(damped.ball_velocity[0, 1]) == .25
    assert float(damped.ball[0, 1] - .0125) == pytest.approx(float(elastic.ball[0, 1] - .0125) * .5)
