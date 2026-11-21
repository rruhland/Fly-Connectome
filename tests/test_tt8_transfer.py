import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt8_transfer_model import bank_predictions, optical_prior
from scripts.experimental.tt8_transfer_rebound import DT, ReboundCourt, pixels
from scripts.experimental.tt8_transfer_switch import controllers
from scripts.experimental.tt8_transfer_trial import play, player_group, verify


def test_source_prior_uses_only_optical_ratios():
    def point(phase, ratios):
        return dict(phase=phase, table_bounce=dict(samples=[dict(ratio=ratio) for ratio in ratios]),
                    events=[dict(ratio=.02)])

    receipt = dict(visits=[dict(points=[
        point('low', [.64, .66]), point('low', [.62]), point('low', [.70, .71]),
        point('lively', [.94]), point('low', []),
    ])])
    prior = optical_prior(receipt, 'low')
    assert prior['center'] == pytest.approx(.65)
    assert prior['points'] == 3
    assert prior['bounces'] == 5


def test_physical_surface_and_screen():
    lively, low = ReboundCourt('lively'), ReboundCourt('low')
    high, muted = lively.opening(71601), low.opening(71601)
    np.testing.assert_array_equal(high.positions[:8], muted.positions[:8])
    assert high.cart_start == muted.cart_start
    assert high.crossing_time - muted.crossing_time > .10
    shot = high.positions[2]
    ball, cart = pixels(lively.frame(shot, .21))
    np.testing.assert_allclose(ball, shot, atol=.002)
    assert cart == pytest.approx(.21, abs=.5 / 78.)
    hidden, still_visible = pixels(lively.frame(None, .21))
    assert hidden is None
    assert still_visible == cart
    assert high.crossing_time > (len(high.positions) - 2) * DT
    assert high.crossing_time <= (len(high.positions) - 1) * DT


def test_restitution_forecast_and_physical_scoring():
    targets = bank_predictions(np.array([-.4, .42]), np.array([2.1, -3.]))
    assert np.all(np.diff(targets) >= 0.)
    assert targets[-1] - targets[0] > .25
    np.testing.assert_allclose(targets, -bank_predictions([.4, .42], [-2.1, -3.]))
    court = ReboundCourt('lively')
    opening = court.opening(71601)
    donors = dict(lively=dict(center=.92, spread=.055), low=dict(center=.65, spread=.055))
    player = player_group(donors, 'lively')['matched']
    result = play(court, opening, player, visibility='curtain')
    assert len(player.samples) == 8
    assert result['final_ratio'] != result['initial_ratio']
    row = dict(material='lively', fall=71601, crossing_m=opening.endpoint,
               time_s=opening.crossing_time, players=dict(matched=result))
    assert verify(dict(rows=[row])) == 1


def test_blind_source_mixture_can_only_distinguish_plates_after_public_feedback():
    donors = dict(lively=dict(center=.92, spread=.055), low=dict(center=.65, spread=.055))
    plays = []
    for name in donors:
        court = ReboundCourt(name)
        player = controllers(donors)['source_mixture']
        plays.append(play(court, court.opening(71601), player, visibility='curtain'))
    lively, low = plays
    count = min(len(lively['commands']), len(low['commands']))
    assert lively['commands'][:count] == low['commands'][:count]
    assert lively['final_ratio'] > low['final_ratio']
