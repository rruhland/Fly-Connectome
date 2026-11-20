"""Measure reflected puck direction from timestamped public motion."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from hockey_shot_transfer import HockeyShotObserver, ShotContact, fit_shot_response, predict_shot_angle


def test_observer_resolves_a_reversal_at_the_visible_striker():
    observer = HockeyShotObserver()
    hits = []
    for frame in range(40):
        t = frame / 60 - .30
        puck = (.310 - .70 * t, .392 + .03 * t) if t < 0 else (.310 + .95 * t, .392 + .30 * t)
        hit = observer.observe(SimpleNamespace(puck=puck, player=(.245, .375)))
        if hit is not None:
            hits.append(hit)
    assert len(hits) == 1
    assert hits[0].seconds == pytest.approx(.30)
    assert hits[0].offset == pytest.approx(.017)
    assert hits[0].separation == pytest.approx(math.hypot(.065, .017))
    assert hits[0].outgoing_angle == pytest.approx(math.atan2(.30, .95))


def test_observer_ignores_a_reversal_away_from_the_striker():
    observer = HockeyShotObserver()
    for frame in range(40):
        t = frame / 60 - .30
        puck = (.110 + .70 * abs(t), .392 + .03 * t)
        assert observer.observe(SimpleNamespace(puck=puck, player=(.245, .375))) is None


def test_fit_identifies_lateral_offset_separately_from_incoming_and_striker_motion():
    reach = .065
    expected = (1.2, -.4, .7)
    shots = []
    for offset in [-.025, .025]:
        for incoming in [-.3, .3]:
            for striker in [-.5, .5]:
                shot = ShotContact(0., offset, reach, incoming, 0., striker, 1.)
                outgoing = predict_shot_angle(shot, reach, expected)
                shots.append(ShotContact(0., offset, reach, incoming, outgoing, striker, 1.))
    assert fit_shot_response(shots, reach) == pytest.approx(expected)
