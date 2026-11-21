"""Check narrow launch sampling and image-conditioned stroke recall."""

from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt21_receive_lesson import StrikeMemory, compare_proposals, guard_prefers_recall, strike_cue
from scripts.experimental.tt21_receive_lesson_trial import CenteredDraws, LIMITED_WIDTH


def test_centered_launch_keeps_the_original_seeded_draw_and_scales_its_deviation():
    source = np.random.default_rng(91)
    reference = np.random.default_rng(91)
    lo, hi = np.array((-.05, 2.58, -20.)), np.array((.05, 2.88, 20.))
    mid = .5 * (lo + hi)
    actual = CenteredDraws(source).uniform(lo, hi)
    np.testing.assert_allclose(actual, mid + LIMITED_WIDTH * (reference.uniform(lo, hi) - mid))
    assert source.uniform(0., 1.) == reference.uniform(0., 1.)


def test_recall_requires_positive_optical_credit_and_uses_new_observed_timing():
    memory = StrikeMemory()
    cue = dict(when=25.4, features=[.024, .141, -.053, .034])
    stroke = dict(start=15, cue=cue, rates=[[0., 0., .2, 0., 0.]])
    assert memory.retrieve(cue, 13) is None
    memory.observe(stroke, .20, 0)
    assert memory.retrieve(cue, 13) is None
    memory.observe(stroke, 1.45, 3)
    after = dict(cue, when=27.4)
    recalled = memory.retrieve(after, 14)
    assert recalled['start'] == 17 and recalled['recalled'] == 3
    assert recalled['rates'] == stroke['rates']
    assert memory.retrieve(dict(after, features=[.27, .141, -.053, .034]), 14) is None


def test_cue_uses_the_public_flight_clock_and_finite_reachable_crossing():
    law = FlightLaw()
    origin = np.array((0., -.43, .2)), np.array((0., -.052, .01))
    first = strike_cue(law, origin, 10)
    assert 18 < first['when'] < 23
    assert strike_cue(law, origin, 12)['when'] == first['when'] + 2
    assert strike_cue(law, (origin[0], -origin[1]), 10) is None


def test_guard_requires_relative_legal_and_cost_advantage():
    assert guard_prefers_recall((1, 2), (2.4, 1.6))
    assert not guard_prefers_recall((0, 1), (2.4, .8))
    assert not guard_prefers_recall((2, 2), (2.4, .8))
    assert not guard_prefers_recall((1, 2), (1.81, 1.60))


def test_guard_scores_the_scheduled_wait_under_shared_conditions():
    def predict(tapes, durations, _ball, offsets):
        np.testing.assert_array_equal(offsets[:3], offsets[3:])
        np.testing.assert_array_equal(durations, [1, 1, 1, 3, 3, 3])
        np.testing.assert_array_equal(tapes[3, :2], 0.)
        np.testing.assert_array_equal(tapes[3, 2], np.full(5, .4))
        return [dict(legal=legal, cost=cost) for legal, cost in
                ((True, .05), (False, 3.), (False, 3.), (True, .1), (True, .2), (False, 3.))]

    position = np.array((.02, -.37, .18))
    dry = SimpleNamespace(history=[(11, position)], law=FlightLaw(), sim=SimpleNamespace(predict=predict))
    stock = dict(start=11, rates=[[.2] * 5])
    memory = dict(start=13, rates=[[.4] * 5])
    chosen, audit = compare_proposals(dry, (position, np.array((0., -.056, .028))), 11, stock, memory)
    assert chosen is memory and audit['kept_recall']
