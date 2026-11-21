"""Check isolation of continuation credit and no-op timing defaults."""

import numpy as np

from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt19_model_rollout import RolloutPlayer
from scripts.experimental.tt21_continuation import ContinuationPlayer, learn_fallback, learn_phase, phase_estimate
from scripts.experimental.tt21_continuation_trial import first_reply, measure_credit


def test_phase_estimate_uses_the_public_flight_model():
    state = np.array((.02, -.37, .18)), np.array((0., -.056, .028))
    law = FlightLaw()
    ahead = phase_estimate(law, state, 1)
    path, speed = law.trajectory(*state, 1)
    np.testing.assert_array_equal(ahead, (path[-1], speed[-1]))
    assert phase_estimate(law, state, 0) is state
    assert phase_estimate(law, state, -1)[0][1] > state[0][1]


def test_credit_excludes_opening_and_requires_a_referee_legal_reply():
    events = [dict(kind='contact', side=1, step=10), dict(kind='legal_return', side=1, step=22),
              dict(kind='contact', side=-1, step=26), dict(kind='legal_return', side=-1, step=44),
              dict(kind='contact', side=1, step=48)]
    assert first_reply(events[:3]) is None
    step = first_reply(events)
    assert step == 26
    assert measure_credit(events, {'points_for': 0}, step) == dict(legal=False, contact=True, win=False)


def test_phase_selection_prefers_legal_landings_and_keeps_baseline_on_ties():
    def row(phase, *, legal=False, contact=False, win=False):
        return dict(phase=phase, credit=dict(legal=legal, contact=contact, win=win))

    rows = [row(0, contact=True), row(1, legal=True, contact=True), row(-1, contact=True, win=True)]
    assert learn_phase(rows)[0] == 1
    assert learn_phase([row(0), row(1), row(-1)])[0] == 0


def test_fallback_uses_credit_only_on_replies_without_nominal_legal_plans():
    def row(seed, phase, *, legal_plan=False, contact=False, legal=False):
        return dict(seed=seed, phase=phase, first_rival_reply=41,
                    plans=[dict(tick=53, legal=legal_plan)],
                    credit=dict(legal=legal, contact=contact, win=False))

    plans = [row(1984, 0), row(1984, -1, contact=True), row(1984, 1),
             row(617, 0, legal_plan=True, contact=True, legal=True), row(617, -1), row(617, 1)]
    rule, evidence = learn_fallback(plans)
    assert rule == 'retry with earlier estimate'
    assert evidence['starts'] == [1984]


def test_guard_keeps_a_nominal_legal_plan_without_replanning(monkeypatch):
    calls = []

    def nominal(_player, estimate):
        calls.append(estimate)
        return dict(legal=True, index=72)

    monkeypatch.setattr(RolloutPlayer, 'choose', nominal)
    player = ContinuationPlayer.__new__(ContinuationPlayer)
    player.reset()
    player.history.append((37, None))
    player.followup = True
    player.phase = 'retry with earlier estimate'
    estimate = np.array((.02, -.37, .18)), np.array((0., -.056, .028))
    assert player.choose(estimate) == dict(legal=True, index=72)
    assert len(calls) == 1 and calls[0] is estimate
