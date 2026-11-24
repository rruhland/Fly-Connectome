"""Fast checks for local online reward memory and purely measured credit."""

import numpy as np

from scripts.experimental.tt36_online import Actor, MAX_POINTS_IN_MEMORY, Memory, labels, outcome
from scripts.experimental.tt36_online_components import transform


def test_memory_is_local_and_uses_miss_magnitude():
    left = np.zeros(11)
    right = left.copy()
    right[0] = 2.
    state = Memory()
    assert state.greedy(left) == 0
    state.learn([dict(context=left, arm=1, reward=outcome(False, None, .005)),
                 dict(context=left, arm=0, reward=outcome(False, None, .18)),
                 dict(context=right, arm=3, reward=outcome(True, None, None))])
    assert state.greedy(left) == 1
    assert state.greedy(right) == 3
    state.learn([dict(context=left, arm=0, reward=outcome(True, None, None))])
    assert state.greedy(left) == 0


def test_missing_evidence_not_equated_to_failure():
    state = Memory()
    z = np.zeros(11)
    state.learn([dict(context=z, arm=1, reward=None)])
    assert state.bank == []
    state.learn([dict(context=z, arm=1, reward=.5)] * (MAX_POINTS_IN_MEMORY + 2))
    assert len(state.bank) == MAX_POINTS_IN_MEMORY


def test_outcome_credit_uses_two_actual_frames_and_only_earned_pulse():
    x = np.zeros(11).tolist()
    choices = [dict(tick=12, context=x, arm=1), dict(tick=30, context=x, arm=3)]
    paddle = np.array([0., -1.3, 1.0])
    stereo = [(16, np.array([.04, -1.26, 1.0]), paddle),
              (18, np.array([.04, -1.34, 1.0]), paddle)]
    events = [dict(kind="legal_return", side=1, time=.05),
              dict(kind="legal_return", side=-1, time=.20),
              dict(kind="legal_return", side=1, time=.80)]
    first, second = labels(choices, stereo, [(34, 3.2)], events, 60, .02)
    assert not first["landed"]
    assert np.isclose(first["optical_gap_m"], .04)
    assert first["optical_passage"]["frames"] == [16, 18]
    assert second["landed"]
    assert second["reward"] == 1.
    assert second["optical_outgoing_mps"] == 3.2


def test_actor_uses_current_observation_and_does_not_update_memory():
    z = np.zeros(6)
    model = dict(centers=[np.zeros(11)], weights=[z], ready=z,
                 options=dict(features="raw-proprioception", limit=.8, recovery=.65,
                              feedback=0., width=1.05))
    optical = np.array([0., 0., 1.10, 0., -3.6, 0.])
    state = Memory()
    state.learn([dict(context=np.r_[optical, z[:5]], arm=1, reward=1.0)])
    res = np.full(5, .065)
    actor = Actor(model, np.ones(6), np.full(6, -2.), np.full(6, 2.), state,
                  np.array([np.zeros(5), res]), np.random.default_rng(1), 0.)
    np.testing.assert_allclose(actor.command(optical, z, 5, False), np.r_[res, 0.])
    assert len(state.bank) == 1
    assert actor.choices[0]["tick"] == 5


def test_legal_only_has_no_grades_from_visual_misses_or_outgoing_speed():
    x = [0.] * 11
    records = [dict(context=x, arm=1, landed=False, optical_outgoing_mps=4., optical_gap_m=.04),
               dict(context=x, arm=3, landed=True, optical_outgoing_mps=None, optical_gap_m=None)]
    assert transform(records, "legal") == [dict(context=x, arm=1, reward=0.),
                                            dict(context=x, arm=3, reward=1.)]


def test_visual_only_uses_the_same_values_for_legal_and_illegal_and_abstains():
    x = [0.] * 11
    observed = dict(context=x, arm=1, optical_outgoing_mps=3., optical_gap_m=.04)
    hidden = dict(context=x, arm=3, landed=True, optical_outgoing_mps=None, optical_gap_m=None)
    a, b = transform([dict(observed, landed=True), hidden, dict(observed, landed=False)], "optical")
    assert a == b
    assert .36 < a["reward"] < .61
