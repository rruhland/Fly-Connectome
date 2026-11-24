import numpy as np

from scripts.experimental.tt38_motor_value import OnlineMotorValue


def test_delayed_motor_credit_uses_issued_action_and_two_real_scores():
    model = OnlineMotorValue({"x": np.zeros((2, 11)), "y": [[0.0, 0.0], [1.0, 1.0]]})
    first = np.arange(13.0) * 0.01
    replacement = first + 0.1
    model.issued(first)
    model.issued(replacement)
    model.feedback([-1, 1], False)
    assert not model.examples
    model.feedback([-1], False)
    assert len(model.examples) == 1 and model.examples[-1][1] == [1.0, 1.0]
    model.issued(first)
    model.feedback([1], True)
    assert model.examples[-1][1] == [1.0, 0.0]
    model.issued(replacement)
    model.feedback([], True)
    assert model.examples[-1][1] == [0.0, 0.0]
    model.issued(first)
    model.issued(None)
    model.feedback([], True)
    assert len(model.examples) == 3


def test_scored_references_wait_for_observed_reversal_and_delayed_reward():
    from scripts.experimental.tt38_scored_joints import ScoredJoints

    s = ScoredJoints()
    for tick in range(48):
        state = np.array([0.0, -1.2, 1.1, 0.0, -2.0 if tick < 28 else 2.0, 0.2])
        s.step(tick, state, np.zeros(6), np.full(6, tick))
        if tick == 16:
            s.credit()
            assert s.strikes == 0
    s.credit()
    assert s.strikes == 1
    assert len(s.states) == len(s.targets) == len(s.leads) == len(s.marks)
    assert len(s.states) > 20
    assert all(
        value[0] == mark[0] for value, mark in zip(s.targets, s.marks, strict=True)
    )
    s.credit()
    assert s.strikes == 1
    s.begin_point()
    s.credit()
    assert s.strikes == 0


def test_rally_credit_can_only_use_a_subsequent_actual_score():
    from scripts.experimental.tt38_rally_value import OnlineRallyValue

    m = OnlineRallyValue({"x": [np.zeros(11)], "y": [[0.5] * 4]})
    for i in range(3):
        m.issued(np.arange(13.0) + i)
        m.feedback([1], False)
        if i < 2:
            m.feedback([-1], False)
            assert not m.examples
    m.feedback([], True)
    assert [y for _, y in m.examples] == [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]


def test_far_return_cannot_receive_near_motor_credit():
    from scripts.experimental.tt38_scored_joints import ScoredJoints

    r = ScoredJoints()
    for tick in range(50):
        optical = np.array([0, 1.1, 1.0, 0, 2.3 if tick < 20 else -2.3, 0])
        r.step(tick, optical, np.zeros(6), np.zeros(6))
    r.credit()
    assert r.strikes == 0
    r.complete({"id": 1})
    assert r.completed == 1


def test_preparation_only_learns_from_a_real_incoming_opportunity():
    from scripts.experimental.tt39_preparation import Preparation

    p = Preparation(fixed=0.5)
    q = np.zeros(6)
    x = np.array([0.1, 0.5, 1.2, 0.2, -3.1, 1.2])
    p.feedback(True, False)
    assert not p.new
    assert p.observe(x, q) == 0.5
    p.observe(x + 0.01, q)
    p.feedback(False, False)
    assert not p.new
    p.feedback(True, False)
    p.feedback(True, False)
    p.observe(x, q)
    p.feedback(False, True)
    assert [r[-1] for r in p.new] == [1.0, 0.0]
    assert p.new == p.examples
