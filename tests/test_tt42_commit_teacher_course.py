"""Negative outcome credit must follow a nearby observed, actually issued motor prefix."""

import numpy as np

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental.tt42_commit_teacher_course import NegativeCredit, diverse_pair


def test_miss_credit_requires_both_optical_context_and_motor_similarity():
    before = np.array([0.02, -1.09, 1.04, 0.2, -3.2, 0.4])
    q = np.linspace(-0.40, 0.41, 6)
    ref = q + 0.08
    seq = np.array([ref + 0.01 * (t + 1) for t in range(mpc.HORIZON)])
    credit = NegativeCredit()
    assert credit.keep([]) == 0
    history = [(t, before, q, ref, goal) for t, goal in enumerate(seq)]
    assert credit.keep(history) > 0
    key = mpc.query(before, q, ref)
    assert credit.resembles_failed(key, seq, ref) == [0.0, 0.0]
    # Similar approach can be rescued by a genuinely different movement.
    assert credit.resembles_failed(key, seq - 0.19, ref) is None
    assert credit.resembles_failed(key + 1.0, seq, ref) is None
    assert credit.digest()["rows"] > 0


def test_missing_or_passed_ball_has_no_negative_motor_credit():
    q = np.zeros(6)
    away = np.array([0.0, -1.0, 1.0, 0.0, 3.0, 0.7])
    assert NegativeCredit().keep([(i, None, q, q, q) for i in range(27)]) == 0
    assert NegativeCredit().keep([(i, away, q, q, q) for i in range(27)]) == 0


def test_two_slot_coverage_does_not_spend_both_slots_on_the_same_motion():
    q = np.zeros((mpc.HORIZON, 6))
    local = [(q, "a", 0.8), (q + 0.02, "b", 0.81), (q + 0.23, "c", 0.93)]
    weak = (q + 2.0, "outside support", 1.01)
    assert diverse_pair(local + [weak]) == (0, 2)
