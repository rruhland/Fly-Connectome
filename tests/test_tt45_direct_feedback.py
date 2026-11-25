from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt45_composed_feedback import compose
from scripts.experimental.tt45_context_feedback import Approach, Feedback, Selector


def test_camera_approach_is_selected_once_per_incoming_ball():
    source = Approach()
    x = np.array([0.02, 0.8, 0.95, 0.03, -3.5, 0.2])
    q = np.zeros(6)
    assert all(source.observe(x, q) is None for _ in range(6))
    key = source.observe(x, q)
    assert key.shape == (16,)
    assert source.observe(None, q) is None
    assert all(source.observe(x, q) is None for _ in range(8))
    assert source.observe(-x, q) is None
    for _ in range(6):
        assert source.observe(x, q) is None
    assert np.array_equal(source.observe(x, q), key)


def test_only_the_current_choice_receives_credit_and_context_changes_preference():
    examples = [
        dict(context=[float(k)] * 16, arm=a, landed=(a == k))
        for k in (0, 1)
        for a in (0, 1)
        for _ in range(12)
    ]
    selector = Selector(examples)
    assert selector.chance(np.zeros(16)).argmax() == 0
    assert selector.chance(np.ones(16)).argmax() == 1
    actor = SimpleNamespace(selector=selector, decision=dict(context=[0.0] * 16, arm=1))
    Feedback.earned(actor, False)
    Feedback.earned(actor, True)
    assert len(selector.examples) == len(examples) + 1
    assert not selector.examples[-1]["landed"]


def test_later_joint_control_is_queried_at_the_result_of_each_prefix():
    seen = []

    def priors(x, q, ref, n):
        seen.append((x.copy(), q.copy(), ref.copy()))
        return [(np.full((21, 6), q[0]), False)]

    field = SimpleNamespace(priors=priors)
    prefixes = np.ones((3, 21, 6)) * np.arange(1, 4)[:, None, None]
    trace = np.zeros((3, 4, 17))
    trace[:, 3, 4] = -1.1
    trace[:, 3, 6:11] = [[0.1] * 5, [0.4] * 5, [0.7] * 5]
    trace[:, 2, 11:] = 0.25
    trace[:, 3, 11:] = 0.85
    trace[2, 3, 4] = 1.0
    result = compose(field, trace, prefixes, 3)
    assert len(result) == 2
    for i, tape in enumerate(result):
        np.testing.assert_array_equal(tape[:3], prefixes[i, :3])
        np.testing.assert_allclose(tape[3:], np.full((21, 6), seen[i][1][0]))
        np.testing.assert_allclose(seen[i][2], 0.25)


def test_shared_body_miss_feedback_learns_signed_reachable_motion():
    from scripts.experimental.tt45_error_learning import Learning

    geometry = SimpleNamespace(
        pose=lambda q, jacobian: (
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            np.eye(3, 5),
            np.zeros((3, 5)),
        )
    )
    key = np.zeros(16)
    model = Learning()
    zero = np.zeros(5)
    assert (
        model.update(
            key, (zero, np.array([0.1, 0.0, 0.0]), None), zero, False, geometry
        )
        == "no_scored_example"
    )
    model.update(key, (zero, np.zeros(3), None), zero, True, geometry)
    assert (
        model.update(
            key, (zero, np.array([0.1, 0.0, 0.0]), None), zero, False, geometry
        )
        == "proximity_error"
    )
    assert 0.02 < model.apply(key)[0] < 0.1
    np.testing.assert_allclose(model.apply(key)[1:], 0.0)


def test_optical_feedback_corrects_the_moving_hand_not_just_the_ball():
    from scripts.experimental.tt45_visual_tracking import corrected

    goal = np.zeros(6)
    at_search = np.zeros(17)
    at_search[4] = -3.0
    new = at_search[:6].copy()
    new[0] += 0.01
    jac = np.eye(3, 5)
    q = np.zeros(5)
    ahead = corrected(goal, new, q, at_search, jac, 0.08)
    assert 0.0 < ahead[0] < 0.02
    q[0] = 0.01
    np.testing.assert_array_equal(corrected(goal, new, q, at_search, jac, 0.08), goal)
    np.testing.assert_array_equal(
        corrected(goal, new, q * 0, at_search, jac, 0.02), goal
    )


def test_mixing_needs_complementary_errors_in_the_same_contact_mode():
    from scripts.experimental.tt47_interpolate import bracket

    action = np.stack([np.zeros((21, 6)), np.ones((21, 6))])
    opposite = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    costs = np.zeros(2)
    p = bracket(action, opposite, costs, [False, False])
    assert len(p) == 2
    np.testing.assert_allclose(p[0], 0.5, atol=0.0001)
    assert not bracket(action, opposite, costs, [False, True])
    assert not bracket(action, abs(opposite), costs, [True, True])


def test_delayed_recovery_reward_credits_the_preceding_incoming_attempt_once():
    from collections import Counter
    from scripts.experimental.tt46_recovery_learning import Actor

    sample = (np.zeros(11), np.ones(5))
    state = SimpleNamespace(
        pending=sample, fit=SimpleNamespace(pairs=[]), diag=Counter()
    )
    Actor.confirm(state)
    Actor.confirm(state)
    assert state.fit.pairs == [sample]
    assert state.diag["earned_next_position"] == 1


def test_online_motor_forecast_matches_batch_ridge_on_observed_examples():
    from scripts.experimental.tt47_predictive_motor import Forecast

    rng = np.random.default_rng(11)
    rows = rng.normal(size=(7, 3))
    targets = rng.normal(size=(7, 5))
    model = Forecast.__new__(Forecast)
    model.n = 0
    model.inverse = np.eye(3) / 3.0
    model.weights = np.zeros((3, 5))
    for x, y in zip(rows, targets, strict=True):
        model.update(x, y)
    batch = np.linalg.solve(rows.T @ rows + 3.0 * np.eye(3), rows.T @ targets)
    np.testing.assert_allclose(model.weights, batch, rtol=1e-12, atol=1e-12)
    assert model.n == 7
