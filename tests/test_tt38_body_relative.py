"""Check that encoder-relative feedback stays causal and its absolute control is exact."""

import numpy as np

from scripts.experimental.tt37_joint_deep import DeepField
from scripts.experimental.tt38_body_relative import Body, FREQ, RelativeField, groups, input_motion
from scripts.experimental.tt38_body_relative_augment import translate
from scripts.experimental.tt38_body_relative_face import single_joint_command
from scripts.experimental.tt38_body_relative_online import PassageField, passage


def linearish_body():
    weights = np.zeros((1 + 2 * len(FREQ), 3))
    for dimension in range(3):
        harmonic = np.eye(5)[dimension]
        at = np.flatnonzero(np.all(FREQ == harmonic, axis=1))[0]
        weights[1 + len(FREQ) + at, dimension] = .3
    return Body(weights)


def test_batched_body_jacobian_is_command_derivative():
    rng = np.random.default_rng(3)
    body = Body(rng.normal(scale=.002, size=(1 + 2 * len(FREQ), 3)))
    q = rng.normal(size=(11, 5))
    direction = rng.normal(size=q.shape)
    ray = body.jacobian(q) @ direction[:, :, None]
    numeric = (body.position(q + 1.e-5 * direction) - body.position(q - 1.e-5 * direction)) / 2.e-5
    np.testing.assert_allclose(ray.squeeze(-1), numeric, atol=1.e-9)


def test_motion_does_not_bridge_hidden_or_replayed_ticks():
    states = np.zeros((6, 11))
    states[:, 6] = [.0, .02, .03, .04, .08, .1]
    marks = np.c_[[3, 4, 6, 7, 8, 9], np.full(6, -.1)]
    receipt = {"practice": [{"frames": 4, "card": {"id": 1}}, {"frames": 2, "card": {"id": 2}}]}
    point, release = groups(receipt)
    p, v = input_motion(states, marks, point, linearish_body())
    np.testing.assert_array_equal(release, [1, 1, 1, 1, 2, 2])
    np.testing.assert_allclose(v[[0, 2, 4]], 0.)
    np.testing.assert_allclose(v[[1, 3, 5]], (p[[1, 3, 5]] - p[[0, 2, 4]]) * 60.)


def test_absolute_regression_matches_original_pointwise():
    rng = np.random.default_rng(1)
    s = rng.normal(scale=.14, size=(120, 11))
    s[:, 4] += np.repeat([-2., 2.], 60)
    y = rng.normal(scale=.08, size=(120, 6))
    bank = {"states": s, "targets_now": y, "marks": np.c_[np.arange(len(s)), np.zeros(len(s))]}
    meta = {"practice": [{"frames": len(s), "card": {"id": 3}}]}
    original = DeepField(bank, "local")
    candidate = RelativeField(bank, meta, linearish_body().weights, "absolute")
    for state in s[::13]:
        x, q = state[:6], state[6:]
        a, conf = original.get(x, q)
        b, candidate_conf = candidate.at(x, q, np.ones(3), np.ones(3))
        assert conf == candidate_conf
        if a is None:
            assert b is None
        else:
            np.testing.assert_allclose(a, b, atol=1e-13)


def test_synthetic_translation_bounds_and_accuracy():
    body = linearish_body()
    q = np.zeros((2, 5))
    q[1, 0] = .1
    lower, upper = np.full(5, -.1), np.full(5, .1)
    moved, supported = translate(body, q, [.021, 0., .009], lower, upper)
    np.testing.assert_array_equal(supported, [True, False])
    assert np.all(moved >= lower) and np.all(moved <= upper)
    np.testing.assert_allclose(body.position(moved[0]) - body.position(q[0]), [.021, 0., .009], atol=.001)


def test_signed_passage_requires_two_nearby_measured_exposures():
    first, second = np.array([.04, .11, -.14]), np.array([.06, -.11, -.10])
    tick, gap = passage(4, first, 6, second)
    assert tick == 5
    np.testing.assert_allclose(gap, [.05, -.12])
    assert passage(4, first, 9, second) is None
    assert passage(4, second, 6, first) is None
    assert passage(4, np.array([.01, .11, .01]), 6, np.array([.01, -.11, .01])) is None


def test_passage_credit_only_happens_after_qualifying_terminal_exposure():
    states = np.zeros((24, 11))
    states[:, 4] = np.repeat([-2., 2.], 12)
    bank = {"states": states, "targets_now": np.zeros((24, 6)),
            "marks": np.c_[np.arange(24), np.zeros(24)]}
    receipt = {"practice": [{"frames": 24, "card": {"id": 1}}]}
    weights = linearish_body().weights
    actor = PassageField(bank, receipt, weights, mode="learn")
    state = np.zeros(9)
    miss = (10., np.array([.1, -.12]), [(7, state, np.zeros(2))])
    actor.gaps = [miss]
    actor.end_point(terminal=False)
    assert actor.memory == []
    actor.clock_tick(12)
    actor.score_feedback(1)
    actor.end_point(terminal=True)
    assert actor.memory == []
    actor.begin_point()
    actor.gaps = [miss]
    actor.end_point(terminal=True)
    np.testing.assert_allclose(actor.correction(state), [.04, -.045])

    frozen = PassageField(bank, receipt, weights, mode="carry", memory=actor.memory.copy())
    frozen.gaps = [miss]
    frozen.end_point(terminal=True)
    assert frozen.memory == actor.memory

    disabled = PassageField(bank, receipt, weights, mode="fixed", memory=actor.memory.copy())
    np.testing.assert_array_equal(disabled.correction(state), [0., 0.])


def test_joint_face_decoupling_decreases_independent_face_disagreement():
    center = linearish_body().weights
    angular = np.zeros_like(center)
    angular[0, 1] = 1.
    for joint, axis in [(3, 0), (4, 2)]:
        harmonic = np.eye(5)[joint]
        at = np.flatnonzero(np.all(FREQ == harmonic, axis=1))[0]
        angular[1 + len(FREQ) + at, axis] = .6
    pose = Body(np.c_[center, angular])
    relative = np.array([.12, -.10, .17, .32, -.22, -.1])
    absolute = np.array([.20, .00, .09, .01, -.01, -.1])
    estimate = single_joint_command(relative, absolute, pose, np.full(5, -1.), np.full(5, 1.))
    target_orientation = pose.position(absolute)[3:]
    np.testing.assert_allclose(pose.position(estimate)[:3], pose.position(relative)[:3])
    assert np.linalg.norm(pose.position(estimate)[3:] - target_orientation) < .5 * np.linalg.norm(
        pose.position(relative)[3:] - target_orientation)
