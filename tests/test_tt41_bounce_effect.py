import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental.tt41_bounce_effect import estimate, forecast
from scripts.experimental.tt41_bounce_return import Lift, corrupt


def test_solo_effect_is_a_face_relative_signed_map():
    n = np.array([0.0, 0.0, 1.0])
    before = np.array([0.0, 0.0, -2.0])
    speed = np.array([0.1, 0.0, 0.4])
    law = [1.9, 0.1]
    rot = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    outgoing = forecast(before, speed, n, law)
    np.testing.assert_allclose(outgoing, [0.01, 0.0, 2.56])
    np.testing.assert_allclose(
        rot @ outgoing, forecast(rot @ before, rot @ speed, rot @ n, law)
    )


def test_fit_learns_motion_from_signed_camera_velocities():
    q = np.tile([0.0, 0.0, 1.0], (8, 1))
    before = np.tile([-0.36, 0.14, -2.1], (8, 1))
    speed = np.c_[np.zeros((8, 2)), np.linspace(-0.5, 0.8, 8)]
    after = forecast(before, speed, q, [1.82, 0.2])
    dat = [
        {"pre": a, "post": b, "blade_speed": u, "n": n, "score": 1}
        for a, b, u, n in zip(before, after, speed, q, strict=True)
    ]
    np.testing.assert_allclose(estimate(dat), [1.82, 0.2])
    assert np.linalg.norm(estimate(dat, nomotion=True) - [1.82, 0.2]) > 0.05


def test_common_action_and_effect_online_update_are_bounded():
    t = (1 + np.arange(14)) / 60
    goals = np.arange(14 * 6).reshape(1, 14, 6) / 100
    np.testing.assert_allclose(
        corrupt(goals, t, np.zeros((1, 11)), np.zeros(1, dtype=int)), goals
    )
    skill = Lift({"pooled": [1.88, 0.01]}, "online")
    skill.chosen = 1
    skill.feedback(False, False)
    assert skill.chosen == 1
    skill.feedback(True, False)
    assert skill.chosen is None
    assert skill.feedback_history == [{"style": 1, "legal": True}]
    np.testing.assert_allclose(skill.beta[1], [2.0, 1.0])
