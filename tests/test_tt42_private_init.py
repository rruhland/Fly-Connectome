from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental.tt29_vision import Flight
from scripts.experimental.tt42_private_init import PrivateInit


def test_truth_changes_only_selected_private_ball_fields():
    a = mpc.arena()
    controls = []
    try:
        a.reset(seed=0, render=False)
        a.ball[:] = [0.12, -1.1, 1.05]
        a.velocity[:] = [-0.28, -3.21, 0.45]
        a.data.qvel[a._ball_dof + 3 : a._ball_dof + 6] = [43.0, 12.0, -9.0]
        a._received_bounce = True
        truthful = a.data.qpos.copy(), a.data.qvel.copy()
        optical = Flight(
            np.array([0.13, -1.13, 1.02]),
            np.array([-0.21, -3.0, -0.11]),
            0.0,
            0.12,
            0.1,
            0.01,
            5,
            False,
        )
        tracker = SimpleNamespace(latest=optical, filter=None)
        for mode in ("stock", "pv", "spin", "all"):
            actor = PrivateInit(a, mode)
            controls.append(actor)
            actor.observe(
                0, tracker, a.joints.copy(), a.data.ctrl[actor.live_actuators].copy()
            )
            assert actor.sync()
        stock, pv, spin, full = controls
        np.testing.assert_allclose(stock.twin.ball, optical.position)
        np.testing.assert_allclose(pv.twin.ball, a.ball)
        np.testing.assert_allclose(pv.twin.velocity, a.velocity)
        np.testing.assert_allclose(pv.twin.spin, stock.twin.spin)
        assert bool(pv.twin._received_bounce) == bool(stock.twin._received_bounce)
        np.testing.assert_allclose(spin.twin.ball, stock.twin.ball)
        np.testing.assert_allclose(spin.twin.spin, a.spin)
        np.testing.assert_allclose(full.twin.ball, a.ball)
        np.testing.assert_allclose(full.twin.spin, a.spin)
        assert full.twin._received_bounce
        np.testing.assert_array_equal(truthful[0], a.data.qpos)
        np.testing.assert_array_equal(truthful[1], a.data.qvel)
    finally:
        for actor in controls:
            actor.twin.close()
        a.close()
