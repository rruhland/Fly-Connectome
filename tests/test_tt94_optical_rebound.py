import numpy as np

from scripts.experimental.tt94_optical_rebound import G, SURFACE, Rebound, estimate


def test_back_project_before_and_after_observed_rebound():
    # Ideal ballistic arcs through the same point under different horizontal speeds.
    for after, zv, vx in [(False, -3.0, 0.6), (True, 2.6, 0.4)]:
        t = np.array([0.022, 0.056]) * (1 if after else -1)
        states = [
            (
                i,
                np.array(
                    [
                        vx * s,
                        0.55 + 3 * s,
                        SURFACE + zv * s - 0.5 * G * s * s,
                        vx,
                        3.0,
                        zv - G * s,
                    ]
                ),
            )
            for i, s in enumerate(t)
        ]
        np.testing.assert_allclose(
            estimate(states, after), [0, 0.55, vx, 3.0, zv], atol=1e-8
        )


def test_observer_needs_independent_postbounce_samples_and_only_emits_once():
    f = Rebound()

    def state(z, v):
        return np.array([0.0, 0.6, z, 0.0, 3.0, v])

    assert f.update(4, state(0.98, -2.4)) is None
    assert f.update(6, state(0.87, -2.8)) is None
    assert f.update(8, state(0.81, 2.3)) is None
    assert f.update(10, state(0.89, 2.0)).shape == (10,)
    assert f.update(12, state(0.94, 1.8)) is None
