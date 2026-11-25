import numpy as np

from scripts.experimental import tt53_responsive_school as school
from scripts.experimental import tt56_long_motor_memory as long
from scripts.experimental import tt56_scored_motor_atlas as atlas
from scripts.experimental import tt56_staircase_practice as staircase


def test_horizon_labels_include_real_intermediate_commands_when_camera_loses_ball():
    x = np.array([0.03, 0.15, 1.0, 0.1, -3.5, 1.3])
    q = np.zeros(6)
    logs = [
        (t, x if t % 2 == 0 else None, q, q, np.full(6, t / 100))
        for t in range(long.FRAMES)
    ]
    state, path, tag = long.collect(logs, "earned")[0]
    assert state.shape == (16,)
    assert tag == "earned"
    np.testing.assert_allclose(path[:, 0], np.arange(long.FRAMES) / 100)


def test_reference_reconstruction_handles_saturation_and_reversal():
    path = atlas.reference_path(
        np.array([0.90]),
        [[0.4], [0.4], [-0.4]],
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
    )
    np.testing.assert_allclose(path.ravel(), [0.90, 1.0, 1.0, 0.60])


def test_unrelated_previous_flight_contributes_no_velocity():
    x = [0.1, -0.82, 0.94]
    ref = np.zeros(6)
    plain = atlas.key(180, x, True, ref, None)
    stale = atlas.key(180, x, True, ref, (79, np.ones(3), np.ones(6)))
    np.testing.assert_array_equal(plain, stale)


def test_practice_draw_is_repeatable_bounded_and_continuous():
    fixture = dict(index=7, seed=83, reply_seed=923, profile="reach", opening="random")

    def example(fraction, scored):
        goal = school.BOUNDS[:, 0] + fraction * np.ptp(school.BOUNDS, axis=1)
        return dict(target=goal.tolist(), near_bounce=True, near_landed=scored)

    history = [
        dict(placements=dict(delivered=[example(0.25, True), example(0.74, False)]))
    ]
    teacher = staircase.Staircase(False, 192, history)
    draws = [teacher.draw(fixture, j, "random", None) for j in range(80)]
    assert all(
        teacher.draw(fixture, j, "random", None) == d for j, d in enumerate(draws)
    )
    values = np.array([d[0] for d in draws])
    assert np.all((school.BOUNDS[:, 0] <= values) & (values <= school.BOUNDS[:, 1]))
    assert all(len(np.unique(c)) == 80 for c in values.T)
    assert {name for _, name in draws} == {"staircase_local", "staircase_uniform"}
