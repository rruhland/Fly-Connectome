import numpy as np

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental.tt41_continuation_value import examples
from scripts.experimental.tt41_fragment_warp import align
from scripts.experimental.tt41_practice_curriculum import PracticePartner
from scripts.experimental.tt41_rate_secants import apply, rate_space
from scripts.experimental.tt41_rising_school import optical_impact


def test_estimate_turn_from_images_bracketing_impact():
    hit = 30.375
    intercept = np.array([-0.1, -1.12, 0.92])
    incoming = np.array([0.12, -4.1, 0.83])
    outgoing = np.array([-0.21, 3.8, 3.5])
    points = []
    for t in range(18, 43, 2):
        dt = (t - hit) / 60
        p = intercept + dt * (incoming if dt < 0 else outgoing) + [0, 0, -4.905 * dt**2]
        points.append((t, p))
    t, obs, res = optical_impact(points, 33)
    np.testing.assert_allclose(obs, np.r_[intercept, incoming], atol=1e-10)
    assert abs(t - hit) < 1e-9 and res < 1e-12


def test_rate_intervention_preserves_baseline_when_zero_and_changes_saturated_issued_action():
    stride = np.linspace(0.03, 0.06, 6)
    ref = np.zeros(6)
    lo = np.full(6, -1.6)
    hi = -lo
    goals = np.ones((10, 6)) * 6
    rates = rate_space(goals, ref, stride, lo, hi)
    np.testing.assert_allclose(rates, 0.8)
    base = apply(rates, ref, stride, lo, hi, np.ones((10, 2)), np.zeros((1, 10)))[0]
    np.testing.assert_allclose(base, np.cumsum(rates * stride, axis=0))
    z = np.zeros((1, 10))
    z[0, 0] = -0.3
    changed = apply(rates, ref, stride, lo, hi, np.ones((10, 2)), z)[0]
    assert changed[0, 0] < base[0, 0]
    np.testing.assert_array_equal(changed[:, 1:], base[:, 1:])


def test_phase_alignment_recovers_the_original_at_the_recorded_state():
    x = np.array([0, -0.9, 1, 0.2, -3.7, 0.8])
    q = np.arange(6) * 0.06
    ref = q + 0.08
    source = mpc.query(x, q, ref)
    path = np.linspace(-0.2, 0.16, mpc.HORIZON * 6).reshape(-1, 6)
    np.testing.assert_allclose(
        align(source, path, x, q, ref, None, spatial=False), path, atol=1e-12
    )


def test_delayed_score_assigns_following_own_return_and_censors_absent_offer():
    r = {
        "actions": [0] * 300,
        "searches": [
            {"tick": 60, "target_errors": [1], "contact_delay_s": 0.11},
            {"tick": 180, "target_errors": [2], "contact_delay_s": 0.13},
        ],
        "events": [
            {"kind": "legal_return", "side": -1, "time": 0.9},
            {"kind": "legal_return", "side": 1, "time": 1.8},
            {"kind": "legal_return", "side": -1, "time": 2.9},
            {"kind": "legal_return", "side": 1, "time": 3.8},
        ],
    }
    assert examples(r) == [([1], 1, 1), ([2], 1, None)]
    r["events"].pop()
    assert examples(r) == [([1], 1, None)]
    r["events"].append({"kind": "point", "time": 4.1, "winner": -1})
    assert examples(r) == [([1], 1, 0), ([2], 0, 0)]


def test_curriculum_only_changes_practice_targets_inside_the_original_continuum():
    from scripts.experimental.tt37_opponent import draw

    coach = PracticePartner()
    c = {"id": 7, "index": 2, "seed": 34, "reply_seed": 51, "profile": "reach"}
    assert coach.draw(c, 0, "random", None) == draw(c, 0, "random", None)
    coach.past = [([0.07, -1.3, 1.03, 3.6], int(i % 3 == 0)) for i in range(24)]
    for i in range(15):
        target, _ = coach.draw(c, i, "random", None)
        assert np.all(target >= coach.bands[:, 0]) and np.all(
            target <= coach.bands[:, 1]
        )
        assert target == coach.draw(c, i, "random", None)[0]
