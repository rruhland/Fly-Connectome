import numpy as np

from scripts.experimental import tt47_reached_coverage as c


def test_issued_reference_reconstruction_uses_bounded_control_rates():
    base = np.arange(6, dtype=float)
    row = {
        "actions": [[.2] * 6, [-.1] * 6],
        "motor_audit": {"target_rate_limit_rad_s": [.8 / c.DT] * 6},
    }
    assert np.allclose(c.references(row, base), [base, base + .2, base + .1])


def test_outgoing_reply_mixture_weights_normalize_and_keep_actual_rows():
    center = np.array([0., .63, .35, 0., 1.29, 1.185, 2.30])
    train = center + np.arange(8)[:, None] * .01
    ix, weights, support = c.next_neighbors(train, train[:1])
    assert ix[0, 0] == 0
    assert np.all(np.diff(weights[0]) <= 0)
    assert np.isclose(weights[0].sum(), 1.)
    assert support[0] > 1.


def test_previous_prediction_rejects_incompatible_impact_timing():
    row = {"searches": [{"tick": 35, "contact_delay_s": 5 * c.DT, "target_errors": [0.] * 7}]}
    hit = {"near_landed": True, "near_contact": {"time": 40 * c.DT}}
    assert c.previous_prediction(row, hit)[0] == [0.] * 7
    row["searches"][0]["contact_delay_s"] = 30 * c.DT
    assert c.previous_prediction(row, hit) is None


def test_release_holdout_can_detect_real_sending_joint_effect():
    rng = np.random.default_rng(51)
    rows = []
    for group in range(10):
        for _ in range(18):
            f = np.array([0., .3, 1.1, 0., -4.2, .4]) + rng.normal(size=6) * .012
            out = np.array([0., .62, .30, .02, 1.27, 1.185, 2.3]) + rng.normal(size=7) * .012
            q = rng.normal(size=5) * .23
            q[0] += -.14 if rng.random() < .5 else .14
            qdot = rng.normal(size=5) * .005
            rows.append(dict(release=group, flight=f, recovery=q, velocity=qdot,
                             sender=q + rng.normal(size=5) * .003, sender_velocity=qdot,
                             incoming_time=.8 + rng.normal()*.01, outgoing=out, hit=int(q[0] > 0)))
    value = c.cv(rows)
    assert value["sender_gain_rad"] > .1
    assert value["gains"]["oracle_brier_gain"]["offer_weighted_mean"] > .05
    assert value["gains"]["marginal_brier_gain"]["offer_weighted_mean"] > .04
