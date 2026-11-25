import numpy as np

from scripts.experimental import tt67_action_sensitivity as model


def fixture():
    return dict(
        obs=dict(x=[0., -.90, .94, 0., -4., .9], q=[0.]*6, ref=[0.]*6),
        contact_line=-1.41, probe=6, shape=np.zeros((model.H, 6)).tolist(),
        J=np.eye(5).tolist(), body_jac=np.zeros((5, 10)).tolist(),
    )


def evaluate(x, mode="forward"):
    return model.policy(fixture(), dict(x=np.array(x), q=np.zeros(6), ref=np.zeros(6)), mode)


def test_zero_visible_difference_leaves_same_stroke():
    anchor = fixture()["obs"]["x"]
    a, meta = evaluate(anchor)
    b, _ = evaluate(anchor, "replay")
    assert np.array_equal(a, b)
    assert meta["timing"] == 1.
    assert not any(meta["gain"])


def test_signed_optical_position_change_only_transports_requested_axes():
    p = np.array(fixture()["obs"]["x"])
    change = np.array([.030, 0., -.025, 0., 0., 0.])
    a, meta = evaluate(p+change)
    b, _ = evaluate(p-change)
    assert np.allclose(a, -b)
    assert a[-1, 0] > 0. and a[-1, 2] < 0.
    assert np.all(a[:, [1,3,4,5]] == 0.)
    assert meta["timing"] == 1.
    replay, _ = evaluate(p+change, "replay")
    assert np.all(replay == 0.)


def test_actuator_template_is_smooth_and_saturating():
    ramp = model.smooth(10)
    assert ramp[0, 0] > 0 and ramp[-1, 0] == 1
    assert np.all(np.diff(ramp[:,0]) >= 0)


def test_native_primary_receipt_matches_incoming_and_keeps_original_controls():
    from scripts.experimental import tt67_sensitivity_report as report
    from scripts.experimental.tt4_transfer_trial import _hash, _read

    r = _read(report.REPORT)
    assert r["complete"] and r["receipt"] == _hash(model.OUT)
    assert r["counts"]["source"]["replay"]["answered"] == 6
    assert r["counts"]["source"]["forward"]["answered"] == 6
    assert r["counts"]["neighbor"]["forward"]["matched"] == 18
    assert r["counts"]["neighbor"]["replay"]["matched"] == 18
    assert [r["counts"]["neighbor"][m]["safe"] for m in r["modes"]] == [9, 10]
    assert r["min_clearance_mm"] > 0
    witness = next(x for x in r["rows"] if x["id"] == 5463 and x["recovered"])
    assert witness["shift_cm"] > 5.5
    assert witness["modes"]["forward"]["contact_radius_cm"] < 5.
