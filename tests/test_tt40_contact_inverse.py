"""Check signed first-contact action labels and the held-readout support gate."""

import numpy as np

from scripts.experimental.tt40_contact_inverse import Effect, basis, project


def test_action_contrast_is_eased_and_excludes_after_first_hit():
    goal = np.array([[.11, 0., .10, -.15, .09], [.10, 0., .18, -.26, .11]])
    assert not basis(-1).any()
    np.testing.assert_allclose(basis(20), [0., 1.])
    q = basis(np.arange(1,25)) @ goal
    step = dict(at=20, impact=dict(time=(20+19.5)/60))
    base = dict(step, reference=np.zeros((24,6)).tolist())
    ref = np.c_[q, np.zeros(24)]
    ref[19:,:] *= -8.  # No action after the camera-first impact can change this fit.
    alternative = dict(step, reference=ref.tolist())
    fitted,residual,n = project(base,alternative)
    assert n == 19 and residual < .004
    np.testing.assert_allclose(fitted,goal.ravel(),atol=.006)
    assert project(base,dict(alternative,impact=None)) is None


def test_positive_observed_gradient_proposes_a_positive_shift():
    state = np.zeros(16)
    records=[]
    for n in range(3):
        control=np.zeros(10)
        control[n]=.14
        records.append(dict(state=state.tolist(),base=[0.,2.5,1.5],delta=[0.,.32,.55],
                            theta=control.tolist(),score=1,release=n,projection_rmse_rad=.01))
    effect=Effect(dict(records=records))
    move,score=effect.correction(state,"inverse")
    assert score["predicted_change_mps"][1] > 0 and score["predicted_change_mps"][2] > 0
    assert np.max(abs(move)) <= .26
    backward,_=effect.correction(state,"reversed")
    np.testing.assert_allclose(backward,-move)
    assert not effect.correction(np.full(16,100.),"inverse")[0].any()


def test_carry_requires_a_local_measured_failure_and_uses_its_signed_velocity():
    state = np.zeros(16)
    step = np.r_[.12, np.zeros(9)]
    fit = dict(records=[dict(state=state,base=[0.,2.4,1.7],delta=[0.,.5,.6],theta=step,
                             score=1,release=12,projection_rmse_rad=.03)])
    witness = [dict(context=state,outgoing=[.16,2.5,1.8])]
    model = Effect(fit,witness)
    good,note = model.correction(state,"carry")
    assert np.any(good) and note["measured_miss_distance"] == 0.
    assert note["observed_base_mix_mps"] == witness[0]["outgoing"]
    shifted=state.copy()
    shifted[4]=.82
    assert np.any(model.correction(shifted,"inverse")[0])
    assert not np.any(model.correction(shifted,"carry")[0])


def test_genuine_scoreless_secant_is_local_and_changes_the_response():
    state=np.zeros(16)
    first=np.r_[.18,np.zeros(9)]
    record=dict(state=state,base=[0.,2.7,1.6],delta=[0.,.40,.60],theta=first,
                score=1,release=1,projection_rmse_rad=.01)
    which=dict(context=state,executed_contrast=first,measured_change=[0.,.01,.02],
               observed_baseline=record["base"])
    cold=Effect(dict(records=[record]))
    learned=Effect(dict(records=[record],owned_secants=[which]))
    a,d0=cold.correction(state,"inverse")
    b,d1=learned.correction(state,"inverse")
    assert d1["own_failure_secant_weight"]==1.
    assert not np.allclose(a,b)
    assert d1["predicted_change_mps"][2] < d0["predicted_change_mps"][2]
