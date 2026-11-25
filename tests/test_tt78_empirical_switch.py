import numpy as np

from scripts.experimental import tt78_empirical_switch as switch


def matched(stock, reward, nominal, k=0):
    feature=switch.CENTER.copy()
    feature[-1]=int(nominal)
    return dict(spec={"id":k},stock={"landed":stock},
                offers=[dict(feature=feature.tolist(),reward=reward,nominal_legal=nominal)])


def test_matched_label_is_increment_not_alternative_success():
    rows=[matched(True,True,False),matched(True,False,True),matched(False,True,False),matched(False,False,False)]
    assert [y for _,_,y in switch.data(rows)]==[0,-1,1,0]


def test_selector_only_takes_empirical_advantage_and_may_disagree_with_twin():
    row=matched(False,True,False)
    w=np.zeros_like(switch.basis(switch.CENTER))
    w[0]=.45
    counts,_=switch.stats([row],w)
    assert counts["oracle"]==counts["gain"]==counts["switch"]==1
    assert counts["twin_veto"]==counts["loss"]==0
    w[0]=-1.
    counts,_=switch.stats([row],w)
    assert counts["gain"]==counts["intervened"]==0
    assert counts["raw_max"]==1


def test_numeric_features_are_public_and_finite_for_a_no_contact_twin():
    x=np.array([.03,-.84,.916,.14,-3.70,1.20])
    w=switch.main_features(x,np.zeros(6),np.zeros(6),{},"sensitivity",{"distance":2.1})
    assert len(w)==len(switch.FEATURES)
    assert np.isfinite(switch.basis(w)).all()
    assert w[-4]==w[-1]==0.


def test_frozen_roster_has_no_release_overlap_with_moves_or_held():
    p=switch.design()
    train,held=(p["assignments"][n] for n in ("train","held"))
    training,testing=({r["id"] for r in rows} for rows in (train,held))
    assert len(training)==len(train) and len(testing)==len(held)
    assert not training & testing
    assert not (training|testing) & set(p["banned"])
    assert sum(r["weak"] for r in held)>len(held)/2
