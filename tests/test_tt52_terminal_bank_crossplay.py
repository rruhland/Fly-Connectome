from scripts.experimental.tt52_terminal_bank_crossplay import fork, mpc


def test_cut_keeps_complete_far_impact_and_does_not_use_near_bounce_to_act():
    hit = {"kind": "contact", "side": -1, "step": 3, "time": 2.5 * mpc.DT}
    release = {"kind": "separation", "collider": "rival_racket", "step": 4, "time": 3.25 * mpc.DT}
    bounce = {"kind": "legal_return", "side": -1, "step": 8, "time": 7.2 * mpc.DT}
    offer = dict(leg=0, far_contact=hit, near_bounce=bounce, near_landed=False)
    row = dict(placements={"delivered": [offer]}, actions=[[0.0]] * 12,
               events=[hit, release, bounce, {"kind": "point"}])
    case = fork(row)
    assert case["cut_tick"] == 4
    assert case["far_separation"] == release
    assert case["role"] == "terminal_offer"


def test_no_offer_remains_a_predeclared_null_cell():
    assert fork({"placements": {"delivered": []}}) == {"role": "no_native_offer", "cut_tick": None}
