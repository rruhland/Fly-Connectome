import numpy as np

from scripts.experimental import tt48_rim_recovery as r


def test_compact_smooth_motor_advice_handoff():
    assert np.array_equal(r.ease(np.array([-1., 0., .5, 1., 2.])), [0, 0, .5, 1, 1])
    d = np.arange(6, dtype=float)
    assert np.array_equal(r.remap(d, -1., -1.), d)
    assert np.array_equal(r.remap(d, -1., 1.), [0, 1, 2, 3, -4, 5])


def test_dynamics_gate_distinguishes_glances_from_good_rim_returns():
    f = np.array([.08, 1., .2, .4, -.5, .1, -1.57, .99, .075])
    assert r.glance(f)
    strong = f.copy()
    strong[3] = 3.6
    assert not r.glance(strong)
    center = f.copy()
    center[8] = .036
    assert not r.glance(center)


def test_held_roster_has_three_real_streams_per_untrained_release():
    p = r._read(r.PLAN)
    d = r._read(r.STOCK)
    assert p["cards"] == d["cards"]
    assert len(p["cards"]) == 24
    assert len({x["id"] for x in p["cards"]}) == 8
    assert len({(x["id"], x["seed"], x["reply_seed"]) for x in p["cards"]}) == 24
    assert not ({x["id"] for x in p["cards"]} & {y["release"] for y in p["corrections"]})
    assert p["cap"] == 240


def test_real_signed_discrepancy_is_not_equivalent_to_reversing_it():
    from scripts.experimental.tt48_edge_evidence_fit import Evidence

    rows = [dict(id=k, x=[0.] * 9, virtual_contact=True, actual_legal=(k != 3),
                 nominal=[0., 2.2, 3.1], rgb_after=[0., 3.35, 3.2])
            for k in (1,2,3)]
    probe=dict(id=4, x=[0.] * 9, weak=True, virtual_contact=True,
               native_legal=False, face=.90, nominal=[0., 2.2, 3.1],
               nominal_p=[0.,-1.55,1.01])
    model=Evidence(rows)
    assert model.assess(probe,"learned")["accepted"]
    assert not model.assess(probe,"reverse")["accepted"]
    assert not model.assess(dict(probe,weak=False),"learned")["accepted"]
    # An identical release cannot authorize its own override.
    assert not model.assess(dict(probe,id=1),"learned")["accepted"]
