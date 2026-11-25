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
