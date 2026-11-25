import numpy as np

from scripts.experimental import tt100_episode_policy as ep
from scripts.experimental import tt100_episode_report as report
from scripts.experimental.tt4_transfer_trial import _read


def test_neutral_settings_and_bounded_two_weight_update():
    assert ep.settings([0., 0.]) == (-.49, 1.)
    assert ep.settings([40., -40.]) == ep.settings([ep.BOUND, -ep.BOUND])
    direction = np.array([.6, -.8])
    origin = np.zeros(2)
    advance, seen = ep.update(origin, direction, [{"plus": 5, "minus": 1}]*2)
    reverse, _ = ep.update(origin, direction, [{"plus": 1, "minus": 5}]*2)
    contradict, tied = ep.update(origin, direction, [{"plus": 5, "minus": 1}, {"plus": 1, "minus": 5}])
    assert np.dot(advance, direction) > 0 and np.array_equal(advance, -reverse)
    assert np.array_equal(contradict, origin)
    assert seen["sign_agreement"] == 1. and tied["sign_agreement"] == 0.
    assert tied["used_contrast_weight"] < seen["used_contrast_weight"]
    clipped, _ = ep.update(np.array([ep.BOUND, -ep.BOUND]), direction, [{"plus": 5, "minus": 1}]*2)
    assert np.array_equal(clipped, [ep.BOUND, -ep.BOUND])


def test_sealed_schedule_has_disjoint_practice_and_true_held_releases():
    spec = ep.design()
    bank, reserved, _, _ = ep.available()
    train = [i["card"]["id"] for b in spec["practices"] for i in b["cards"]]
    held = [i["card"] for i in spec["held"]]
    assert len(set(train)) == len(train) == 6
    assert set(train) == set(spec["train_release_ids"])
    assert not set(train).intersection(bank | reserved)
    assert len({i["id"] for i in held}) == len(held) == 6
    assert not {i["id"] for i in held}.intersection(bank | set(train))
    assert all(i in _read(ep.transfer.PLAN)["practice"] for i in held)
    x = _read(ep.NEXT)
    ext = {i["card"]["id"] for i in x["held"]}
    assert len(ext) == 4 and not ext.intersection(bank | set(train) | {i["id"] for i in held})
    for b in spec["practices"]:
        assert np.isclose(np.linalg.norm(b["direction"]), 1)
        assert all(set(i["order"]) == {"plus", "minus"} for i in b["cards"])


def test_first_changed_command_reports_exact_same_incoming_only():
    def tape(actions, bounce, landed):
        return {"actions": actions, "placements": {"delivered": [
            {"leg": 0, "far_contact": {"step": 0, "y": .49},
             "near_bounce": {"step": 1, "y": bounce}, "near_landed": landed}]}}

    a = tape([[0.], [0.], [0.]], -.8, False)
    b = tape([[0.], [0.], [1.]], -.8, True)
    assert report.first_difference(a, b) == {
        "tick": 2, "leg": 0, "same_delivery": True,
        "baseline_scored": False, "candidate_scored": True,
    }
    assert report.first_difference(a, a) is None
    assert not report.first_difference(a, tape(b["actions"], -.78, True))["same_delivery"]


def test_archived_full_comparison_uses_frozen_policy_and_exact_first_offer():
    h = _read(ep.OUT)
    d = ep.design()
    c = _read(ep.CHECK)
    assert h["complete"] and [r["card"] for r in h["rows"]] == [r["card"] for r in d["held"]]
    for r in h["rows"]:
        ordinary, episode = (r["arms"][k] for k in ("ordinary", "episode"))
        assert ordinary["theta"] == [0.0, 0.0] and episode["theta"] == c["theta"]
        assert ordinary["memory"]["before"] == episode["memory"]["before"]
        for p in (ordinary, episode):
            assert p["complete"] and p["events"][-1]["kind"] == "point"
            assert p["memory"]["before"] == p["memory"]["after"]
        shift = report.first_difference(ordinary, episode)
        assert shift["same_delivery"] and shift["leg"] == 0
        assert shift["baseline_scored"] and shift["candidate_scored"]
