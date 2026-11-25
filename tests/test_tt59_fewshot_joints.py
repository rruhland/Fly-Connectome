from collections import Counter

import numpy as np

from scripts.experimental import tt59_carry_fewshot as carry
from scripts.experimental import tt59_fewshot_joint_games as games
from scripts.experimental import tt59_fewshot_joints as tt
from scripts.experimental.tt4_transfer_trial import _hash, _read


def test_retries_skip_repeated_commands_and_respect_failed_attempts():
    a = np.zeros((tt.H, 6))
    near = np.full_like(a, .04)
    far = np.full_like(a, .32)
    tries = tt.Attempts([(a, {"score": 0}), (near, {"score": .01}), (far, {"score": .15})])
    assert np.array_equal(tries.choose()[0], a)
    tries.miss(a)
    assert np.array_equal(tries.choose()[0], far)
    tries.miss(far)
    assert np.array_equal(tries.choose()[0], near)
    tries.miss(near)
    assert tries.choose() is None


def test_pairs_shuffle_labels_not_public_baseline_or_fixed_candidate_budget():
    plan = _read(tt.PLAN)
    with np.load(tt.MODEL) as f:
        assert len(f["ids"]) == 18 and len(set(f["ids"])) == 17
        assert not set(f["ids"]) & {r["id"] for r in plan["cases"]}
        assert np.all(f["ids"] != f["ids"][f["permutation"]])
        assert np.array_equal(np.sort(f["permutation"]), np.arange(len(f["ids"])))
        x, context = f["x"][0], f["features"][0]
    readout = tt.Corrections()
    anchors = np.zeros((3, tt.H, 6))
    a, va = readout.proposals(x, context, anchors, "paired")
    b, vb = readout.proposals(x, context, anchors, "shuffled")
    assert a.shape == b.shape == (tt.CANDIDATES, tt.H, 6)
    assert not np.array_equal(a, b)
    constants = ("context_release", "distance", "anchor", "gain", "phase_tick", "failure_prior")
    assert all(all(x[k] == y[k] for k in constants) for x, y in zip(va, vb, strict=True))


def test_sealed_successes_have_exact_stock_miss_positive_cad_and_disjoint_follow_up():
    old, later = _read(tt.PLAN), _read(carry.PLAN)
    overlap = set(old["excluded"]) | {r["id"] for r in old["cases"]}
    assert not overlap & {r["id"] for r in later["cases"]}
    for path, plan, modes in (
        (tt.OUT, tt.PLAN, tt.MODES), (carry.OUT, carry.PLAN, ("base18", "carry23")),
    ):
        receipt = _read(path)
        assert receipt["complete"] and receipt["protocol"] == _hash(plan)
        learned = Counter()
        for row in receipt["rows"]:
            assert not row["stock"]["landed"]
            assert row["stock"]["matched_far_hit"] and row["stock"]["matched_near_bounce"]
            for mode in modes:
                arm = row["arms"][mode]
                assert len(arm["trials"]) <= tt.ATTEMPTS
                for step in arm["trials"]:
                    if step["reward"]:
                        assert step["matched_far_hit"] and step["matched_near_bounce"] and step["landed"]
                        assert step["clearance"]["penetration_samples"] == 0
                        assert step["clearance"]["nearest"]["signed_m"] > 0
                        assert arm["first"] == step["number"]
                        learned[mode] += 1
        expected = {"paired": 5, "shuffled": 2} if path == tt.OUT else {"base18": 1, "carry23": 1}
        assert learned == expected


def test_full_points_are_one_launch_and_independently_replayed():
    receipt = _read(games.OUT)
    assert receipt["complete"] and receipt["protocol"] == _hash(games.PLAN)
    assert len(receipt["rows"]) == 4
    for row in receipt["rows"]:
        for arm in row["arms"].values():
            assert arm["complete_point"]
            assert arm["replay"]["launches"] == 1
            assert arm["replay"]["clearance"]["penetration_samples"] == 0
