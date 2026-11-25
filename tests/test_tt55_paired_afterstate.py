from pathlib import Path

import numpy as np

from scripts.experimental.tt4_transfer_trial import _read
from scripts.experimental.tt55_paired_afterstate import PLAN, line_info
from scripts.experimental.tt55_paired_preference import fit, signs


def test_measured_geometry_and_score_are_independent():
    line = {"xyz": [0.02, -1.10, 0.94], "velocity": [0.0, -3.6, 0.15]}
    answer, geometry, mode = line_info(None, None, {"receive_line": line, "near_landed": True})
    assert answer == line and mode == "measured"
    assert geometry[0] < 1.00 and abs(geometry[1]) < 0.10


def test_release_disjoint_protocol_and_common_independent_seeds():
    assert Path(PLAN).is_file()
    p = _read(PLAN)
    assert len(p["held"]) == 17 and len(p["train"]) == 32
    assert not ({x["id"] for x in p["train"]} & set(p["excluded_training"]))
    seeds = [s for role in ("held", "train") for x in p[role] for s in x["goals"]]
    assert len(set(seeds)) == len(seeds) == 49 * 6


def test_paired_ridge_has_no_candidate_order_preference():
    x = np.array([[.2, .4], [.5, -.1], [1., .3]])
    y = np.array([.1, -.4, .2])
    w = fit(x, y)
    np.testing.assert_array_equal(fit(-x, -y), w)
    assert signs([1., 0., 1.], [1., 0., -1.])["same_sign"] == 1
