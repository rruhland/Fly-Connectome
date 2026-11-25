import numpy as np

from scripts.experimental import tt91_packed_retina as cam
from scripts.experimental import tt91_retina_benchmark as trial
from scripts.experimental.tt29_vision import Sighting
from scripts.experimental.tt4_transfer_trial import _hash,_read
from scripts.experimental.tt75_retina import Retina


def test_fused_integer_rejector_matches_reference_moments():
    rng=np.random.default_rng(91017)
    original=object.__new__(Retina)
    optimized=object.__new__(cam.CulledMonoRetina)
    original.size=optimized.size=192
    A=np.diag([3.,6.])
    for _ in range(16):
        rgb=rng.integers(0,256,(192,192,3),dtype=np.uint8)
        hit=Sighting(np.array([96.0,96.0]),10.,False,80)
        assert original.usable(rgb,hit,A)==optimized.usable(rgb,hit,A)


def test_packing_is_rejected_on_a_real_optically_valid_trace():
    d=_read(trial.ROOT/"packing.json.gz")
    assert d["plan"]==_hash(trial.ROOT/"packing.protocol.json")
    early,late=d["rows"][:2]
    assert early["parity"]["packed"]["optical_point_exact"]==194
    assert late["parity"]["packed"]["optical_point_availability_same"]==249
    assert late["parity"]["packed"]["optical_point_exact"]<249


def test_all_six_true_rgb_replays_have_exact_sensor_and_estimator_bytes():
    from scripts.experimental import tt91_retina_report as report

    d=_read(report.REPORT)["paired"]
    p=d["summary"]["mono"]
    assert d["complete"] and d["receipt"]==_hash(trial.OUT)
    assert len(d["rows"])==6 and all(r["native_prefix_exact"] for r in d["rows"])
    assert p["totals"]["exposures"]==1409 and p["totals"]["valid"]==1314
    assert p["raster_changes"]["total"]==309878784 and not p["raster_changes"]["changed"]
    assert p["parity"]["detector_exact"]==p["parity"]["crops"]==2802
    assert p["parity"]["optical_point_exact"]==1409
    assert p["fits"]["fit_packet_exact"]==p["fits"]["consumed"]==375
    assert p["price"]["ms_per_exposure"]<d["summary"]["original"]["price"]["ms_per_exposure"]


def test_actual_scored_full_point_shares_commands_and_native_outcomes():
    from scripts.experimental import tt91_retina_action as action
    from scripts.experimental import tt91_retina_report as report

    tested=_read(action.OUT)
    assert tested["plan"]==_hash(action.PLAN)
    assert len(tested["rows"])==2 and all(tested["parity"].values())
    original,mono=(tested["rows"][key] for key in ("original","mono"))
    assert original["actions"]==mono["actions"] and len(mono["actions"])==1222
    assert original["score"]["learner"]==mono["score"]["learner"]==12
    assert mono["observer"]["camera_calls"]==2444
    audit=_read(report.AUDIT)
    assert audit["command_sha256"]==trial.school.digest_json(mono["actions"])
    assert audit["receipt"]["clearance"]["samples"]==9775
    assert not audit["receipt"]["clearance"]["penetration_samples"]
    assert mono["cpu_s"]<original["cpu_s"]
