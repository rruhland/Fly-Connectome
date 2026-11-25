import numpy as np

from scripts.experimental import tt87_reaim_wrist as reaim


def test_optical_rotation_is_proper_and_tracks_original_visibility_cone():
    a=[m for m, note in reaim.views() if note in ([0.,0.,0.],[55,140,150])]
    assert len(a)==2
    for q in a:
        assert np.allclose(q.T@q,np.eye(3)) and np.isclose(np.linalg.det(q),1)
    sample=np.array([[0.,0.,.30],[0.,0.,-.10]])
    assert reaim.screen(sample,a[0],margin=.90).tolist()==[True,False]


def test_strict_mount_photometric_holds_are_disjoint_from_all_fit_frames():
    p=reaim.design()
    d=reaim._read(reaim.wrist.OUT)
    train={r["case"]["id"] for r in d["rows"]}
    held={r["id"] for r in p["held"]}
    assert len(train)==16 and len(held)==len(p["held"])==10
    assert not train & held
    assert np.allclose(p["aperture_m"],[.006,.054,-.035])
    assert p["fovy"]==124.


def test_winner_retains_fixed_aperture_and_bounded_original_contact():
    d=reaim._read(reaim.MODEL); v=d["winner"]
    assert len(v["witnesses"])==d["frames"]
    assert all(w["admissible"] and w["worst"] is None for w in v["witnesses"])
    assert reaim.MOUNT[0]<=v["minimum_mount_gap_m"]<=v["maximum_mount_gap_m"]<=reaim.MOUNT[1]
    assert v["maximum_mount_gap_m"]<0
