import numpy as np

from scripts.experimental import tt93_open_wrist_fixture as mount
from scripts.experimental import tt93_wrist_course as cue


def test_triangles_from_raw_camera_rays_need_a_baseline():
    target=np.array([.06,-.98,1.014])
    eyes=(np.array([1.,0.,1.6]),np.array([.06,-1.43,1.06]),np.array([-1.21,.1,1.4]))
    rays=[(p,(target-p)/np.linalg.norm(target-p)) for p in eyes]
    recovered,_=cue.triangulate(rays)
    assert np.allclose(recovered,target,atol=1.e-8)
    assert cue.triangulate([rays[0],rays[0]]) is None


def test_reserved_actual_game_launches_disjoint_from_earlier_sensor_frames():
    p=cue._read(cue.PLAN)
    scout={r["card"]["id"] for r in p["scout"]}
    points={r["card"]["id"] for r in p["future_full_points"]}
    assert not scout & points and len(points)==14
    assert (scout|points).isdisjoint(p["blocked"])


def test_drawing_does_not_change_the_default_sensor_geometry():
    d=cue._read(mount.MODEL)
    a=mount.mpc.arena(width=64,height=48)
    try:
        assert np.allclose(a.model.body(mount.CAMERA_BODY).pos,mount.STOCK)
        assert a.model.geom("tt29_camera_bracket").id>=0
    finally:a.close()
    if d["winner"] is None:
        return
    w=d["winner"]; spec=w["spec"]
    with mount.builder(spec):
        b=mount.mpc.arena(width=64,height=48)
    try:
        assert np.allclose(b.model.body(mount.CAMERA_BODY).pos,spec["aperture"])
        assert np.isclose(b.model.camera(mount.WRIST_CAMERA).fovy[0],124.)
        assert np.isclose(b.model.body_mass[b.model.body(mount.CAMERA_BODY).id],.015)
        assert b.model.body_mass[b.model.body("learner_wrist_camera_mount").id]>.002
        assert len([b.model.geom(i).name for i in range(b.model.ngeom) if b.model.geom(i).name.startswith("tt93_rear_strut_")])==4
        assert all(b.model.geom(i).name!="tt29_camera_bracket" for i in range(b.model.ngeom))
    finally:b.close()
