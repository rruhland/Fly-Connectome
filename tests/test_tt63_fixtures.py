import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental import tt63_fixtures as fixtures
from scripts.experimental import tt63_fixture_qualification as qualification


def test_pitched_neck_moves_axis_across_the_grasped_shaft_and_preserves_joint():
    ori=Rotation.from_euler('y',90.,degrees=True).as_quat()[[3,0,1,2]]
    blade=ET.Element('geom', pos='0 0 -.092', size='.076 .006', quat=fixtures._vector(ori))
    p,q=fixtures.neck_geometry(blade,17.)
    axis=Rotation.from_quat(q[[1,2,3,0]]).apply([0.,0.,1.])
    pivot=np.array([0.,0.,fixtures.NECK]); offset=pivot-p
    np.testing.assert_allclose(np.linalg.norm(offset),.060)
    np.testing.assert_allclose(offset@axis,0.,atol=1e-10)
    np.testing.assert_allclose(abs(axis[2]),np.sin(np.deg2rad(17.)))


def test_body_variants_do_not_leak_and_keep_unrelated_compiled_arrays_equal():
    base=fixtures.TT20Arena._model_document
    for variant in ('stock','neck-minus','low-shelf26','stock'):
        with fixtures.mounted(variant):
            a=fixtures.mpc.arena(width=64,height=48,detail='cad')
            try:
                ident=fixtures.identity(a.model,variant)
                if variant=='stock':
                    assert not any(a.model.geom(i).name.startswith('tt29_clamp_tt63_') for i in range(a.model.ngeom))
                    expected=ident
                else:
                    assert fixtures.support_check(a,variant)['safe']
                    assert all(np.array_equal(v,ident[k]) for k,v in expected.items())
            finally:a.close()
        assert fixtures.TT20Arena._model_document is base


def test_held_interior_claim_requires_answer_strict_clearance_and_face():
    result=dict(same=True,landed=True,answered=True,clearance=dict(penetration_samples=0),
                contact=dict(radial_m=.05,face_alignment=.85))
    assert qualification.received(result,interior=True)
    assert not qualification.received(dict(result,answered=False),interior=True)
    assert not qualification.received(dict(result,contact=dict(radial_m=.065,face_alignment=.99)),interior=True)
    assert not qualification.received(dict(result,clearance=dict(penetration_samples=1)),interior=True)
