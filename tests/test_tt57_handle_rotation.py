import numpy as np
import pytest

from scripts.experimental.tt38_joint_mpc import arena
from scripts.experimental.tt57_handle_rotation import proof_model, rotate, shape_relation


def test_handle_roll_preserves_physics_and_has_a_stock_signed_equivalent():
    a=arena(width=64,height=48)
    b=arena(width=64,height=48)
    try:
        identity=proof_model(a)
        initial=a.model.geom('learner_racket').quat.copy()
        counts=rotate(a,45.)
        assert counts['mass_kg']==pytest.approx(.115)
        assert proof_model(a)==identity
        assert not np.array_equal(a.model.geom('learner_racket').quat,initial)
        equivalents=shape_relation(a,b,45.,b._arm_home(1)[0])
        assert min(x['optical_face_match'] for x in equivalents)>1.-1e-12
        assert max(x['offset_center_m'] for x in equivalents)<.015
    finally:
        a.close();b.close()
