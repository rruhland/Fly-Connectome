from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experimental import tt54_opponent_approx as approximation
from scripts.experimental.tt29_fast_camera import CachedStereoRenderer
from scripts.experimental.tt54_packed_stereo import PackedStereo, Unsupported


def test_approximate_partner_scores_are_bitwise_equal_on_finite_and_fault_rows():
    rng=np.random.default_rng(5401)
    n=486
    p=np.empty((3,n))
    p[:]=rng.uniform([-.65,.45,.94],[.65,1.75,1.45],size=(n,3)).T
    v=rng.uniform([-.9,-7.,1.3],[.9,-3.8,4.0],size=(n,3))
    w=rng.uniform(-110.,110.,size=(n,3))
    target=np.array([.115,-1.275,1.065,3.58])
    p=p.T
    assert not p.flags.c_contiguous
    old=approximation.BASE(p,v,w,target)
    fast=approximation.approximate(p,v,w,target)
    assert np.isfinite(old).sum()>10
    assert np.isinf(old).sum()>10
    assert np.array_equal(old.view(np.uint64),fast.view(np.uint64))
    assert np.array_equal(np.argsort(old),np.argsort(fast))


def test_nonstock_numpy_inputs_use_unmodified_original(monkeypatch):
    monkeypatch.setattr(approximation,"BASE",lambda *args:"stock")
    p=np.zeros((12,3),np.float32)
    assert approximation.approximate(p,p,p,np.zeros(4))=="stock"
    p=p.astype(float); p[0,0]=np.nan
    assert approximation.approximate(p,p,p,np.zeros(4))=="stock"


def test_unreviewed_render_configuration_falls_back(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL","egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM","egl")
    camera=PackedStereo.__new__(PackedStereo)
    camera.arena=SimpleNamespace(width=192,height=144,detail="preview",streams=("left","right"))
    camera.renderer=SimpleNamespace()
    with pytest.raises(Unsupported):
        camera.enable()
    camera.hardware=None
    monkeypatch.setattr(CachedStereoRenderer,"capture",lambda self: "stock")
    assert camera.capture()=="stock"


def test_normal_opt_in_scopes_changes_and_does_not_replace_another_variant(monkeypatch):
    from scripts.experimental import tt29_partner, tt33_joint
    from scripts.experimental.tt54_training_speedups import enabled
    assert tt33_joint.CachedStereoRenderer is CachedStereoRenderer
    assert tt29_partner._flight_approx is approximation.BASE
    with enabled():
        assert tt33_joint.CachedStereoRenderer is PackedStereo
        assert tt29_partner._flight_approx is approximation.approximate
    assert tt33_joint.CachedStereoRenderer is CachedStereoRenderer
    prior=lambda *args:"other controller"
    monkeypatch.setattr(tt29_partner,"_flight_approx",prior)
    with enabled(cameras=False):
        assert tt29_partner._flight_approx is prior
        assert tt33_joint.CachedStereoRenderer is CachedStereoRenderer
