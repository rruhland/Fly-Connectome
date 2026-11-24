"""Numerical checks for continuous signed-impact adaptation."""
import numpy as np
from types import SimpleNamespace

from scripts.experimental.tt37_adaptive import Store, body_direction, bouts, ridge_inverse


def test_inversion_reduces_an_observed_velocity_error():
    j=np.diag([.7,1.2,.76])
    error=np.array([-.8,.9,1.9])
    action=ridge_inverse(j,error)
    assert np.linalg.norm(error-j@action)<np.linalg.norm(error)


def test_no_evidence_produces_no_correction():
    s=Store(None)
    action,channel=s.correct(np.zeros(11),np.random.default_rng(5),True)
    assert not np.any(action) and channel=="cold"


def test_visible_passage_can_teach_a_signed_body_adjustment():
    theta=np.linspace(0,2*np.pi,30)
    q=np.c_[.15*np.cos(theta),.15*np.sin(theta),np.zeros((30,3))]
    # An observed joint-to-blade response, with the first two actuators
    # changing horizontal and vertical blade position.
    rows=np.c_[q,q[:,0],np.zeros(30),q[:,1]]
    a=body_direction(np.zeros(5),np.array([.05,-.05]),rows)
    assert a[0]>0>a[1]


def test_opponent_fault_with_no_inbound_stroke_does_not_create_a_sample():
    assert bouts([],[],SimpleNamespace(stereo=[]),{},1/60,[]) == []
