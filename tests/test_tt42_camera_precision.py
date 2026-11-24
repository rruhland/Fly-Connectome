from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt29_vision import Flight, VERTICAL, flight_terms
from scripts.experimental.tt42_camera_precision import DRAG, DT, Suffix, installed


def camera_spec():
    return dict(eyes=[[-.50,1.2,1.3],[.50,1.2,1.3]],
                rotation=np.broadcast_to([[1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]], (2,3,3)),
                focal=[540.,540.],center=[[383.5,287.5],[383.5,287.5]],dims=[768,576])


def project(camera, point):
    coords = np.einsum('cij,cj->ci',camera['rotation'],point-np.asarray(camera['eyes']))
    return np.asarray(camera['center'])+np.asarray(camera['focal'])[:,None]*coords[:,:2]/coords[:,2:]


def test_precise_causal_rays_recover_a_recent_smooth_arc_and_keep_prior_phase():
    camera=camera_spec()
    fit=Suffix(camera)
    p=np.array([.12,-1.03,.94])
    v=np.array([.40,-3.2,1.28])
    for k in range(-6,1):
        t=k*DT
        d=DRAG*np.linalg.norm(v)
        move,fall=flight_terms(t,d)
        state=p+v*move+VERTICAL*fall
        fit.update(t,project(camera,state),np.r_[state+[.012,-.01,.02],v])
    same=np.r_[p+[.012,-.01,.02],v]
    assert fit.estimate(0.,same,bounced=False) is None
    # The live call has one fixed stock phase per control time.
    fit.last=None
    solution=fit.estimate(0.,same,bounced=True)
    assert solution is not None
    assert np.linalg.norm(solution['pv'][:3]-p)<2e-6
    assert np.linalg.norm(solution['pv'][3:]-v)<5e-6


def test_private_sync_overlay_does_not_change_the_live_estimate_or_bounce_flag():
    stock=Flight(np.array([.0,-1.04,1.01]),np.array([0.,-3.6,1.2]),0.,.6,.2,.02,5,False)
    corrected=np.array([.013,-1.041,1.009,.02,-3.5,1.1])
    used=[]
    eng=SimpleNamespace(twin=SimpleNamespace(control_dt=DT),bounced=True,tick=0,latest=stock)
    eng.observe=lambda *args: None
    eng.sync=lambda: used.append(eng.latest) or True
    eye=SimpleNamespace(step=lambda *args: None, accepted=lambda *args: dict(pv=corrected))
    installed(eng,eye)
    assert eng.sync()
    assert eng.latest is stock and eng.bounced and not used[0].bounced
    assert np.array_equal(used[0].position,corrected[:3])
    # Arithmetic for full interpolation need not reproduce the source bits.
    assert np.allclose(used[0].velocity,corrected[3:])


def test_old_image_pairs_cannot_be_reused_as_a_fresh_fit():
    camera=camera_spec()
    fit=Suffix(camera)
    pv=np.array([.0,-1.05,.95,.1,-3.5,1.2])
    for k in range(6):
        fit.update(k*DT,project(camera,pv[:3]+(k-5)*DT*pv[3:]),pv)
    assert fit.estimate(8*DT,pv,bounced=True) is None
