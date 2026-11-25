import numpy as np

from scripts.experimental import tt95_patch_lessons as lessons
from scripts.experimental.tt4_transfer_trial import _hash,_read


def test_paddle_patch_coordinates_flip_active_face_without_changing_world_point():
    n=np.array([.0,.8,.6])
    t=np.array([.020,-.024,.032])
    uv=lessons.coords(n,t)
    assert np.isclose(n@t,0.)
    np.testing.assert_allclose(lessons.inv.coordinates(n,uv),t)
    np.testing.assert_allclose(lessons.inv.coordinates(-n,uv*[-1,1]),t)
    v=np.array([.05,.72,1.05])
    full=v
    normal=v-lessons.tangent(n,v)
    assert np.isclose(full@n,normal@n) and np.linalg.norm(full-normal)>.39


def test_encoder_alignment_cannot_borrow_future_motion():
    t=[0.,.01,.02,.03,.04]
    q=np.outer(t,[1.,2.,3.,-1.,-2.])
    a=lessons.encoder_at(t,q,.022)
    q[3:]+=10.
    b=lessons.encoder_at(t,q,.022)
    np.testing.assert_allclose(a,b)
    np.testing.assert_allclose(a[1],[1,2,3,-1,-2])


def test_natural_paid_labels_retain_real_patch_motion_and_never_assessment():
    p=_read(lessons.PLAN)
    d=_read(lessons.MOUNTED)
    m=_read(lessons.ROOT/'lesson-receipt.json')
    assert d['complete'] and d['practice']==_hash(lessons.PRACTICE)
    assert m['counts']=={'0':271,'1':59}
    assert m['mounted_encoder_native_median_mps']<.065
    assert {x['id'] for x in d['rows']}.isdisjoint(p['training']['reserve_forbidden'])
    labels=[r for x in d['rows'] for r in x['labels']]
    h=[r for r in labels if r['receive_height'] is not None and r['receive_height']<1.005]
    assert len(h)==8
    vz=[lessons.tangent(np.asarray(r['normal']),np.asarray(r['velocity']))[2] for r in h]
    assert min(vz)<-.18 and max(vz)>.40
