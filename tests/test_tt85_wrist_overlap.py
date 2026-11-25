import numpy as np

from scripts.experimental import tt85_wrist_overlap as overlap


def example(lead,match=True,clipped=False,blade=120,postbounce=True):
    return dict(lead_s=lead,match=match,sight={"clipped":clipped},
                blade_pixels=blade,inbound_after_bounce=postbounce,
                camera_ray_screen=True,encoder_bearing_m=.002)


def test_bearing_error_only_measures_transverse_offset():
    cam=overlap.Lens(384,288,124.).at(np.zeros(3),np.eye(3))
    center=cam.center
    assert overlap.angular_error(cam, center, np.array([0.,0.,-.20]))<1.e-9
    assert np.isclose(overlap.angular_error(cam, center, np.array([.006,0.,-.20])),.006)


def test_watchdog_requires_two_usable_predeadline_frames():
    base=[example(.08),example(.01),example(.10,clipped=True),example(.11,postbounce=False)]
    assert not overlap.per_shot({'rows':base})[1]['useful']
    _, got=overlap.per_shot({'rows':base+[example(.10)]})
    assert got['useful'] and got['ready_with_blade'] and got['encoder_bearing_mm']==2
    assert got['matched_early']==2


def test_receipt_pins_distinct_ordinary_body_examples_and_current_fixture():
    p=overlap._read(overlap.PLAN)
    assert p['bounds']['resolution']==[384,288] and p['bounds']['rate_hz']==60
    rows=p['assignments']
    assert len({r['id'] for r in rows})==len(rows)==16
    assert {r['id'] for r in rows}.isdisjoint(p['forbidden'])
    assert all(r['card']['opening']=='random' and r['card']['profile']=='reach' for r in rows)
