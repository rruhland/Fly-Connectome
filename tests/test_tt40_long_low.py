import numpy as np

from scripts.experimental.tt40_long_low import ease, preparation, same_trace


def test_prep_reference_hits_both_knots_with_fixed_grip_and_monotone_motion():
    first, waypoint, end = np.zeros(5), np.ones(5)*.4, np.ones(5)
    result=preparation(first,waypoint,end,53,-.10)
    split=round(.47*53)
    np.testing.assert_allclose(result[split-1,:5],waypoint)
    np.testing.assert_allclose(result[-1,:5],end)
    np.testing.assert_allclose(result[:,5],-.10)
    assert (np.diff(result[:,:5],axis=0)>0.).all()
    np.testing.assert_allclose(ease(np.array([-1.,0.,.5,1.,2.])),[0.,0.,.5,1.,1.])


def test_event_equivalence_allows_serialization_and_roundoff_not_actual_contact_changes():
    a=[dict(kind='contact',side=-1,location=(np.float64(.4),-.7,.88),time=np.float64(4.))]
    b=[dict(kind='contact',side=-1,location=[.4,-.7,.88+1e-10],time=4.)]
    assert same_trace(a,b)
    b[0]['location'][-1]+=.002
    assert not same_trace(a,b)
    b[0]['location'][-1]-=.002
    b[0]['side']=1
    assert not same_trace(a,b)
