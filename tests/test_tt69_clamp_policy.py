import numpy as np

from scripts.experimental import tt69_clamp_policy as policy


def test_same_body_lesson_requires_observed_arc_and_enough_joint_travel_time():
    incoming = np.array([[.27, -.05, policy.AXIS, 1.00, .15, -2.9, .22, 0., 0., 0.]])
    card = dict(id=8, leg=0, key=policy.marker(incoming[0]).tolist(),
                anchor=policy.marker(incoming[0]).tolist(), y=[.09, .40],
                goal=[[.15]*6, [.34]*6], q=[[.0]*6, [.2]*6], radius=.025,
                depth=policy.AXIS, contact_goal=[.5]*6, contact_q=[.33]*6)
    bank=policy.Field(dict(rows=[card]))
    x=np.array([-.04, -.2, .95, .15, -2.9, .22])
    assert bank.proposals(incoming,x,np.zeros(6),np.zeros(6))[0][1]['id']==8
    assert not bank.proposals(incoming,x,np.zeros(6),np.full(6,-1.0))
    assert not bank.proposals(np.empty((0,10)),x,np.zeros(6),np.zeros(6))


def test_scene_landmark_uses_observed_pv_instead_of_exact_contact_tick():
    arc=np.array([[.1,.24,-.61,.90,.04,-2.7,.33,0,0,0],
                  [.3,.26,-1.44,.93,.05,-2.6,.19,0,0,0]])
    pick=policy.point_at(arc,-1.45)
    np.testing.assert_equal(policy.marker(pick),[.26,.93,.05,-2.6,.19])
    assert policy.point_at(arc,-1.66) is None
