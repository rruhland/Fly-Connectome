import numpy as np

from scripts.experimental.tt51_partial_finish import Finish, Readiness, body_envelope


def motion(y,v):
    return np.array([.038,y,1.03,.04,v,.50])


def test_partial_preparation_waits_for_played_outbound_and_returns_to_stock_after_cue():
    ready=Readiness()
    q=np.array([0.,-.8,-1.7,.4,-.6,0.])
    ready.observe(0,motion(-.9,2.7),q,q)
    assert not np.any(ready.correction(0,motion(-.9,2.7),q))
    ready.observe(1,motion(-.95,-3.7),q,q)
    for t in range(2,29):
        v=motion(min(1.,-.98+.061*(t-2)),2.7)
        ready.observe(t,v,q,q)
    assert np.all(ready.correction(28,v,q)<=body_envelope)
    assert ready.correction(28,v,q)[2]>.32
    x=motion(.6,-3.7)
    ready.observe(29,x,q,q)
    assert ready.last_inbound==29
    x=motion(-.45,-3.7)
    ready.observe(48,x,q,q)
    assert not np.any(ready.correction(48,x,q))


def test_skill_uses_current_encoder_body_even_for_same_observed_ball():
    q=np.array([-.1,-.8,-1.6,.45,-.59]);p=np.array([-.07,.988,.04,-3.74,.35]);z=np.zeros(5)
    r=dict(id=101,partial=dict(skill=dict(contact_y=-1.29,phases=[-18,0,8],refs=[[0.]*6]*3,
           samples=[dict(phase=-12,future=p.tolist(),q=q.tolist(),dq=z.tolist(),lag=z.tolist())])))
    field=Finish(dict(rows=[r]))
    path=np.zeros((2,10));path[0,:7]=np.r_[.146,p[0],-1.10,p[1:]];path[1,:7]=np.r_[.20,p[0],-1.29,p[1:]]
    shots,ids=field.proposals(motion(-.43,-3.74),path,(q,z,z))
    assert len(shots)==3 and ids==[101]*3
    shots,ids=field.proposals(motion(-.43,-3.74),path,(q+2.,z,z))
    assert not shots and not ids
