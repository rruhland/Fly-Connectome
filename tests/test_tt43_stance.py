import numpy as np

from scripts.experimental.tt43_stance import interval, minimax
from scripts.experimental.tt43_stance_chart import joint_coverage
from scripts.experimental.tt43_stance_native import direct_swing


def test_single_pose_timing_bound_retains_different_clocks():
    q=np.array([[0.,0.],[2.,0.],[1.,1.]])
    t=np.array([.4,.7,.5])
    result=minimax(q,t,rate=2.)
    assert np.isclose(result["slack_s"],.05)
    assert np.allclose(result["ready"][0],.7)
    lo,hi=interval(q,t,.051,rate=2.)
    assert np.any(lo>hi)


def test_coverage_uses_best_endpoint_for_each_arc_and_signed_mode():
    hubs=np.array([[0.]])
    arcs=[[{"q":[.416],"eta":.18,"face":-1},{"q":[.832],"eta":.33,"face":1}]]
    assert np.allclose(joint_coverage(hubs,arcs,"eta"),[[.13]])
    assert np.allclose(joint_coverage(hubs,arcs,"eta",lambda r:r["face"]<0),[[.08]])


def test_direct_swing_limits_backswing_and_preserves_gripper():
    q=np.array([[0.,.01,.02,.03,.04]])
    full=np.ones((1,5))
    eta=np.array([.30])
    params=np.zeros((1,13))
    out=direct_swing(q,full,eta,params,np.array([1]),1/60,32)
    assert out.shape==(1,32,6)
    assert np.allclose(out[:,:,5],-.1)
    assert np.allclose(out[0,0,:5],q[0]-.055*.63)
    assert np.allclose(out[0,-1,:5],q[0]+.087*.63)
