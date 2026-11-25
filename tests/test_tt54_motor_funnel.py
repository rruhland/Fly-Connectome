import numpy as np

from scripts.experimental.tt54_motor_funnel import physical_goals
from scripts.experimental.tt54_executable_seeds import executable


def test_future_goals_are_matched_only_after_consuming_bounded_slew():
    goals=np.broadcast_to([.4,-.4],(2,5,2)).copy()
    goals[1,0]=[.02,-.02]
    result=physical_goals(goals,np.zeros(2),np.array([.08,.1]),np.array([-.196,-.24]),np.array([.196,.24]))
    np.testing.assert_allclose(result[0,:,0],[.064,.128,.192,.196,.196])
    np.testing.assert_allclose(result[0,:,1],[-.08,-.16,-.24,-.24,-.24])
    np.testing.assert_allclose(result[1,0],[.02,-.02])
    np.testing.assert_allclose(result[1,4],result[0,4])


def test_recentering_an_absolute_path_keeps_issued_microstep_references():
    rng=np.random.default_rng(4382)
    raw=rng.uniform(-2.,2.,size=(7,18,2))
    ref=np.array([.24,-.11]); inc=np.array([.0101,.0083])
    lo=np.array([-.21,-.33]); hi=np.array([.47,.62])
    result=executable(raw,ref,inc,lo,hi,8)
    np.testing.assert_allclose(executable(result,ref,inc,lo,hi,8),result,rtol=0.,atol=6e-15)
    # Not a cosmetic relabel: the center that is fed to the perturbation is
    # materially closer to the physically issued first reference.
    assert (np.abs(np.clip(raw,lo,hi)-result)>.2).sum()>20
