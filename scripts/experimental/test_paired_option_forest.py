"""Ordering, actual reward windows, exported tree parity and reward-gated holds."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from pong_paired_option_forest import horizontal_scene,features,OptionForest,ForestController,window_end


def test_horizontal_order_preserves_yx_and_ignores_entity_input_order():
    rows=np.array([[.1,.8,0,1,1,1],[.9,.2,2,0,1,1],[.2,.2,0,0,0,0]])
    expected=rows[[2,1,0]];np.testing.assert_array_equal(horizontal_scene(rows),expected)
    np.testing.assert_array_equal(features(rows,[3,2,1],4),features(rows[::-1],[3,2,1],4))
    f=features(rows,[3,2,1],4);np.testing.assert_array_equal(f[:18],expected.ravel());assert f[-4:].tolist()==[3,3,2,1]
    assert features(np.zeros((5,6)),[0,0,0],4) is None


def test_export_matches_sklearn_exact_values_actions_and_ties():
    rng=np.random.default_rng(2026102902);F=rng.normal(size=(48,32));Y=rng.integers(-1,2,size=(48,3)).astype(float)
    model=ExtraTreesRegressor(n_estimators=64,max_depth=5,min_samples_leaf=2,max_features=1.,bootstrap=False,random_state=2026102902,n_jobs=1).fit(F,Y)
    forest=OptionForest.export(model,4);queries=np.r_[F,rng.normal(size=(2048,32))];expected=model.predict(queries);actual=forest.predict(queries)
    assert np.array_equal(actual,expected) and np.array_equal(np.argmax(actual,1),np.argmax(expected,1))
    zero=ExtraTreesRegressor(n_estimators=64,max_depth=5,min_samples_leaf=2,random_state=2026102902,n_jobs=1).fit(F,np.zeros((48,3)))
    z=OptionForest.export(zero,4);assert z.inspect(np.zeros((1,6)),[0,0,0])['option']==3


def test_first_actual_hit_gates_exact16camera_hold_and_baseline_reconsideration():
    class Forest:
        def inspect(self,scene,q):return dict(option=2,advantages=[0,0,1],representation_supported=True)
    c=ForestController(Forest());assert c.choose([],0,[0,0,0])[0]==0;c.observe_reward(0);assert c.choose([],1,[0,0,0])[0]==1
    c.observe_reward(1);assert c.choose([],0,[0,0,0])[1]['mode']=='earned_start'
    for i in range(15):assert c.choose([],1,[0,0,0])[1]['mode']=='earned_hold'
    assert c.remaining==0 and c.choose([],0,[0,0,0])[1]['mode']=='earned_start'


def test_binary_window_end_is_distinct_from_terminal_or_censored():
    r=dict(hits=0,terminated=False,truncated=True)
    assert window_end(r,159)==('environment_timeout_censored',False,None)
    assert window_end(r,160)==('full_window_no_hit',True,0.)
    assert window_end(dict(r,hits=1,terminated=True),160)==('own_hit',True,1.)
    assert window_end(dict(r,truncated=False,terminated=True),40)==('natural_termination',True,0.)
    assert window_end(dict(r,truncated=False),40) is None
