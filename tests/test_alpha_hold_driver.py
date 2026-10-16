import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('alpha_hold',Path(__file__).parents[1]/'scripts/experimental/run_alpha_hold_pong.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def report(wins=14,hits=6):return dict(complete=True,episodes=24,wins=wins,own_hit_followed_wins=hits)
def controls():return [report(w,h) for w,h in ((9,1),(10,2),(9,1),(10,2))]

def test_control_maxima_are_independent_and_raise_each_gate():
    strongest=[report(13,1),report(9,5),report(8,0),report(10,2)]
    assert driver.thresholds(strongest)==dict(total=17,own_hit=9)
    result=driver.choose_arm({'hold4':report(17,8)},{'hold4':strongest})
    assert result['selected_arm'] is None
    result=driver.choose_arm({'hold4':report(17,9)},{'hold4':strongest})
    assert result['selected_arm']=='hold4'

def test_only_complete_qualifying_endpoints_compete_and_ties_choose_simpler():
    groups={arm:controls() for arm in driver.ARMS}
    assert driver.choose_arm({arm:report() for arm in driver.ARMS},groups)['selected_arm']=='low_alpha'
    assert driver.choose_arm({'hold4':report(15) ,'low_alpha':report()},groups)['selected_arm']=='hold4'
    assert driver.choose_arm({'hold4':dict(report(20),complete=False)},groups)['selected_arm'] is None
