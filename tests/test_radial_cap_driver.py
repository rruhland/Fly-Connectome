import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('radial_driver',Path(__file__).parents[1]/'scripts/experimental/run_radial_cap_pong.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_fixed_single_arm_and_numerical_stop_allocation():
    learning,evaluation=driver.declared_jobs()
    assert learning['transitions']==30000 and learning['seed']==172900127
    assert learning['alpha']==.001 and learning['lam']==.995 and learning['opponent_speed']==.25
    assert learning['max_abs_q']==1000 and learning['checkpoints']==[6000,15000,30000]
    assert evaluation['config']['episodes']==24 and evaluation['config']['seed']==904271833
    assert evaluation['weights_from']=='learn_raw_cap_at30000.npy'


def test_paired_development_trigger_checks_both_lifts():
    control=[dict(complete=True,episodes=24,wins=w,own_hit_followed_wins=h) for w,h in ((9,1),(10,2),(9,1),(10,2))]
    report=dict(complete=True,episodes=24,wins=14,own_hit_followed_wins=6)
    assert driver.promising(report,control)
    assert not driver.promising(dict(report,wins=13),control)
    assert not driver.promising(dict(report,own_hit_followed_wins=5),control)
