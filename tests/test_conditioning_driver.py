import importlib.util
import json
from pathlib import Path
import sqlite3
from fly_connectome.control.scaling import BLOCKS

spec=importlib.util.spec_from_file_location('conditioning_driver',Path(__file__).parents[1]/'scripts/experimental/run_conditioning_toy.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_fixed_mechanism_allocation_and_independent_seeds():
    jobs=driver.jobs()
    counts={stage:sum(driver.ToySpec(**row['spec']).transition_budget for row in jobs if row['stage']==stage)
            for stage in ('calibration','learn','evaluate')}
    assert counts==dict(calibration=512,learn=1200,evaluate=1920)
    assert len(set(driver.TRAIN+driver.EVAL+driver.RANDOM))==9
    exposure=driver.jobs(200,include_calibration=False,include_baselines=False)
    assert sum(driver.ToySpec(**job['spec']).transition_budget for job in exposure)==10560
    assert all(job['stage']!='calibration' for job in exposure)


def test_calibration_fit_uses_features_without_reward_or_target_labels(tmp_path):
    with sqlite3.connect(tmp_path/'calibration.db') as db:
        db.execute('CREATE TABLE replay_samples(seq INTEGER PRIMARY KEY,record TEXT)')
        record=dict(input_block_norm2={name:float(i+1) for i,name in enumerate(BLOCKS)},reward=1,target_side=0)
        db.executemany('INSERT INTO replay_samples VALUES (?,?)',[(i,json.dumps(record)) for i in range(512)])
    driver.fit(tmp_path)
    before=(tmp_path/'scaling.json').read_bytes()
    with sqlite3.connect(tmp_path/'calibration.db') as db:
        record['reward']=-999;record['target_side']=1
        db.execute('UPDATE replay_samples SET record=?',(json.dumps(record),))
    driver.fit(tmp_path)
    assert (tmp_path/'scaling.json').read_bytes()==before
