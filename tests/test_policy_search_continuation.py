"""The second allocation inherits exact optimizer state without reusing partial fitness."""
from pathlib import Path
from concurrent.futures import Future
import json,sys,time
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pong_policy_search_continuation as driver
from policy_search_math import generation_spec


def test_exact_lineage_skipped_draw_and_pending_generation_recovery(tmp_path,monkeypatch):
    required=[driver.LINEAGE/'theta_at7.npy',driver.LINEAGE/'g07-definition.json',driver.SOURCE_BRIDGE,driver.CACHE]
    if not all(path.exists() for path in required):
        pytest.skip('historical optimizer integration requires preserved local lineage/calibration/source-bridge artifacts')
    # This optimizer-lineage fixture intentionally recreates the archived source context.
    # Production prepare still fails closed when actual source and bridge disagree.
    frozen_source=json.loads(driver.SOURCE_BRIDGE.read_text())['source']
    monkeypatch.setattr(driver,'source_signature',lambda:frozen_source)
    manifest,transform,z,state,theta=driver.prepare(tmp_path)
    np.testing.assert_array_equal(theta,np.load(driver.LINEAGE/'theta_at7.npy'))
    np.testing.assert_array_equal(np.load(tmp_path/'policy_at0.npy'),np.load(driver.LINEAGE/'policy_at7.npy'))
    skipped=json.loads((driver.LINEAGE/'g07-definition.json').read_text())
    assert state['rng_state']==skipped['rng_state_after_draw']
    assert state['rng_state']!=skipped['input_rng_state']
    assert manifest['workers']==4 and manifest['previous_training_cameras_unused']==3814
    assert manifest['maximum_cumulative_training_cameras']==179903
    definition,directions=driver.prepare_generation(tmp_path,transform,z,state,theta)
    assert definition['generation']==0 and definition['global_generation']==8
    expected=generation_spec(theta,state['rng_state'],8,z,training_seed=driver.TRAINING_SEED)
    np.testing.assert_array_equal(directions,expected[0])
    assert definition['training_serve_seeds']==expected[2]
    old_seeds={seed for g in range(8) for seed in json.loads((driver.LINEAGE/f'g{g:02d}-definition.json').read_text())['training_serve_seeds']}
    assert not old_seeds.intersection(definition['training_serve_seeds'])
    reports={j['id']:dict(complete=True,episodes=1,total_reward=float(i%3-1),transitions=17) for i,j in enumerate(definition['jobs'])}
    missing=definition['jobs'][-1]['id'];saved=reports.pop(missing)
    monkeypatch.setattr(driver,'complete_report',lambda directory,name:reports.get(name))
    with pytest.raises(ValueError,match='partial generation'):driver.finish_generation(tmp_path,definition,directions,theta,z)
    np.testing.assert_array_equal(np.load(tmp_path/'theta_at0.npy'),theta)
    assert not (tmp_path/'theta_at1.npy').exists()
    recovered,again=driver.prepare_generation(tmp_path,transform,z,state,theta)
    assert recovered==definition;np.testing.assert_array_equal(again,directions)
    reports[missing]=saved
    updated,weights,result=driver.finish_generation(tmp_path,definition,directions,theta,z)
    second,weights2,result2=driver.finish_generation(tmp_path,definition,directions,theta,z)
    assert updated==second and result==result2 and updated['completed_generations']==1
    assert result['global_generation']==8
    np.testing.assert_array_equal(weights,weights2)


def test_original_controls_define_gate_not_continuation_start():
    controls=[dict(wins=10,own_hit_followed_wins=2),dict(wins=8,own_hit_followed_wins=0)]
    assert driver.control_gate(controls)==dict(total=14,own_hit=6)
    assert driver.control_gate(controls+[dict(wins=11,own_hit_followed_wins=5)])==dict(total=15,own_hit=9)
    # Theta7's12/4 is descriptive, deliberately absent from the matched controls.
    assert driver.GLOBAL_OFFSET+driver.GENERATIONS-1==23
    assert driver.INHERITED_UPDATES+driver.GENERATIONS==23


def test_four_worker_camera_reservations_and_partial_grant(tmp_path,monkeypatch):
    observed={};reports={};round_sizes=[]
    class Pool:
        def submit(self,fn,directory,name,grant,work):
            result=Future();observed[name]=observed.get(name,0)+grant
            reports[name]=dict(complete=True)
            result.set_result(dict(job=name,ok=True,result={},wall_seconds=0.))
            return result
    actual_wait=driver.wait
    def wait(active,**kwargs):round_sizes.append(len(active));return actual_wait(active,**kwargs)
    monkeypatch.setattr(driver,'wait',wait)
    monkeypatch.setattr(driver,'sample_count',lambda directory,name:observed.get(name,0))
    monkeypatch.setattr(driver,'read_report',lambda directory,name:reports.get(name))
    monkeypatch.setattr(driver,'complete_report',lambda directory,name:reports.get(name))
    monkeypatch.setattr(driver,'pin_completion',lambda directory,name:None)
    result=driver.execute_jobs(tmp_path,[dict(id=str(i)) for i in range(5)],Pool(),deadline=time.time()+60,camera_cap=4500)
    assert round_sizes[0]==4 and max(round_sizes)==4
    assert result['complete'] and sum(observed.values())==4400 and result['committed_cameras']==4400
