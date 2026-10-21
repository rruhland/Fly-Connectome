"""Stationary task validation, historical population equivalence, durable phase boundaries."""
from pathlib import Path
import json,sqlite3,sys,time
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_stationary_curriculum as driver
from policy_search_math import SearchTransform
from fly_connectome.control.environment import PongCamera
from fly_connectome.control.pong_evaluation import PongEvaluationConfig


def test_stationary_legal_and_default_unchanged_negative_nonfinite_rejected():
    stationary=PongCamera(719283,opponent_speed=0.)
    PongEvaluationConfig(719283,opponent_speed=0.).validate()
    for action in [0,2,1]*20:
        if stationary.needs_reset:stationary.reset()
        stationary.step(action)
        assert float(stationary.game.opponent[0])==.5
        assert float(stationary.game.opponent_velocity[0])==0.
    assert PongCamera().game.config.opponent_speed==.45
    for speed in [-.01,float('nan'),float('inf'),-float('inf')]:
        with pytest.raises(ValueError):PongCamera(opponent_speed=speed)
        with pytest.raises(ValueError):PongEvaluationConfig(7,opponent_speed=speed).validate()


def test_exact_historical_first_population_and_stationary_only_change(tmp_path,monkeypatch):
    needed=[driver.BRIDGE,driver.PRIOR_BRIDGE,driver.CACHE,driver.ORIGINAL/'g00-definition.json',driver.ORIGINAL/'search-transform.npy']
    if not all(p.exists() for p in needed):pytest.skip('historical population integration requires preserved local calibration/lineage/source proofs')
    bridge=json.loads(driver.BRIDGE.read_text());ancestor=json.loads((driver.ORIGINAL/'manifest.json').read_text())
    actual_pins=driver.pins
    current=actual_pins()
    if current['source']!=bridge['source'] or current['math_hash']!=ancestor['math_hash']:
        with pytest.raises(ValueError,match='baseline source/projection chain'):
            driver.prepare(tmp_path/'reject-current-source')
    # Recreate only this historical fixture's archived source context. New
    # generic helper defaults must still reproduce its population exactly;
    # production validation above remains strict for the actual current source.
    def archived_pins():
        result=actual_pins();result.update(source=bridge['source'],math_hash=ancestor['math_hash'])
        return result
    monkeypatch.setattr(driver,'pins',archived_pins)
    manifest,transform,z,theta,rng=driver.prepare(tmp_path)
    directory,state,theta=driver.initialize_phase(tmp_path,driver.PHASES[0],theta,rng,0,manifest['started_wall_ns'])
    definition,directions=driver.prepare_generation(directory,transform,z,state,theta,0,0.)
    old=json.loads((driver.ORIGINAL/'g00-definition.json').read_text())
    assert definition['training_serve_seeds']==old['training_serve_seeds']
    assert definition['rng_state_after_draw']==old['rng_state_after_draw']
    np.testing.assert_array_equal(directions,np.load(driver.ORIGINAL/'g00-directions.npy'))
    for job,previous in zip(definition['jobs'],old['jobs']):
        assert {k:v for k,v in job.items() if k!='config'}=={k:v for k,v in previous.items() if k!='config'}
        assert [k for k in job['config'] if job['config'][k]!=previous['config'][k]]==['opponent_speed']
        assert job['config']['opponent_speed']==0.
        np.testing.assert_array_equal(np.load(directory/job['weights_from']),np.load(driver.ORIGINAL/previous['weights_from']))


def fixture(directory):
    directory.mkdir();driver.atomic_json(directory/'manifest.json',dict(jobs=[]))
    matrix=np.eye(2,4);theta=np.array([[.2,-.1],[-.3,.5],[.1,-.4]]);z=np.eye(2)
    driver.immutable_array(directory/'search-transform.npy',matrix)
    driver.weights_file(directory,'policy_at0.npy',theta@matrix)
    state=dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=driver.array_hash(theta),rng_state=np.random.default_rng(8173).bit_generator.state,initial_policy_hash=driver.sha(directory/'policy_at0.npy'))
    driver.immutable_array(directory/'theta_at0.npy',theta);driver.atomic_json(directory/'search-state.json',state)
    return SearchTransform(matrix,{}),z,state,theta


def test_partial_population_never_updates_and_postdraw_rng_transfers(tmp_path,monkeypatch):
    directory=tmp_path/'partial';transform,z,state,theta=fixture(directory)
    def incomplete(directory,jobs,pool,**kwargs):
        with sqlite3.connect(directory/(jobs[0]['id']+'.db')) as c:
            c.execute('CREATE TABLE replay_samples(seq INTEGER,record TEXT)')
            c.executemany('INSERT INTO replay_samples VALUES(?,?)',[(i,'{}') for i in range(5)])
        return dict(complete=False,failed=False,committed_cameras=5,unfinished=[j['id'] for j in jobs])
    monkeypatch.setattr(driver.engine,'execute_jobs',incomplete)
    frozen,result=driver.run_phase(directory,driver.PHASES[0],transform,z,state,theta,0,None,time.time()+60)
    assert frozen['completed_generations']==0 and not frozen['valid'] and frozen['unused_cameras']==5
    assert frozen['used_cameras']==0 and frozen['unused_draw_skipped'] and frozen['next_global_generation']==1
    definition=json.loads((directory/'g00-definition.json').read_text())
    assert frozen['next_rng_state']==definition['rng_state_after_draw']!=state['rng_state']
    assert not (directory/'theta_at1.npy').exists();np.testing.assert_array_equal(result,theta)
    # Recovery reuses the persisted endpoint/phase timestamp and never reruns work.
    monkeypatch.setattr(driver.engine,'execute_jobs',lambda *a,**k:pytest.fail('frozen phase must not execute'))
    again,_=driver.run_phase(directory,driver.PHASES[0],transform,z,state,theta,0,None,time.time()+1000)
    assert again==frozen


def test_unexecuted_pending_draw_is_skipped_and_phase_anchor_cannot_reset(tmp_path,monkeypatch):
    directory=tmp_path/'empty';transform,z,state,theta=fixture(directory)
    monkeypatch.setattr(driver.engine,'execute_jobs',lambda *a,**k:pytest.fail('expired budget must not execute'))
    frozen,_=driver.run_phase(directory,driver.PHASES[0],transform,z,state,theta,4,None,time.time()-1)
    assert frozen['observed_cameras']==0 and frozen['unused_cameras']==0
    assert frozen['unused_draw_skipped'] and frozen['next_global_generation']==5
    root=tmp_path/'root';root.mkdir();driver.immutable_array(root/'search-transform.npy',transform.matrix)
    (root/'radial-cap.json').write_text('{}')
    monkeypatch.setattr(driver,'pins',lambda:dict(source={},runtime={},driver_hash='fixture'))
    phase,next_state,value=driver.initialize_phase(root,driver.PHASES[1],theta,frozen['next_rng_state'],5,frozen['frozen_wall_ns'])
    assert next_state['rng_state']==frozen['next_rng_state']
    np.testing.assert_array_equal(value,theta)
    np.testing.assert_array_equal(np.load(phase/'policy_at0.npy'),theta@transform.matrix)
    saved=json.loads((phase/'manifest.json').read_text())
    assert saved['global_generation_offset']==5 and saved['phase_started_wall_ns']==frozen['frozen_wall_ns']
    driver.initialize_phase(root,driver.PHASES[1],theta,frozen['next_rng_state'],5,frozen['frozen_wall_ns'])
    with pytest.raises(ValueError,match='phase lineage/budget'):
        driver.initialize_phase(root,driver.PHASES[1],theta,frozen['next_rng_state'],5,frozen['frozen_wall_ns']+1)


def test_complete_phase_boundary_retains_rng_without_extra_skip(tmp_path):
    directory=tmp_path/'complete';transform,z,state,theta=fixture(directory)
    state['completed_generations']=6
    rng,index,skipped=driver.next_boundary(directory,state,0)
    assert rng==state['rng_state'] and index==6 and not skipped
    assert driver.engine.control_gate([dict(wins=10,own_hit_followed_wins=2)])==dict(total=14,own_hit=6)
