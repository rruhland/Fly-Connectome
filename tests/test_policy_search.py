"""Mapping, score-only optimizer, and durable search-boundary contracts."""
from pathlib import Path
import copy,hashlib,json,random,sqlite3,sys
from concurrent.futures import Future
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import policy_search_math as math
import run_pong_policy_search as driver
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.control.capping import cap_values
from fly_connectome.vision import VisualStateEncoder


@pytest.fixture(scope='module')
def fitted():
    if not driver.CACHE.exists():
        pytest.skip('historical transform integration requires the preserved score-free calibration cache')
    encoder=VisualStateEncoder(64,64,.02)
    raw=driver.read_calibration(driver.CACHE)
    transform,calibration=math.fit_transform(raw,encoder.layout,encoder.dimension)
    return raw,transform,calibration


def test_generic_mapping_scores_blank_bias_and_positive_cap(fitted):
    raw,transform,z=fitted
    rng=np.random.default_rng(7165271);theta=rng.standard_normal((3,214));weights=transform.mapped_weights(theta)
    expected=z@theta.T;actual=np.asarray(raw@weights.T)
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=2e-12)
    np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
    np.testing.assert_array_equal(transform.matrix@np.zeros(4389),np.zeros(214))
    assert transform.diagnostics['projected_bias_energy']==1.
    assert transform.diagnostics['maximum_whitening_gain']<=10.
    assert transform.diagnostics['free_action_contrasts']==2*transform.diagnostics['supported_rank']
    # Existing full policy sees identical greedy decisions after radial scaling.
    policy=FrozenLinearPolicy(weights)
    for row in [0,91,512,1023]:
        x=raw[row];indices=x.indices;values=x.data
        a,q=policy.choose(indices,values)
        ci,cv=indices,cap_values(values,256.)
        b,cq=policy.choose(ci,cv)
        assert a==b==int(expected[row].argmax())
        if np.dot(values,values)<=256:np.testing.assert_array_equal(q,cq)


def test_mirrored_update_exact_direction_zero_variance_and_ties():
    calibration=np.eye(2);theta=np.zeros((3,2))
    directions=np.array([[[1.,0.],[-1.,0.],[0.,0.]],[[0.,1.],[0.,-1.],[0.,0.]]])
    returns=np.array([[1.,-1.],[0.,0.]])
    actual,info=math.mirrored_update(theta,directions,returns,calibration,top=2,step_size=.1)
    expected=.1/(2*np.std(returns))*2*directions[0]
    if math.function_rms(expected,calibration)>.1:expected*=.1/math.function_rms(expected,calibration)
    np.testing.assert_array_equal(actual,expected)
    for values in [np.zeros((2,2)),np.ones((2,2)),np.array([[1.,1.],[-1.,-1.]])]:
        result,record=math.mirrored_update(theta,directions,values,calibration,top=2)
        np.testing.assert_array_equal(result,theta)
    _,record=math.mirrored_update(theta,directions,np.zeros((2,2)),calibration,top=1)
    assert record['selected_directions']==[0] and record['zero_update']
    with pytest.raises(ValueError,match='full finite'):math.mirrored_update(theta,directions,[[1.,None],[0.,0.]],calibration)


def test_rotation_and_rng_recovery_are_local_and_reproducible(fitted):
    _,_,z=fitted;rng=np.random.default_rng(driver.OPTIMIZER_SEED)
    initial=.1*math.sample_directions(rng,z,1)[0];state=copy.deepcopy(rng.bit_generator.state)
    first=math.generation_spec(initial,state,0,z)
    np.random.seed(92817);np.random.random(37)
    again=math.generation_spec(initial,state,0,z)
    for a,b in zip(first[:2],again[:2]):np.testing.assert_array_equal(a,b)
    assert first[2:]==again[2:]
    np.testing.assert_allclose(first[1][:,0]+first[1][:,1],np.broadcast_to(2*initial,(8,3,214)),rtol=0,atol=1e-14)
    assert all(abs(math.function_rms(d,z)-1)<1e-12 for d in first[0])
    serves=[s for g in range(8) for s in math.generation_spec(initial,state,g,z)[2]]
    assert len(set(serves))==32 and not set(serves)&{driver.DEV,*driver.FINAL,849375169}
    assert first[2]!=math.generation_spec(initial,state,0,z,training_seed=736187003)[2]


def test_pending_generation_recovery_never_updates_partial_or_redraws(tmp_path,monkeypatch,fitted):
    _,transform,z=fitted
    monkeypatch.setattr(driver,'register_jobs',lambda directory,jobs:None)
    theta=np.zeros((3,214));rng=np.random.default_rng(782713)
    state=dict(completed_generations=0,rng_state=rng.bit_generator.state)
    driver.immutable_array(tmp_path/'search-transform.npy',transform.matrix)
    driver.weights_file(tmp_path,'policy_at0.npy',transform.mapped_weights(theta))
    definition,directions=driver.prepare_generation(tmp_path,transform,z,state,theta)
    reports={job['id']:dict(complete=True,episodes=1,total_reward=float((i%3)-1),transitions=17,hits=999) for i,job in enumerate(definition['jobs'])}
    last=definition['jobs'][-1]['id'];saved=reports.pop(last)
    monkeypatch.setattr(driver,'complete_report',lambda directory,name:reports.get(name))
    with pytest.raises(ValueError,match='partial generation'):driver.finish_generation(tmp_path,definition,directions,theta,z)
    assert not (tmp_path/'search-state.json').exists() and not (tmp_path/'theta_at1.npy').exists()
    recovered,again=driver.prepare_generation(tmp_path,transform,z,state,theta)
    assert recovered==definition;np.testing.assert_array_equal(directions,again)
    reports[last]=saved
    next_state,updated,result=driver.finish_generation(tmp_path,definition,directions,theta,z)
    next_again,updated_again,result_again=driver.finish_generation(tmp_path,definition,directions,theta,z)
    assert next_again==next_state and result_again==result
    np.testing.assert_array_equal(updated_again,updated)
    # Hits are deliberately absurd; fitness remains ONLY terminal scores.
    expected=np.array([reports[j['id']]['total_reward'] for j in definition['jobs']]).reshape(8,2,4).mean(2)
    np.testing.assert_array_equal(result['returns'],expected)


def test_recovery_mid_training_serve_exact_records_graph_and_camera_phase(tmp_path):
    torch.set_num_threads(1)
    weights=np.random.default_rng(997721).normal(0,.001,(3,4389))
    def create(path):
        random.seed(7192);np.random.seed(7192);torch.manual_seed(7192)
        return DurableRun(path,session=FreshPongEvaluator(weights,PongEvaluationConfig(125682733,
            episodes=1,max_episode_ticks=17,opponent_speed=.25)),snapshot_transitions=2)
    continuous=create(tmp_path/'continuous.db')
    while not continuous.session.complete:continuous.step()
    expected=fingerprint(continuous.session);continuous.stop()
    split=create(tmp_path/'split.db')
    for _ in range(3):split.step()
    split.close_without_checkpoint();split=DurableRun(tmp_path/'split.db')
    assert split.replayed==1 and split.session.env.decisions==3
    while not split.session.complete:split.step()
    assert fingerprint(split.session)==expected;split.stop()
    with sqlite3.connect(tmp_path/'continuous.db') as a,sqlite3.connect(tmp_path/'split.db') as b:
        assert a.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==b.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()


def test_camera_grants_cannot_overspend_or_use_incomplete_fitness(tmp_path,monkeypatch):
    observed={};reports={}
    class Pool:
        def submit(self,fn,directory,name,grant,work):
            result=Future();observed[name]=observed.get(name,0)+grant
            reports[name]=dict(complete=False)
            result.set_result(dict(job=name,ok=True,result=dict(new_transitions=grant),wall_seconds=0.))
            return result
    monkeypatch.setattr(driver,'sample_count',lambda directory,name:observed.get(name,0))
    monkeypatch.setattr(driver,'read_report',lambda directory,name:reports.get(name))
    monkeypatch.setattr(driver,'complete_report',lambda directory,name:None)
    monkeypatch.setattr(driver,'pin_completion',lambda directory,name:None)
    import time
    result=driver.execute_jobs(tmp_path,[dict(id='one'),dict(id='two')],Pool(),deadline=time.time()+60,camera_cap=650)
    assert sum(observed.values())==600 and result['committed_cameras']==600 and not result['complete']


def test_completed_rollout_detects_report_and_semantic_journal_corruption(tmp_path):
    name='finished';db=tmp_path/(name+'.db')
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE replay_meta(id INTEGER,config BLOB);CREATE TABLE replay_artifacts(hash TEXT,payload BLOB);CREATE TABLE replay_samples(seq INTEGER,record BLOB);CREATE TABLE replay_active(id INTEGER,payload BLOB);')
        c.execute('INSERT INTO replay_meta VALUES(1,?)',(b'configuration',))
        c.execute('INSERT INTO replay_artifacts VALUES(?,?)',('bank',b'frozen'))
        c.execute('INSERT INTO replay_samples VALUES(0,?)',(b'camera-record',))
        c.execute('INSERT INTO replay_active VALUES(1,?)',(b'complete-state',))
    driver.atomic_json(tmp_path/(name+'.json'),dict(complete=True,total_reward=1.))
    driver.pin_completion(tmp_path,name);assert driver.complete_report(tmp_path,name)['total_reward']==1.
    with sqlite3.connect(db) as c:c.execute('UPDATE replay_active SET payload=?',(b'corrupted',))
    with pytest.raises(ValueError,match='artifacts changed'):driver.complete_report(tmp_path,name)
    with sqlite3.connect(db) as c:c.execute('UPDATE replay_active SET payload=?',(b'complete-state',))
    driver.atomic_json(tmp_path/(name+'.json'),dict(complete=True,total_reward=-1.))
    with pytest.raises(ValueError,match='artifacts changed'):driver.complete_report(tmp_path,name)
