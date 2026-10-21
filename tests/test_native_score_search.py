"""Synthetic native-score contracts: zero camera or physics observations."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_native_score_search as study


def problem(tmp_path,monkeypatch):
    for key,value in dict(TRAINING_SEED=21901,OPTIMIZER_SEED=21902,DEV_SEED=21903,
                         ACTION_SEED=21904,BOOTSTRAP=21905,TAG=99117).items():monkeypatch.setattr(study,key,value)
    rng=np.random.default_rng(819);z=rng.standard_normal((32,4))
    theta=.3*study.prior.sample_directions(rng,z,1)[0]
    state=dict(completed_updates=0,next_global_generation=0,theta_file='theta_at0.npy',
        theta_hash=study.array_hash(theta),rng_state=copy.deepcopy(rng.bit_generator.state),pending_diagnostics=None)
    np.save(tmp_path/'search-transform.npy',np.eye(4));np.save(tmp_path/'theta_at0.npy',theta)
    study.weights_file(tmp_path,'seed1.npy',theta);study.weights_file(tmp_path,'policy_at0.npy',theta)
    study.atomic_json(tmp_path/'boundary-proof.json',dict(endpoint=dict(theta_hash=state['theta_hash'],next_rng_state=state['rng_state'])))
    study.atomic_json(tmp_path/'search-state.json',state);study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=[2**31+59]))
    manifest=dict(jobs=[],serve_seeds=list(range(101,165)),action_seeds=[2**31+60],started_wall_ns=time.time_ns())
    study.atomic_json(tmp_path/'manifest.json',manifest);reports={};samples={};calls=[]
    monkeypatch.setattr(study.engine,'sample_count',lambda directory,name:samples.get(name,0))
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    def execute(directory,jobs,pool,deadline,cap,current):
        calls.append([j['id'] for j in jobs])
        if all(j['id'] in reports for j in jobs):return dict(complete=True,failed=False)
        if deadline-time.time()<=25:return dict(complete=False,failed=False)
        for j in jobs:
            win=int((j.get('direction_index',0)%2==0 and j.get('sign_index',0)==0 and j['serve_index']<5)
                or j['serve_index']==0)
            samples[j['id']]=5;reports[j['id']]=dict(complete=True,episodes=1,wins=win,total_reward=1. if win else -1.,
                transitions=5,physics_seconds=.1,hits=999999)
        return dict(complete=True,failed=False)
    monkeypatch.setattr(study,'execute',execute)
    return z,theta,state,manifest,reports,samples,calls,execute


@pytest.mark.parametrize('returns',[np.zeros((8,2)),np.ones((8,2)),np.arange(16).reshape(8,2)/16])
def test_positive_scale_rule_with_ties_and_zero_fitness(returns):
    rng=np.random.default_rng(811);z=rng.standard_normal((64,5));theta=.3*study.prior.sample_directions(rng,z,1)[0]
    directions=study.prior.sample_directions(rng,z,8);scale=7.25
    a=study.relative_scales(theta,z);b=study.relative_scales(scale*theta,z)
    np.testing.assert_allclose(b['noise_scale'],scale*a['noise_scale'],rtol=2e-15)
    pa=theta[None,None]+a['noise_scale']*directions[:,None]*np.array([1.,-1.])[None,:,None,None]
    pb=scale*theta[None,None]+b['noise_scale']*directions[:,None]*np.array([1.,-1.])[None,:,None,None]
    np.testing.assert_allclose(pb,scale*pa,rtol=2e-13,atol=2e-15)
    ua,da=study.mirrored_update(theta,directions,returns,z,step_size=a['step_size'])
    ub,db=study.mirrored_update(scale*theta,directions,returns,z,step_size=b['step_size'])
    np.testing.assert_allclose(ub,scale*ua,rtol=2e-13,atol=2e-15)
    assert da['selected_directions']==db['selected_directions'] and da['zero_update']==db['zero_update']
    assert study.function_rms(ua-theta,z)<=.100000000001*study.function_rms(theta,z)
    with pytest.raises(ValueError):study.relative_scales(np.zeros_like(theta),z)


def test_objective_is_only_win_not_hit_or_timeout(tmp_path,monkeypatch):
    z,theta,state,_,reports,_,_,execute=problem(tmp_path,monkeypatch)
    definition,directions=study.prepare_population(tmp_path,z,state,theta,())
    assert len(definition['jobs'])==192 and len(definition['diagnostic_jobs'])==12
    execute(tmp_path,definition['jobs'],None,time.time()+60,10000,0)
    before,values,_=study.population_fitness(tmp_path,definition)
    for r in reports.values():
        r['hits']=0
        if not r['wins']:r['total_reward']=0.  # loss -> timeout has the same fitness
    after,again,_=study.population_fitness(tmp_path,definition)
    np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(values,again)
    assert definition['anchor_groups']['start']==definition['anchor_groups']['current_center']
    r=next(iter(reports.values()));r['total_reward']=1. if not r['wins'] else 0.
    with pytest.raises(ValueError,match='raw score'):study.population_fitness(tmp_path,definition)


def test_result_state_crash_reconciles_at_exhausted_budget(tmp_path,monkeypatch):
    z,theta,state,m,_,_,_,execute=problem(tmp_path,monkeypatch)
    definition,directions=study.prepare_population(tmp_path,z,state,theta,())
    execute(tmp_path,definition['jobs'],None,time.time()+60,10000,0)
    original=study.atomic_json
    def crash(path,value):
        if path.name=='search-state.json':raise OSError('result-state crash')
        original(path,value)
    monkeypatch.setattr(study,'atomic_json',crash)
    with pytest.raises(OSError):study.finish_fitness(tmp_path,definition,directions,theta,z)
    assert study.transactions.committed_fitness_rows(study,tmp_path,state)==0
    monkeypatch.setattr(study,'atomic_json',original);monkeypatch.setattr(study,'TRAIN_CAMERAS',960)
    monkeypatch.setattr(study,'TRAIN_SECONDS',0.)
    frozen,recovered,new=study.transactions.train(study,tmp_path,None,m,z,state,theta)
    assert recovered['completed_updates']==1 and recovered['rng_state']==definition['rng_state_after_draw']
    assert frozen['used_fitness_cameras']==960 and frozen['diagnostic_cameras']==0
    assert frozen['pending_diagnostics']=='g00' and not frozen['valid']
    expected=np.load(tmp_path/'theta_at1.npy');np.testing.assert_array_equal(new,expected)
    assert study.transactions.train(study,tmp_path,None,m,z,recovered,new)[0]==frozen


def test_partial_anchors_count_without_dropping_completed_population(tmp_path,monkeypatch):
    z,theta,state,m,_,samples,calls,execute=problem(tmp_path,monkeypatch)
    def partial(directory,jobs,pool,deadline,cap,current):
        if jobs[0]['id'].startswith('g02_anchor'):
            samples[jobs[0]['id']]=5;return dict(complete=False,failed=False)
        return execute(directory,jobs,pool,deadline,cap,current)
    monkeypatch.setattr(study,'execute',partial)
    frozen,state,theta=study.transactions.train(study,tmp_path,None,m,z,state,theta)
    assert frozen['valid'] and frozen['new_updates']==3
    assert frozen['used_fitness_cameras']==2880 and frozen['unused_partial_fitness_cameras']==0
    assert frozen['diagnostic_cameras']==65 and frozen['total_training_phase_cameras']==2945
    assert frozen['pending_diagnostics']=='g02' and frozen['diagnostic_status']=='partial'
    assert not (tmp_path/'g03-definition.json').exists()


def test_action_ids_and_fresh_roots_are_excluded_from_training(tmp_path,monkeypatch):
    z,theta,state,m,_,_,_,_=problem(tmp_path,monkeypatch);seen={}
    def stop(directory,z,state,theta,excluded):
        seen['excluded']=set(excluded);raise RuntimeError('definition probe only')
    monkeypatch.setattr(study,'prepare_population',stop)
    with pytest.raises(RuntimeError,match='definition probe'):
        study.transactions.train(study,tmp_path,None,m,z,state,theta)
    assert {2**31+59,2**31+60}<=seen['excluded']
    actual=study.schedule([])
    assert len(set(actual['serves']))==len(set(actual['control_actions']))==64
    assert not set(actual['serves'])&set(actual['control_actions'])
    metadata=tmp_path/'metadata';metadata.mkdir()
    study.atomic_json(metadata/'manifest.json',dict(control_action_seeds=[2**31+59]))
    destination=tmp_path/'exclusion-proof';destination.mkdir()
    study.transfer.snapshot_exclusions(destination,metadata)
    assert 2**31+59 in json.loads((destination/'seed-exclusions.json').read_text())['ids']
    with pytest.raises(ValueError):study.schedule([study.DEV_SEED])


def test_paired_max_evidence_and_common_natural_offset():
    groups={}
    for policy in study.POLICIES:
        groups[policy]=[]
        for i in range(64):
            natural=i<8;win=natural or i<(40 if policy=='endpoint' else 20)
            groups[policy].append(dict(reward=1. if win else -1.,own_returns=int(win and not natural),
                first_incoming=None if natural else dict(tick=100+i,success=win)))
    value=study.evidence(groups)
    assert value['qualifies'] and value['controls']==list(study.CONTROLS)
    assert value['endpoint']['total']['lift']==value['endpoint']['own_hit']['lift']
    assert value['endpoint']['total']['ci95']==value['endpoint']['own_hit']['ci95']
    assert value['endpoint']['total']['strongest_control_rate']-value['endpoint']['own_hit']['strongest_control_rate']==8/64
    assert value['endpoint_minus_start']['total']==value['endpoint_minus_start']['own_hit']
    assert value['endpoint']['total']['lift']==20/64


def test_retained_outcome_preserves_clock_and_rejects_anchor_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(study.transfer,'outcome_evidence',lambda directory,m:dict(raw='fixed'))
    study.atomic_json(tmp_path/'g00-anchor-result.json',dict(value=1));m={}
    saved=dict(summary=dict(wall_seconds=12.,overall_budget_met=True),evidence=study.outcome_evidence(tmp_path,m))
    study.atomic_json(tmp_path/'study-completion.json',saved);study.atomic_json(tmp_path/'summary.json',saved['summary'])
    monkeypatch.setattr(study.time,'time',lambda:1e15)
    assert study.retained_outcome(tmp_path,m)==saved['summary']
    study.atomic_json(tmp_path/'g00-anchor-result.json',dict(value=2))
    with pytest.raises(ValueError,match='retained search evidence'):study.retained_outcome(tmp_path,m)


def test_evaluation_failure_persists_and_frozen_boundary_rejects_tamper(tmp_path,monkeypatch):
    job=dict(id='failed_eval',purpose='fresh_evaluation')
    study.atomic_json(tmp_path/'manifest.json',dict(jobs=[job]))
    monkeypatch.setattr(study.engine,'sample_count',lambda *args:0)
    monkeypatch.setattr(study.completion,'complete_report',lambda *args:None)
    calls=[]
    def fail(*args):
        calls.append(1);(tmp_path/'rollout-costs.jsonl').write_text(json.dumps(dict(job=job['id'],ok=False))+'\n')
        return dict(complete=False,failed=True,committed_cameras=0,unfinished=[job['id']])
    monkeypatch.setattr(study,'execute',fail)
    first=study.evaluate(tmp_path,[job],None,time.time()+60)
    assert first['failed'] and not first['complete'] and len(calls)==1
    assert study.evaluate(tmp_path,[job],None,0)==first and len(calls)==1
    boundary=json.loads((tmp_path/'rollout-boundary.json').read_text());boundary['evaluation']['failed']=False
    study.atomic_json(tmp_path/'rollout-boundary.json',boundary)
    with pytest.raises(ValueError,match='frozen evaluation boundary'):study.evaluate(tmp_path,[job],None,0)


def test_calibration_bytes_bound_to_fit_and_both_lineages(tmp_path,monkeypatch):
    cache=tmp_path/'cache.npz';cache.write_bytes(b'synthetic score-free bytes')
    monkeypatch.setattr(study.engine,'CACHE',cache);digest=study.sha(cache)
    lineage=tmp_path/'ancestor';lineage.mkdir()
    study.atomic_json(lineage/'manifest.json',dict(calibration_source_hash=digest))
    monkeypatch.setattr(study.transfer,'LINEAGES',dict(seed1=lineage,seed2=lineage))
    study.atomic_json(tmp_path/'transform-fit.json',dict(calibration_source_hash=digest))
    study.atomic_json(tmp_path/'projection.json',dict(calibration=dict(source_sha256=digest)))
    assert study.validate_calibration(tmp_path)==digest
    cache.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='calibration provenance'):study.validate_calibration(tmp_path)
