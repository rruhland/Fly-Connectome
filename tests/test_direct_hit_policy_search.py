"""Synthetic-only optimizer/reservation/restart tests; no camera observations."""
import copy,hashlib,json,os,subprocess,sys,time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import policy_search_math as math
import run_direct_hit_policy_search as study


def generic_problem():
    rng=np.random.default_rng(8317);z=rng.standard_normal((32,4))
    theta=.1*math.sample_directions(rng,z,1)[0]
    return z,theta,copy.deepcopy(rng.bit_generator.state)


def test_legacy_four_serve_draws_are_exact_and_six_serves_preserve_direction_rng():
    z,theta,state=generic_problem()
    # Frozen before promotion from commit570222e, NumPy2.5.4/single-thread CPU.
    # This fixture stays independent after the implementation is integrated.
    legacy_serves=[1426473286,1221597214,1406296316,2057205108]
    legacy_rng=dict(bit_generator='PCG64',state=dict(
        state=82325583455525536942089500881315681755,
        inc=141132632283053394319354955151727482999),has_uint32=0,uinteger=0)
    default=math.generation_spec(theta,state,0,z)
    six=math.generation_spec(theta,state,0,z,serves_per_generation=6)
    assert hashlib.sha256(default[0].tobytes()).hexdigest()=='bc6d7b110a167ce1594aae6355a5e5f1d124e4c758bd83bc3f9bd8c460df8947'
    assert hashlib.sha256(default[1].tobytes()).hexdigest()=='e818d9e04ae515bbdc6205748cb1380750efe1e394c00bd6c1e3b79492adb9db'
    assert default[2]==legacy_serves and default[3]==legacy_rng
    for index in (0,1):
        np.testing.assert_array_equal(default[index],six[index])
    assert six[2][:4]==legacy_serves and len(set(six[2]))==6 and legacy_rng==six[3]
    excluded=math.generation_spec(theta,state,0,z,serves_per_generation=6,excluded_seeds=legacy_serves)
    assert not set(excluded[2])&set(legacy_serves) and excluded[3]==legacy_rng
    with pytest.raises(ValueError):math.generation_spec(theta,state,0,z,serves_per_generation=0)


def population():
    jobs=[];reports={}
    for d in range(8):
        for sign in range(2):
            for serve in range(6):
                name=f'{d}_{sign}_{serve}'
                jobs.append(dict(id=name,direction_index=d,sign_index=sign,serve_index=serve))
                reports[name]=dict(complete=True,episodes=1,hits=int((d+serve+sign)%3==0),
                    total_reward=(-1.)**sign,transitions=17,physics_seconds=.34)
    return dict(generation=0,global_generation=0,training_serve_seeds=list(range(6)),jobs=jobs),reports


def test_six_shared_serve_population_and_pending_definition_resume_exact(tmp_path):
    z,theta,rng=generic_problem()
    transform=math.SearchTransform(np.eye(4,4389),{})
    study.atomic_json(tmp_path/'manifest.json',dict(jobs=[]))
    state=dict(completed_generations=0,rng_state=rng)
    definition,directions=study.prepare_population(tmp_path,transform,z,state,theta,0,.65)
    again,replayed=study.prepare_population(tmp_path,transform,z,state,theta,0,.65)
    assert again==definition
    np.testing.assert_array_equal(replayed,directions)
    assert len(definition['jobs'])==96 and len(set(definition['training_serve_seeds']))==6
    assert not set(definition['training_serve_seeds'])&set(study.serve_schedule())
    for d in range(8):
        for sign in range(2):
            group=[j for j in definition['jobs'] if j['direction_index']==d and j['sign_index']==sign]
            assert [j['config']['seed'] for j in group]==definition['training_serve_seeds']
            mapped=(theta+(.25 if sign==0 else -.25)*directions[d])@transform.matrix
            np.testing.assert_array_equal(np.load(tmp_path/group[0]['weights_from']),mapped)


def test_fitness_uses_hits_only_with_explicit_cells_and_rejects_partial(monkeypatch,tmp_path):
    definition,reports=population()
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    returns,hits,_=study.population_fitness(tmp_path,definition)
    for r in reports.values():r['total_reward']*=193.
    again,again_hits,_=study.population_fitness(tmp_path,definition)
    np.testing.assert_array_equal(returns,again);np.testing.assert_array_equal(hits,again_hits)
    assert returns.shape==(8,2) and hits.shape==(8,2,6)
    reports.pop(definition['jobs'][-1]['id'])
    with pytest.raises(ValueError,match='partial'):study.population_fitness(tmp_path,definition)
    duplicate=copy.deepcopy(definition);duplicate['jobs'][-1]=duplicate['jobs'][0]
    with pytest.raises(ValueError,match='duplicate'):study.population_fitness(tmp_path,duplicate)


def test_update_and_post_result_crash_restart_are_exact(monkeypatch,tmp_path):
    z,theta,rng=generic_problem();definition,reports=population()
    directions,_,_,after=math.generation_spec(theta,rng,0,z,serves_per_generation=6)
    definition.update(input_theta_hash=study.array_hash(theta),rng_state_after_draw=after)
    np.save(tmp_path/'search-transform.npy',np.eye(4));study.weights_file(tmp_path,'policy_at0.npy',theta)
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    original=dict(completed_generations=0,theta_hash=study.array_hash(theta),rng_state=rng)
    study.atomic_json(tmp_path/'search-state.json',original)
    atomic=study.atomic_json
    def crash_before_state(path,value):
        if path.name=='search-state.json':raise OSError('simulated result-to-state crash')
        atomic(path,value)
    monkeypatch.setattr(study,'atomic_json',crash_before_state)
    with pytest.raises(OSError,match='result-to-state'):
        study.finish_population(tmp_path,definition,directions,theta,z)
    assert json.loads((tmp_path/'search-state.json').read_text())==original
    result=json.loads((tmp_path/'g00-result.json').read_text())
    assert (tmp_path/'theta_at1.npy').exists()
    monkeypatch.setattr(study,'atomic_json',atomic)
    # A crash after immutable artifacts/result but before advancing the state
    # reruns the same full population, never adds its update twice.
    state,updated,result2=study.finish_population(tmp_path,definition,directions,theta,z)
    assert result==result2 and state['rng_state']==after
    expected,_=math.mirrored_update(theta,directions,np.array(result['returns']),z)
    np.testing.assert_array_equal(updated,expected)
    assert state['completed_generations']==1 and result['total_hits']==sum(r['hits'] for r in reports.values())
    for r in reports.values():r['hits']=0
    returns,_,_=study.population_fitness(tmp_path,definition)
    unchanged,diagnostic=math.mirrored_update(theta,directions,returns,z)
    np.testing.assert_array_equal(unchanged,theta)
    assert diagnostic['zero_update'] and diagnostic['selected_directions']==[0,1,2,3]


def test_partial_generation_freezes_original_theta_rng_and_resume_boundary(tmp_path,monkeypatch):
    z,theta,rng=generic_problem();directory=tmp_path
    study.atomic_json(directory/'manifest.json',dict(jobs=[]))
    state=dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=study.array_hash(theta),rng_state=rng)
    directions,_,_,after=math.generation_spec(theta,rng,0,z,serves_per_generation=6)
    def build(directory,transform,z,state,theta,offset,speed):
        definition=dict(generation=0,global_generation=0,input_theta_hash=study.array_hash(theta),
            input_rng_state=rng,rng_state_after_draw=after,jobs=[])
        study.atomic_json(directory/'g00-definition.json',definition)
        return definition,directions
    monkeypatch.setattr(study.engine,'execute_jobs',lambda *a,**k:dict(complete=False,failed=False,committed_cameras=0,unfinished=[]))
    def forbidden(*a,**k):pytest.fail('partial population must not call updater')
    frozen,unchanged=study.phase_engine.run_phase(directory,study.SPEC,None,z,state,theta,0,None,time.time()+60,
        population_builder=build,population_updater=forbidden)
    assert not frozen['valid'] and frozen['completed_generations']==0
    assert frozen['used_cameras']==0 and frozen['unused_draw_skipped']
    assert frozen['next_rng_state']==after and frozen['next_global_generation']==1
    np.testing.assert_array_equal(theta,unchanged)
    again,_=study.phase_engine.run_phase(directory,study.SPEC,None,z,state,theta,0,None,time.time()+600,
        population_builder=forbidden,population_updater=forbidden)
    assert again==frozen


def test_completion_callbacks_force_zero_work_recovery_and_four_worker_reservations(tmp_path,monkeypatch):
    engine=study.engine;counts={'recovered':17};reports={'recovered':dict(complete=True)};pins=set();grants=[]
    class Pool:
        def submit(self,fn,directory,name,grant,work):
            grants.append((name,grant));counts[name]=counts.get(name,0)+grant;reports[name]=dict(complete=True)
            result=Future();result.set_result(dict(job=name,ok=True,result={},wall_seconds=0.));return result
    monkeypatch.setattr(engine,'sample_count',lambda directory,name:counts.get(name,0))
    monkeypatch.setattr(engine,'read_report',lambda directory,name:reports.get(name))
    def verify(directory,job):return reports.get(job['id']) if job['id'] in pins else None
    def pin(directory,job):pins.add(job['id'])
    jobs=[dict(id=name) for name in ('recovered','a','b','c','d')]
    result=engine.execute_jobs(tmp_path,jobs,Pool(),deadline=time.time()+60,camera_cap=3617,initial_consumed=17,
        completion_report=verify,completion_pin=pin)
    assert result['complete'] and grants[0]==('recovered',0)
    assert sum(grant for name,grant in grants)==3600 and counts['recovered']==17
    assert pins=={j['id'] for j in jobs}
    def corrupted(directory,job):raise ValueError('mismatched completion pin')
    with pytest.raises(ValueError,match='mismatched completion'):
        engine.execute_jobs(tmp_path,jobs,Pool(),deadline=time.time()+60,camera_cap=3617,
            completion_report=corrupted,completion_pin=pin)


def test_fixed_fresh_cohorts_and_initial_readout_is_in_strongest_control_set():
    jobs=study.evaluation_jobs(dict(completed_generations=2))
    assert len(jobs)==7*32 and 'initial_restricted' in study.CONTROLS
    assert len(set(study.serve_schedule()))==32 and not set(study.serve_schedule())&set(study.engine.FINAL)
    for policy in ('endpoint','initial_restricted'):
        cohort=[j for j in jobs if j['policy']==policy]
        assert len(cohort)==32 and all(j['config']['policy_kind']=='trained' and j['config']['action_hold_frames']==1 for j in cohort)


def test_import_does_not_freeze_gc_and_worker_initializer_keeps_collection_enabled():
    code="""
import gc
before=gc.get_freeze_count()
import run_direct_hit_policy_search as study
assert gc.get_freeze_count()==before
study.initialize_worker()
assert gc.isenabled() and gc.get_freeze_count()>0
print(study.GC_WORKER_MODE)
"""
    child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=os.environ.copy(),check=True)
    assert child.stdout.strip()==study.GC_WORKER_MODE
