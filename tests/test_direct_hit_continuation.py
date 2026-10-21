"""Synthetic continuation boundaries; zero new camera observations."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_direct_hit_continuation as study


def problem(tmp_path,monkeypatch):
    rng=np.random.default_rng(8129);z=rng.standard_normal((24,4))
    theta=.1*study.prior.sample_directions(rng,z,1)[0];state=dict(completed_updates=0,
        next_global_generation=10,theta_file='theta_at0.npy',theta_hash=study.array_hash(theta),
        rng_state=copy.deepcopy(rng.bit_generator.state),pending_diagnostics=None)
    np.save(tmp_path/'search-transform.npy',np.eye(4));np.save(tmp_path/'theta_at0.npy',theta)
    study.weights_file(tmp_path,'endpoint.npy',theta);study.weights_file(tmp_path,'policy_at0.npy',theta)
    study.atomic_json(tmp_path/'boundary-proof.json',dict(endpoint=dict(theta_hash=state['theta_hash'],next_rng_state=state['rng_state'])))
    study.atomic_json(tmp_path/'search-state.json',state)
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=[]))
    manifest=dict(jobs=[],serve_seeds=list(range(1,65)),started_wall_ns=time.time_ns())
    study.atomic_json(tmp_path/'manifest.json',manifest)
    reports={};samples={};calls=[]
    monkeypatch.setattr(study.engine,'sample_count',lambda directory,name:samples.get(name,0))
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    def execute(directory,jobs,pool,deadline,cap,current):
        calls.append([j['id'] for j in jobs])
        if all(job['id'] in reports for job in jobs):return dict(complete=True,failed=False)
        if deadline-time.time()<=25:return dict(complete=False,failed=False)
        for job in jobs:
            name=job['id'];samples[name]=5
            reports[name]=dict(complete=True,episodes=1,hits=int((job.get('direction_index',0)+job['serve_index'])%3==0),
                total_reward=-1.,transitions=5,physics_seconds=.1)
        return dict(complete=True,failed=False)
    monkeypatch.setattr(study,'execute',execute)
    return z,theta,state,manifest,reports,samples,calls,execute


def test_pins_preupdate_centers_and_reuses_only_initial_diagnostic_identity(tmp_path,monkeypatch):
    z,theta,state,_,_,_,_,_=problem(tmp_path,monkeypatch)
    definition,directions=study.prepare_population(tmp_path,z,state,theta,())
    assert len(definition['jobs'])==96 and len(definition['diagnostic_jobs'])==6
    assert definition['anchor_groups']['current_center']==definition['anchor_groups']['theta9']
    assert definition['anchor_readouts']['source_cumulative_update']==9
    assert definition['anchor_readouts']['initial_identity_reused']
    assert definition['global_generation']==10 and definition['generation']==0
    assert len(set(definition['training_serve_seeds']))==6
    assert all(j['purpose']=='training_fitness' for j in definition['jobs'])
    assert all(j['purpose']=='training_anchor' for j in definition['diagnostic_jobs'])
    again,replayed=study.prepare_population(tmp_path,z,state,theta,())
    assert again==definition;np.testing.assert_array_equal(replayed,directions)
    later=dict(state,completed_updates=2,next_global_generation=12,theta_hash=study.array_hash(theta+.01),theta_file='theta_at2.npy')
    later_definition,_=study.prepare_population(tmp_path,z,later,theta+.01,())
    assert len(later_definition['diagnostic_jobs'])==12
    assert later_definition['anchor_groups']['current_center']!=later_definition['anchor_groups']['theta9']
    assert later_definition['anchor_readouts']['source_cumulative_update']==11


def test_completed_fitness_crash_reconciles_even_after_camera_and_wall_exhaustion(tmp_path,monkeypatch):
    z,theta,state,manifest,_,_,_,execute=problem(tmp_path,monkeypatch)
    definition,directions=study.prepare_population(tmp_path,z,state,theta,())
    execute(tmp_path,definition['jobs'],None,time.time()+60,10000,0)
    original=study.atomic_json
    def crash(path,value):
        if path.name=='search-state.json':raise OSError('result-to-state crash')
        original(path,value)
    monkeypatch.setattr(study,'atomic_json',crash)
    with pytest.raises(OSError,match='result-to-state'):study.finish_fitness(tmp_path,definition,directions,theta,z)
    assert json.loads((tmp_path/'search-state.json').read_text())==state
    assert (tmp_path/'g10-result.json').exists() and study.committed_fitness_rows(tmp_path,state)==0
    monkeypatch.setattr(study,'atomic_json',original)
    monkeypatch.setattr(study,'TRAIN_CAMERAS',480);monkeypatch.setattr(study,'TRAIN_SECONDS',0.)
    frozen,recovered,updated=study.train(tmp_path,None,manifest,z,state,theta)
    expected,_=study.mirrored_update(theta,directions,np.array(json.loads((tmp_path/'g10-result.json').read_text())['returns']),z)
    np.testing.assert_array_equal(updated,expected)
    assert recovered['completed_updates']==1 and recovered['rng_state']==definition['rng_state_after_draw']
    assert frozen['used_fitness_cameras']==480 and frozen['unused_partial_fitness_cameras']==0
    assert frozen['diagnostic_cameras']==0 and frozen['pending_diagnostics']=='g10'
    assert frozen['next_global_generation']==11 and not frozen['unused_draw_skipped']
    assert not frozen['valid'] and frozen['diagnostic_status']=='partial'
    assert study.train(tmp_path,None,manifest,z,recovered,updated)[0]==frozen


def test_anchor_commit_crash_does_not_reapply_fitness_or_reset_rng(tmp_path,monkeypatch):
    z,theta,state,manifest,_,_,calls,execute=problem(tmp_path,monkeypatch)
    definition,directions=study.prepare_population(tmp_path,z,state,theta,())
    execute(tmp_path,definition['jobs'],None,time.time()+60,10000,0)
    state,theta=study.finish_fitness(tmp_path,definition,directions,theta,z)
    execute(tmp_path,definition['diagnostic_jobs'],None,time.time()+60,10000,480)
    study.finish_diagnostics(tmp_path,definition) # crash before clearing pending phase
    monkeypatch.setattr(study,'TRAIN_SECONDS',0.)
    frozen,recovered,updated=study.train(tmp_path,None,manifest,z,state,theta)
    assert recovered['completed_updates']==1 and recovered['pending_diagnostics'] is None
    assert recovered['rng_state']==definition['rng_state_after_draw']
    assert frozen['used_fitness_cameras']==480 and frozen['diagnostic_cameras']==30
    np.testing.assert_array_equal(updated,theta)
    assert len(calls)==3 and all('anchor' in name for name in calls[-1])


def test_inclusive_budget_distinguishes_committed_partial_and_anchor_rows(tmp_path,monkeypatch):
    z,theta,state,manifest,reports,samples,_,execute=problem(tmp_path,monkeypatch)
    def partial(directory,jobs,pool,deadline,cap,current):
        if jobs[0]['id'].startswith('g12'):
            name=jobs[0]['id'];samples[name]=7
            return dict(complete=False,failed=False)
        if jobs[0]['purpose']=='training_anchor':
            assert json.loads((directory/'search-state.json').read_text())['completed_updates']==1
            assert (directory/'g10-result.json').exists()
        return execute(directory,jobs,pool,deadline,cap,current)
    monkeypatch.setattr(study,'execute',partial)
    frozen,state,theta=study.train(tmp_path,None,manifest,z,state,theta)
    assert frozen['new_updates']==2 and frozen['cumulative_learned_updates']==11 and frozen['valid']
    assert frozen['used_fitness_cameras']==960 and frozen['unused_partial_fitness_cameras']==7
    assert frozen['diagnostic_cameras']==30 and frozen['total_training_phase_cameras']==997
    assert frozen['next_global_generation']==13 and frozen['unused_draw_skipped']
    pending=json.loads((tmp_path/'g12-definition.json').read_text())
    assert frozen['next_rng_state']==pending['rng_state_after_draw']
    assert study.train(tmp_path,None,manifest,z,state,theta)[0]==frozen
    changed=dict(frozen,next_global_generation=12);study.atomic_json(tmp_path/'training-endpoint.json',changed)
    with pytest.raises(ValueError,match='frozen endpoint'):study.train(tmp_path,None,manifest,z,state,theta)
    study.atomic_json(tmp_path/'training-endpoint.json',frozen)
    changed_state=dict(state,rng_state=copy.deepcopy(pending['rng_state_after_draw']))
    with pytest.raises(ValueError,match='optimizer boundary'):study.train(tmp_path,None,manifest,z,changed_state,theta)


def test_logged_fitness_failure_survives_restart_and_prevents_valid_endpoint(tmp_path,monkeypatch):
    z,theta,state,manifest,_,samples,calls,_=problem(tmp_path,monkeypatch)
    definition,_=study.prepare_population(tmp_path,z,state,theta,())
    job=definition['jobs'][0];samples[job['id']]=3
    (tmp_path/'rollout-costs.jsonl').write_text(json.dumps(dict(job=job['id'],ok=False,error='prior failure'))+'\n')
    frozen,state,theta=study.train(tmp_path,None,manifest,z,state,theta)
    assert not calls and not frozen['valid'] and frozen['fitness_failure']
    assert frozen['unused_partial_fitness_cameras']==3 and frozen['next_global_generation']==11
    assert frozen['training_stop']=='prior logged fitness failure'


def test_exact_initial_boundary_and_tampered_inherited_rng_fail_closed(tmp_path,monkeypatch):
    z,theta,state,_,_,_,_,_=problem(tmp_path,monkeypatch)
    lineage=tmp_path/'old';lineage.mkdir();monkeypatch.setattr(study,'LINEAGE',lineage)
    np.save(lineage/'theta_at9.npy',theta)
    endpoint=dict(completed_generations=9,next_global_generation=10,unused_draw_skipped=True,
        unused_cameras=14312,next_rng_state=state['rng_state'],theta_hash=study.array_hash(theta))
    study.atomic_json(lineage/'training-endpoint.json',endpoint)
    study.atomic_json(lineage/'g09-definition.json',dict(rng_state_after_draw=state['rng_state']))
    proof=dict(endpoint=endpoint,endpoint_hash=study.sha(lineage/'training-endpoint.json'),
        skipped_definition_hash=study.sha(lineage/'g09-definition.json'),
        original_learned_updates=9,first_global_generation=10,unused_rows_reused=0)
    study.atomic_json(tmp_path/'boundary-proof.json',proof)
    assert study.validate_initial_boundary(tmp_path)==endpoint
    changed=copy.deepcopy(proof);changed['endpoint']['next_rng_state']['state']['state']+=1
    study.atomic_json(tmp_path/'boundary-proof.json',changed)
    with pytest.raises(ValueError,match='skipped-draw'):study.validate_initial_boundary(tmp_path)


def rally(success):
    return dict(own_returns=int(success),own_opportunities=1,
        first_incoming=dict(success=bool(success),tick=17,pre_ball=[.1,.4],pre_velocity=[-.4,.1]))


def test_nine_fresh_cohorts_and_secondary_comparisons_do_not_change_primary(monkeypatch):
    monkeypatch.setattr(study,'SERVES',4)
    jobs=study.evaluation_jobs(None,dict(serve_seeds=list(range(1,65))),dict(completed_updates=3))
    assert len(jobs)==576
    assert all(j['weights_from']=='policy_at3.npy' for j in jobs if j['policy'] in ('endpoint','blank'))
    assert all(j['weights_from']=='endpoint.npy' for j in jobs if j['policy']=='theta9')
    assert all(j['config']['blank_camera']==(j['policy']=='blank') for j in jobs)
    data={p:[rally(p=='endpoint') for _ in range(4)] for p in study.POLICIES}
    primary,secondary,extra=study.evidence(data)
    assert primary['arms']['endpoint']['promising_development_signal'] and extra
    assert secondary['theta9']['first_incoming_lift']==1
    data['theta9']=[rally(True) for _ in range(4)]
    again,_,extra=study.evidence(data)
    assert again==primary and not extra
    data['blank'][0]['first_incoming']['tick']=19
    with pytest.raises(ValueError,match='trajectory differs'):study.evidence(data)


def test_extra_learning_requires_no_invalid_bootstrap_draws(monkeypatch):
    def evidence(*args,**kwargs):
        return dict(paired_arm_comparison=dict(invalid_bootstrap_draws=1,
            first_incoming_lift_ci95=[.1,.2],all_opportunity_lift_ci95=[.1,.2]))
    monkeypatch.setattr(study,'paired_evidence',evidence)
    assert not study.evidence({})[2]
