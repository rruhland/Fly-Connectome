"""Task boundary and optimizer phase tests; no new candidate allocation."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_first_return_curriculum as study
from policy_search_math import SearchTransform
from run_pong_development import validate_first_hit_job
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig


def near_contact(*,cutoff=True,skip=False):
    torch.set_num_threads(1)
    config=PongEvaluationConfig(710191,episodes=1,max_episode_ticks=30,policy_kind='noop',
        skip_baseline_perception=skip,opponent_speed=.65)
    session=FreshPongEvaluator(None,config,first_hit_cutoff=cutoff)
    # Synthetic physical fixture only: legal incoming ball reaches the paddle on
    # camera two. No geometry enters the policy or any candidate experiment.
    session.env.game.ball[0]=torch.tensor([.076,.5])
    session.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
    return session


def test_first_hit_retains_whole_camera_raw_truth_and_default_prefix():
    cut,full=near_contact(),near_contact(cutoff=False)
    assert not hasattr(full,'first_hit_cutoff')
    first=cut.step();assert first['hits']==0 and not first['task_stopped'] and not cut.complete
    expected=full.step();assert {k:v for k,v in first.items() if not k.startswith('task_')}==expected
    last=cut.step();expected=full.step()
    assert {k:v for k,v in last.items() if not k.startswith('task_')}==expected
    assert last['hits']==1 and last['task_stopped'] and last['task_stop_reason']=='first_own_hit'
    assert not last['terminated'] and not last['truncated'] and last['reward']==0
    assert cut.complete and not full.complete and cut.transitions==2
    assert torch.equal(cut.previous,cut.env.game.render(64,64)[0]) and cut.latest_encoded is not None
    summary=cut.summary()
    assert summary['task_stops']==1 and summary['wins']==summary['losses']==summary['truncations']==0
    assert summary['episodes']==1 and summary['completion_fraction']==0
    assert summary['rallies'][0]['task_stopped'] and not summary['rallies'][0]['terminated']
    with pytest.raises(StopIteration):cut.step()
    assert 'task_stops' not in full.summary() and 'task_stopped' not in expected


def test_cutoff_rejects_multi_episode_and_frozen_mutation():
    with pytest.raises(ValueError,match='exactly one'):
        FreshPongEvaluator(None,PongEvaluationConfig(3,episodes=2,policy_kind='noop'),first_hit_cutoff=True)
    with pytest.raises(ValueError,match='bool'):
        FreshPongEvaluator(None,PongEvaluationConfig(3,episodes=1,policy_kind='noop'),first_hit_cutoff=1)
    session=near_contact(skip=True);session.first_hit_cutoff=False
    with pytest.raises(RuntimeError,match='changed frozen'):session.step()


def test_durable_pending_then_task_complete_recovery(tmp_path):
    run=DurableRun(tmp_path/'task.db',session=near_contact(skip=True),snapshot_transitions=100)
    first=run.step();pending=fingerprint(run.session);run.close_without_checkpoint()
    run=DurableRun(tmp_path/'task.db');assert run.replayed==1 and fingerprint(run.session)==pending
    assert not run.session.complete and not first['task_stopped']
    last=run.step();final=fingerprint(run.session);run.close_without_checkpoint()
    run=DurableRun(tmp_path/'task.db')
    assert run.replayed==2 and fingerprint(run.session)==final and run.session.complete
    assert run.session.summary()['task_stops']==1 and last['hits']==1
    run.stop()
    run=DurableRun(tmp_path/'task.db');assert run.replayed==0 and run.session.complete;run.close_without_checkpoint()


def test_worker_cutoff_is_only_declared_single_episode_training():
    good=dict(kind='evaluate',purpose='training_fitness',config=dict(episodes=1),first_hit_cutoff=True)
    assert validate_first_hit_job(good)
    for change in (dict(purpose='fresh_evaluation'),dict(purpose='calibration'),dict(kind='learn'),
                   dict(config=dict(episodes=2)),dict(first_hit_cutoff=1)):
        with pytest.raises(ValueError,match='training fitness only'):validate_first_hit_job(dict(good,**change))
    assert not validate_first_hit_job(dict(kind='evaluate'))


def problem(directory,monkeypatch,spec=None,index=0):
    directory.mkdir(exist_ok=True)
    spec=dict(study.PHASES[0] if spec is None else spec)
    rng=np.random.default_rng(8193);z=rng.standard_normal((32,4));theta=.1*study.prior.sample_directions(rng,z,1)[0]
    matrix=np.eye(4);transform=SearchTransform(matrix,{})
    np.save(directory/'search-transform.npy',matrix);np.save(directory/'theta_at0.npy',theta)
    study.weights_file(directory,'policy_at0.npy',theta)
    state=dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=study.array_hash(theta),
        rng_state=copy.deepcopy(rng.bit_generator.state),initial_policy_hash=study.sha(directory/'policy_at0.npy'))
    study.atomic_json(directory/'search-state.json',state)
    manifest=dict(jobs=[],phase=spec,global_generation_offset=index,input_theta_hash=state['theta_hash'],
        input_rng_state=state['rng_state'],excluded_training_seeds=[])
    study.atomic_json(directory/'manifest.json',manifest)
    samples={};reports={};calls=[]
    monkeypatch.setattr(study.engine,'sample_count',lambda directory,name:samples.get(name,0))
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    def execute(directory,jobs,pool,**kwargs):
        calls.append(jobs)
        for job in jobs:
            hit=int((job['direction_index']+job['sign_index']+job['serve_index'])%3==0)
            samples[job['id']]=5
            reports[job['id']]=dict(complete=True,episodes=1,hits=hit,total_reward=0 if hit else -1.,
                wins=0,losses=1-hit,truncations=0,task_stops=hit,task_definition='first-own-hit-task-v1',
                transitions=5,physics_seconds=.1)
        return dict(complete=True,failed=False,committed_cameras=sum(samples.values()))
    monkeypatch.setattr(study.engine,'execute_jobs',execute)
    return spec,transform,z,state,theta,samples,reports,calls,execute


def test_first_population_shared_twelve_and_full_phase_six(tmp_path,monkeypatch):
    spec,t,z,state,theta,_,_,_,_=problem(tmp_path/'first',monkeypatch,index=3)
    definition,directions=study.prepare_population(tmp_path/'first',t,z,state,theta,3,.65)
    assert len(definition['jobs'])==192 and len(set(definition['training_serve_seeds']))==12
    assert definition['global_generation']==3 and definition['generation']==0
    assert all(j['first_hit_cutoff'] and j['purpose']=='training_fitness' for j in definition['jobs'])
    again,draw=study.prepare_population(tmp_path/'first',t,z,state,theta,3,.65)
    assert again==definition;np.testing.assert_array_equal(draw,directions)
    spec,t,z,state,theta,_,_,_,_=problem(tmp_path/'full',monkeypatch,spec=study.PHASES[1],index=4)
    definition,_=study.prepare_population(tmp_path/'full',t,z,state,theta,4,.65)
    assert len(definition['jobs'])==96 and all('first_hit_cutoff' not in j for j in definition['jobs'])


def test_result_state_crash_reconciles_even_exhausted_budget(tmp_path,monkeypatch):
    spec,t,z,state,theta,samples,_,calls,execute=problem(tmp_path,monkeypatch)
    spec['camera_cap']=960
    manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['phase']=spec;study.atomic_json(tmp_path/'manifest.json',manifest)
    definition,directions=study.prepare_population(tmp_path,t,z,state,theta,0,.65)
    execute(tmp_path,definition['jobs'],None)
    original=study.atomic_json
    def crash(path,value):
        if path.name=='search-state.json':raise OSError('result-before-state')
        original(path,value)
    monkeypatch.setattr(study,'atomic_json',crash)
    with pytest.raises(OSError,match='result-before-state'):study.finish_population(tmp_path,definition,directions,theta,z)
    assert study.committed_rows(tmp_path,state)==0 and (tmp_path/'g00-result.json').exists()
    monkeypatch.setattr(study,'atomic_json',original)
    frozen,updated=study.run_phase(tmp_path,spec,t,z,state,theta,0,None,time.time()-1)
    recovered=json.loads((tmp_path/'search-state.json').read_text())
    assert len(calls)==1 and recovered['completed_generations']==1
    assert frozen['used_cameras']==960 and frozen['unused_cameras']==0 and not frozen['valid']
    # Phase engine pins the next unexecuted draw before budget check, then skips it.
    assert frozen['unused_draw_skipped'] and frozen['next_global_generation']==2
    np.testing.assert_array_equal(updated,np.load(tmp_path/'theta_at1.npy'))
    assert study.run_phase(tmp_path,spec,t,z,recovered,updated,0,None,time.time()+100)[0]==frozen
    changed=dict(frozen,used_cameras=959);study.atomic_json(tmp_path/'training-endpoint.json',changed)
    with pytest.raises(ValueError,match='counters/lineage'):study.run_phase(tmp_path,spec,t,z,recovered,updated,0,None,time.time()+100)


def test_partial_failure_and_frozen_phase_boundary_preserved(tmp_path,monkeypatch):
    spec,t,z,state,theta,samples,_,calls,_=problem(tmp_path,monkeypatch,index=7)
    definition,_=study.prepare_population(tmp_path,t,z,state,theta,7,.65)
    samples[definition['jobs'][0]['id']]=3
    (tmp_path/'rollout-costs.jsonl').write_text(json.dumps(dict(job=definition['jobs'][0]['id'],ok=False))+'\n')
    frozen,updated=study.run_phase(tmp_path,spec,t,z,state,theta,7,None,time.time()+60)
    assert not calls and frozen['failed'] and not frozen['valid']
    assert frozen['used_cameras']==0 and frozen['unused_cameras']==3
    assert frozen['next_global_generation']==8 and frozen['next_rng_state']==definition['rng_state_after_draw']
    root=tmp_path/'root';root.mkdir()
    for name in ('search-transform.npy','radial-cap.json','projection.json'):
        if name.endswith('.npy'):np.save(root/name,np.eye(4))
        else:(root/name).write_text('{}')
    study.atomic_json(root/'seed-exclusions.json',dict(ids=[]))
    study.atomic_json(root/'manifest.json',dict(source={},runtime={},driver_hash='x',coordinator_hash='x',
        started_wall_ns=999,serve_seeds=[11,12]))
    nextdir,nextstate,value=study.initialize_phase(root,study.PHASES[1],updated,frozen['next_rng_state'],8,frozen['frozen_wall_ns'])
    np.testing.assert_array_equal(value,updated)
    assert nextstate['rng_state']==frozen['next_rng_state']
    with pytest.raises(ValueError,match='phase lineage/clock'):
        study.initialize_phase(root,study.PHASES[1],updated,frozen['next_rng_state'],8,frozen['frozen_wall_ns']+1)


def rally(success):
    return dict(own_returns=int(success),own_opportunities=1,
        first_incoming=dict(success=bool(success),tick=17,pre_ball=[.1,.4],pre_velocity=[-.4,.1]))


def test_final_cohorts_own_initial_stage_and_blank_are_separate(monkeypatch):
    jobs=study.evaluation_jobs(dict(serve_seeds=list(range(1,65))))
    assert len(jobs)==576 and all('first_hit_cutoff' not in j for j in jobs)
    assert all(j['weights_from']=='policy_initial.npy' for j in jobs if j['policy']=='initial_restricted')
    assert all(j['weights_from']=='policy_phase0.npy' for j in jobs if j['policy']=='phase0')
    assert all(j['config']['blank_camera']==(j['policy']=='blank') for j in jobs)
    monkeypatch.setattr(study,'SERVES',4)
    data={p:[rally(p in ('endpoint','phase0')) for _ in range(4)] for p in study.POLICIES}
    primary,secondary=study.evidence(data)
    assert primary['arms']['endpoint']['promising_development_signal']
    assert secondary['phase0_minus_initial']['first_incoming_lift']==1
    assert secondary['endpoint_minus_phase0']['first_incoming_lift']==0
    data['phase0']=[rally(False) for _ in range(4)]
    assert study.evidence(data)[0]==primary
    data['blank'][0]['first_incoming']['tick']=19
    with pytest.raises(ValueError,match='trajectory differs'):study.evidence(data)
