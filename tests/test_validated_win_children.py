"""Synthetic child-selection transactions: no cameras, physics or study roots."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_validated_win_children as study


@pytest.fixture
def toy(tmp_path,monkeypatch):
    roots=dict(training=104729,optimizer=104743,development=104759,
               control_actions=104761,bootstrap=104773)
    monkeypatch.setattr(study,'ROOTS',roots);monkeypatch.setattr(study,'TAG',99731)
    monkeypatch.setattr(study,'MAX_CYCLES',1);monkeypatch.setattr(study,'MIN_CYCLES',1)
    rng=np.random.default_rng(4321);z=rng.standard_normal((32,2))
    theta=.3*study.sample_directions(rng,z,1)[0]
    matrix=rng.standard_normal((2,5));np.save(tmp_path/'search-transform.npy',matrix)
    study.immutable_array(tmp_path/'theta_at0.npy',theta)
    study.weights_file(tmp_path,'start.npy',theta@matrix)
    initial=dict(theta_hash=study.array_hash(theta),next_rng_state=np.random.default_rng(roots['optimizer']).bit_generator.state)
    study.atomic_json(tmp_path/'boundary-proof.json',initial)
    state=dict(completed_cycles=0,accepted_updates=0,next_cycle=0,theta_file='theta_at0.npy',
               theta_hash=initial['theta_hash'],rng_state=initial['next_rng_state'])
    study.atomic_json(tmp_path/'search-state.json',state)
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=[2**31+59]))
    manifest=dict(jobs=[],serve_seeds=list(range(201,265)),action_seeds=[2**31+61],
                  frozen_file_hashes={'search-transform.npy':study.sha(tmp_path/'search-transform.npy')},
                  started_wall_ns=time.time_ns())
    study.atomic_json(tmp_path/'manifest.json',manifest)
    reports={};samples={};calls=[]
    monkeypatch.setattr(study.engine,'sample_count',lambda directory,name:samples.get(name,0))
    monkeypatch.setattr(study.completion,'complete_report',lambda directory,job:reports.get(job['id']))
    monkeypatch.setattr(study,'training_failed',lambda directory:False)
    def no_execute(*args):
        calls.append(1);raise AssertionError('zero-camera fixture must not dispatch work')
    monkeypatch.setattr(study,'execute',no_execute)
    def fill(jobs,wins,limit=None):
        for j in jobs[:limit]:
            win=int(j['serve_index']<wins[j['policy_index']])
            samples[j['id']]=5
            reports[j['id']]=dict(complete=True,episodes=1,wins=win,total_reward=1. if win else -1.,
                                   transitions=5,hits=987654)
    return dict(directory=tmp_path,z=z,theta=theta,matrix=matrix,state=state,manifest=manifest,
                reports=reports,samples=samples,calls=calls,fill=fill)


def population(toy):
    return study.prepare_cycle(toy['directory'],toy['z'],toy['state'],toy['theta'],())


def duel(toy,definition,net):
    toy['fill'](definition['discovery_jobs'],[2,5,2,1,2])
    jobs=study.validation_jobs(definition,0);study.engine.register_jobs(toy['directory'],jobs)
    toy['fill'](jobs,[3,3+net]);return jobs


def test_discovery_rejection_consumes_sixty_and_reserves_unused_validation(toy):
    d=population(toy);toy['fill'](d['discovery_jobs'],[4,4,4,3,2])
    frozen,state,theta=study.train(toy['directory'],None,toy['manifest'],toy['z'],toy['state'],toy['theta'])
    assert frozen['valid'] and state['completed_cycles']==1 and state['accepted_updates']==0
    assert frozen['used_training_cameras']==frozen['total_training_phase_cameras']==300
    assert len(json.loads((toy['directory']/'manifest.json').read_text())['jobs'])==60
    assert not toy['calls'];np.testing.assert_array_equal(theta,toy['theta'])
    assert study.transaction.choose_child([1,4,4,2,1])==0  # stable child tie
    next_draw=study.prepare_cycle(toy['directory'],toy['z'],state,theta,())
    assert not set(d['discovery_seeds']+d['validation_seeds'])&set(next_draw['discovery_seeds']+next_draw['validation_seeds'])
    assert not set(d['validation_seeds'])&{j['config']['seed'] for j in d['discovery_jobs']}


@pytest.mark.parametrize('net,accepted',[(0,False),(1,False),(2,True)])
def test_validation_margin_commits_only_exact_executed_child(toy,net,accepted):
    d=population(toy);duel(toy,d,net)
    state,theta=study.transaction.finish_cycle(study,toy['directory'],d,toy['state'],toy['theta'])
    expected=np.load(toy['directory']/d['child_theta_files'][0]) if accepted else toy['theta']
    np.testing.assert_array_equal(theta,expected)
    np.testing.assert_array_equal(np.load(toy['directory']/'incumbent_after_cycle1.npy'),expected@toy['matrix'])
    result=json.loads((toy['directory']/'c00-result.json').read_text())
    assert result['accepted']==accepted and state['accepted_updates']==int(accepted)
    assert result['committed_training_cameras']==540 and study.transaction.committed_rows(study,toy['directory'],state)==540


def test_incomplete_duel_keeps_incumbent_and_all_costs_and_consumed_rng(toy,monkeypatch):
    d=population(toy);toy['fill'](d['discovery_jobs'],[2,5,2,1,2])
    jobs=study.validation_jobs(d,0);study.engine.register_jobs(toy['directory'],jobs)
    toy['fill'](jobs,[3,5],47);monkeypatch.setattr(study,'TRAIN_CAMERAS',535)
    frozen,state,theta=study.train(toy['directory'],None,toy['manifest'],toy['z'],toy['state'],toy['theta'])
    assert not frozen['valid'] and state==toy['state'] and not toy['calls']
    assert frozen['used_training_cameras']==0 and frozen['unused_partial_cycle_cameras']==535
    assert frozen['unused_draw_skipped'] and frozen['next_cycle']==1 and frozen['next_rng_state']==d['rng_state_after_draw']
    np.testing.assert_array_equal(theta,toy['theta'])
    assert not (toy['directory']/'c00-result.json').exists()
    assert study.train(toy['directory'],None,toy['manifest'],toy['z'],state,theta)[0]==frozen
    frozen['next_rng_state']=toy['state']['rng_state']
    study.atomic_json(toy['directory']/'training-endpoint.json',frozen)
    with pytest.raises(ValueError,match='frozen incumbent'):
        study.train(toy['directory'],None,toy['manifest'],toy['z'],state,theta)


@pytest.mark.parametrize('failed',[False,True])
def test_result_state_crash_reconciles_once_at_exhausted_budget(toy,monkeypatch,failed):
    d=population(toy);duel(toy,d,2);original=study.atomic_json
    def crash(path,value):
        if path.name=='search-state.json':raise OSError('result-state crash')
        original(path,value)
    monkeypatch.setattr(study,'atomic_json',crash)
    with pytest.raises(OSError,match='result-state crash'):
        study.transaction.finish_cycle(study,toy['directory'],d,toy['state'],toy['theta'])
    assert (toy['directory']/'c00-result.json').exists()
    assert study.transaction.committed_rows(study,toy['directory'],toy['state'])==0
    monkeypatch.setattr(study,'atomic_json',original);monkeypatch.setattr(study,'TRAIN_CAMERAS',540)
    monkeypatch.setattr(study,'TRAIN_SECONDS',0.)
    monkeypatch.setattr(study,'MAX_CYCLES',2)
    monkeypatch.setattr(study,'training_failed',lambda directory:failed)
    frozen,state,theta=study.train(toy['directory'],None,toy['manifest'],toy['z'],toy['state'],toy['theta'])
    assert state['completed_cycles']==state['accepted_updates']==1 and frozen['used_training_cameras']==540
    assert frozen['unused_partial_cycle_cameras']==0 and not toy['calls']
    assert frozen['training_failure']==failed and frozen['valid'] is not failed
    if failed:assert frozen['training_stop']=='persisted training failure'
    np.testing.assert_array_equal(theta,np.load(toy['directory']/d['child_theta_files'][0]))
    hashes={p.name:study.sha(p) for p in toy['directory'].glob('*') if p.is_file()}
    assert study.train(toy['directory'],None,toy['manifest'],toy['z'],state,theta)[0]==frozen
    assert {p.name:study.sha(p) for p in toy['directory'].glob('*') if p.is_file()}==hashes


def test_committed_incumbent_mapped_head_tamper_rejected(toy):
    d=population(toy);duel(toy,d,2)
    state,_=study.transaction.finish_cycle(study,toy['directory'],d,toy['state'],toy['theta'])
    path=toy['directory']/'incumbent_after_cycle1.npy';value=np.load(path);value.flat[0]+=1.;np.save(path,value)
    with pytest.raises(ValueError,match='mapped readout'):
        study.transaction.committed_rows(study,toy['directory'],state)


@pytest.mark.parametrize('stage',['discovery','validation'])
@pytest.mark.parametrize('mutation',['missing','duplicate'])
def test_full_unique_sixty_or_forty_eight_cells_required(toy,stage,mutation):
    d=population(toy)
    jobs=d['discovery_jobs'] if stage=='discovery' else study.validation_jobs(d,0)
    seeds=d['discovery_seeds'] if stage=='discovery' else d['validation_seeds']
    roles=list(d['readouts']) if stage=='discovery' else ['incumbent','child0']
    weights=[d['readouts'][r]['weight_file'] for r in roles]
    bad=copy.deepcopy(jobs[:-1] if mutation=='missing' else [*jobs[:-1],jobs[0]])
    with pytest.raises(ValueError,match='roster|cell'):study.collect_wins(toy['directory'],bad,seeds,weights)


@pytest.mark.parametrize('mutation',['child_theta','mapped_weights','theta_hash','cycle','rng'])
def test_readout_tamper_and_stale_lineage_cannot_commit(toy,mutation):
    d=population(toy);duel(toy,d,2)
    if mutation in ('child_theta','mapped_weights'):
        field='theta_file' if mutation=='child_theta' else 'weight_file'
        path=toy['directory']/d['readouts']['child0'][field];value=np.load(path);value.flat[0]+=1.;np.save(path,value)
    elif mutation=='theta_hash':d['input_theta_hash']='0'*64
    elif mutation=='cycle':d['cycle']=1
    else:d['input_rng_state']=np.random.default_rng(999).bit_generator.state
    with pytest.raises(ValueError,match='stale|parameters|pinned'):
        study.transaction.finish_cycle(study,toy['directory'],d,toy['state'],toy['theta'])
    assert not (toy['directory']/'c00-result.json').exists()


def test_win_only_objective_ignores_hits_and_loss_timeout_difference(toy):
    d=population(toy);toy['fill'](d['discovery_jobs'],[2,5,2,1,2]);first=study.discovery_result(toy['directory'],d)
    for report in toy['reports'].values():
        report['hits']=0
        if not report['wins']:report['total_reward']=0.
    second=study.discovery_result(toy['directory'],d)
    assert first['wins_by_policy']==second['wins_by_policy'] and first['selected_child']==second['selected_child']==0
    one=next(iter(toy['reports'].values()));one['total_reward']=0. if one['wins'] else 1.
    with pytest.raises(ValueError,match='terminal win'):study.discovery_result(toy['directory'],d)


def test_persisted_failure_and_current_seed_exclusions(toy,monkeypatch):
    monkeypatch.setattr(study,'training_failed',lambda directory:True)
    frozen,_,_=study.train(toy['directory'],None,toy['manifest'],toy['z'],toy['state'],toy['theta'])
    assert frozen['training_failure'] and not frozen['valid'] and not toy['calls']
    (toy['directory']/'training-endpoint.json').unlink()
    monkeypatch.setattr(study,'training_failed',lambda directory:False);seen={}
    def capture(directory,z,state,theta,excluded):
        seen['excluded']=set(excluded);raise RuntimeError('definition only')
    monkeypatch.setattr(study,'prepare_cycle',capture)
    with pytest.raises(RuntimeError,match='definition only'):
        study.train(toy['directory'],None,toy['manifest'],toy['z'],toy['state'],toy['theta'])
    assert {2**31+59,2**31+61}<=seen['excluded']
    scheduled=study.schedule([])
    assert len(set(scheduled['serves']))==len(set(scheduled['control_actions']))==64
    assert not set(scheduled['serves'])&set(scheduled['control_actions'])
    with pytest.raises(ValueError,match='root collision'):study.schedule([study.ROOTS['optimizer']])


def test_ten_cohort_max_and_paired_incremental_statistics(toy):
    groups={}
    for name in study.POLICIES:
        wins=40 if name=='endpoint' else 25 if name=='start' else 20
        groups[name]=[dict(reward=1. if i<wins else -1.,own_returns=int(8<=i<wins),
            first_incoming=None if i<8 else dict(tick=100+i,success=i<wins)) for i in range(64)]
    result=study.evidence(groups,draws=2000)
    assert len(groups)==10 and len(result['controls'])==9 and result['qualifies']
    assert result['endpoint']['total']['lift']==15/64
    assert result['endpoint_minus_start']['total']['lift']==15/64
    assert result['endpoint_minus_original']['total']['lift']==20/64
    for name in ('endpoint','endpoint_minus_start','endpoint_minus_original'):
        assert result[name]['total']['lift']==result[name]['own_hit']['lift']
        assert result[name]['total']['ci95']==result[name]['own_hit']['ci95']
    assert result['endpoint']['total']['strongest_control_rate']-result['endpoint']['own_hit']['strongest_control_rate']==8/64
    # The added original head participates in the strongest-control screen.
    groups['original']=copy.deepcopy(groups['endpoint'])
    assert not study.evidence(groups,draws=2000)['qualifies']
    broken=copy.deepcopy(groups);broken['random10'][10]['first_incoming']['tick']+=1
    with pytest.raises(ValueError,match='geometry'):study.evidence(broken,draws=100)


def test_retained_completion_keeps_original_clock_and_eligibility(toy,monkeypatch):
    monkeypatch.setattr(study.transfer,'outcome_evidence',lambda directory,m:dict(raw='fixed'))
    p=toy['directory'];study.atomic_json(p/'c00-result.json',dict(value=1))
    summary=dict(wall_seconds=12.,evidence_eligible=True,overall_budget_met=True)
    study.atomic_json(p/'study-completion.json',dict(summary=summary,evidence=study.outcome_evidence(p,toy['manifest'])))
    study.atomic_json(p/'summary.json',summary);monkeypatch.setattr(study.time,'time',lambda:1e15)
    assert study.retained_outcome(p,toy['manifest'])==summary
    study.atomic_json(p/'c00-result.json',dict(value=2))
    with pytest.raises(ValueError,match='retained child evidence'):study.retained_outcome(p,toy['manifest'])


def test_draft_guard_is_explicit_and_side_effect_free(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'DRAFT_ONLY',True);monkeypatch.setattr(study,'ROOTS',None)
    with pytest.raises(ValueError,match='draft only'):study.stage(tmp_path/'not-created')
    with pytest.raises(ValueError,match='draft only'):study.supervise(tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()
