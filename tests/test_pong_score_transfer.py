"""Frozen-score allocation contracts and paired statistics; no camera data."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pong_score_transfer as study


def test_nine_cohorts_exact_task_and_real_worker_handoff(tmp_path,monkeypatch):
    jobs=study.declared_jobs(dict(serves=[101,103],control_actions=[107,109]))
    assert len(jobs)==18 and study.CAMERA_CAP==576*500
    for policy in study.POLICIES:
        group=[j for j in jobs if j['policy']==policy]
        assert [j['serve_index'] for j in group]==[0,1]
        for j in group:
            c=j['config'];trained=policy in (*study.ARMS,'seed1_initial','seed2_initial')
            assert c['opponent_speed']==.35 and c['max_episode_ticks']==1200 and c['episodes']==1
            assert not c['blank_camera'] and c['skip_baseline_perception']==(not trained)
            assert c['action_hold_frames']==(10 if policy=='random10' else 1)
            assert j['weights_from']==(policy+'.npy' if trained else None)
            assert 'camera_intervention' not in j and 'training_cutoff' not in j
    randoms=[j for j in jobs if j['policy'].startswith('random')]
    assert [j['config']['action_seed'] for j in randoms]==[107,109,107,109]
    worker=study.prior.engine.base;w=np.ones((3,4389))*.001
    study.prior.weights_file(tmp_path,'seed1.npy',w);(tmp_path/'radial-cap.json').write_text('{}')
    study.atomic_json(tmp_path/'manifest.json',dict(source=worker.source_signature(),runtime=worker.runtime_signature(),
        driver_hash=study.sha(worker.__file__),jobs=[jobs[0]]))
    class ReachedConstructor(Exception):pass
    def intercept(actual,config):
        np.testing.assert_array_equal(actual,w);assert config.opponent_speed==.35 and config.seed==101
        raise ReachedConstructor
    monkeypatch.setattr(worker,'FreshPongEvaluator',intercept)
    with pytest.raises(ReachedConstructor):worker.worker(tmp_path,jobs[0]['id'],emit=False)
    assert not list(tmp_path.glob('*.db'))


def test_seed_snapshot_covers_newer_domain_schedules_and_is_immutable(tmp_path):
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    (old/'manifest.json').write_text(json.dumps(dict(schedules=dict(development=[11,13],control_actions=[17,19]),
        seed_roots=dict(environment=23,learner=29),jobs=[dict(config=dict(seed=31,action_seed=2**31+103))])))
    (old/'seed-exclusions.json').write_text(json.dumps(dict(ids=[37,41])))
    study.snapshot_exclusions(new,old);before=(new/'seed-exclusions.json').read_bytes()
    assert {11,13,17,19,23,29,31,37,41,2**31+103,*study.prior.engine.FINAL}<=set(json.loads(before)['ids'])
    (old/'manifest.json').write_text('{}');study.snapshot_exclusions(new,old)
    assert (new/'seed-exclusions.json').read_bytes()==before and not (new/'manifest.json').exists()


def test_seed_recipe_uses_test_only_domains_and_rejects_reuse(monkeypatch):
    for key,value in [('DEV_SEED',104729),('ACTION_SEED',104743),('BOOTSTRAP',104759),('TAG',104761)]:
        monkeypatch.setattr(study,key,value)
    first=study.schedule([]);again=study.schedule(first['serves'][:5])
    assert len(set(again['serves']))==64 and again['serves'][:59]==first['serves'][5:]
    assert not set(again['serves'])&set(study.prior.engine.FINAL)
    with pytest.raises(ValueError,match='root collision'):study.schedule([104729])


def rally(win=False,hit=False):
    return dict(reward=1. if win else -1.,own_returns=int(hit),own_opportunities=1,truncated=False,
        first_incoming=dict(success=hit,tick=103,pre_ball=[.07,.3],pre_velocity=[-.5,.1]))


def test_independent_control_maxima_cluster_bootstrap_and_both_seed_gate(monkeypatch):
    monkeypatch.setattr(study,'SERVES',8)
    groups={n:[rally() for _ in range(8)] for n in study.POLICIES}
    for n in study.ARMS:groups[n]=[rally(True,True) for _ in range(8)]
    groups['initial']=[rally(True,False) if i<4 else rally() for i in range(8)]
    groups['seed2_initial']=[rally(True,True) if i<2 else rally() for i in range(8)]
    a=study.evidence(groups)
    assert a['replicated_score_transfer']
    for v in a['arms'].values():assert v['total']['lift']==.5 and v['own_hit']['lift']==.75
    assert a['fixed_two_policy_average']['total']['lift']==.5
    groups['seed2']=[rally() for _ in range(8)]
    b=study.evidence(groups)
    assert not b['replicated_score_transfer'] and b['arms']['seed1']['passes'] and not b['arms']['seed2']['passes']
    bad=copy.deepcopy(groups);bad['up'][0]['first_incoming']['tick']+=1
    with pytest.raises(ValueError,match='geometry differs'):study.evidence(bad)
    groups['up'].pop();assert not study.evidence(groups)['complete']


def test_natural_wins_and_timeouts_do_not_masquerade_as_own_hit_wins():
    rows=[rally(True,False),rally(True,True),dict(rally(False,True),reward=0.,truncated=True)]
    assert study.score_counts(rows)==dict(wins=2,own_hit_followed_wins=1,natural_serve_wins=1,losses=0,timeouts=1)


def test_manifest_resume_source_pins_and_frozen_input_rejection(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'SERVES',2)
    for key,value in [('DEV_SEED',104729),('ACTION_SEED',104743),('BOOTSTRAP',104759),('TAG',104761)]:monkeypatch.setattr(study,key,value)
    fixed={'seed1.npy':None,'seed2.npy':None}
    for n in fixed:(tmp_path/n).write_bytes(b'fixture');fixed[n]=study.sha(tmp_path/n)
    study.atomic_json(tmp_path/'lineage-proof.json',dict(current_source=study.prior.source_signature(),
        runtime=study.prior.runtime_signature(),fixed_artifacts=fixed,policies={}))
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=list(study.prior.engine.FINAL)))
    study.atomic_json(tmp_path/'schedules.json',study.schedule(list(study.prior.engine.FINAL)))
    study.atomic_json(tmp_path/'prelaunch-checks.json',dict(passed=True,source=study.prior.source_signature()))
    assert not (tmp_path/'manifest.json').exists()
    m=study.prepare(tmp_path);assert 'started_wall_ns' not in m
    m['started_wall_ns']=123456789;study.atomic_json(tmp_path/'manifest.json',m)
    before=(tmp_path/'manifest.json').read_bytes()
    assert study.prepare(tmp_path)==m and (tmp_path/'manifest.json').read_bytes()==before
    assert len(m['jobs'])==18 and len(m['primary_controls'])==7
    (tmp_path/'seed1.npy').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='policy/transform changed'):study.prepare(tmp_path)


def test_completed_outcome_keeps_original_wall_eligibility_and_rejects_tamper(tmp_path,monkeypatch):
    m=dict(jobs=[]);summary=dict(status='partial',overall_budget_met=False,wall_seconds=1201.)
    for n in ('manifest.json','return-annotations.json'):study.atomic_json(tmp_path/n,{})
    study.atomic_json(tmp_path/'study-completion.json',dict(summary=summary,evidence=study.outcome_evidence(tmp_path,m)))
    study.atomic_json(tmp_path/'summary.json',summary)
    assert study.retained_outcome(tmp_path,m)==summary
    study.atomic_json(tmp_path/'summary.json',dict(summary,status='complete'))
    with pytest.raises(ValueError,match='summary changed'):study.retained_outcome(tmp_path,m)


def test_scoring_inclusive_budget_and_completed_resume_never_rescore(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'SERVES',1)
    jobs=study.declared_jobs(dict(serves=[101],control_actions=[103]))
    m=dict(jobs=jobs,started_wall_ns=1_000_000_000)
    study.atomic_json(tmp_path/'manifest.json',m)
    monkeypatch.setattr(study,'prepare',lambda d:m)
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(study,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(study.time,'time_ns',lambda:2_000_000_000)
    monkeypatch.setattr(study.time,'time',lambda:1202.)
    def execute(*args,**kwargs):
        (tmp_path/'rollout-costs.jsonl').write_text(json.dumps(dict(ok=True))+'\n')
        return dict(complete=True,failed=False,committed_cameras=0)
    monkeypatch.setattr(study.prior.engine,'execute_jobs',execute)
    monkeypatch.setattr(study.prior.engine,'counts',lambda *args:dict(training_fitness=0,fresh_evaluation=0))
    monkeypatch.setattr(study.prior.engine,'read_report',lambda *args:dict(transitions=0,initial_cameras=1))
    monkeypatch.setattr(study.prior.completion,'complete_report',lambda *args:dict(complete=True))
    row=dict(rally(),maximum_consecutive_returns=0,seconds=1.,unfinished_incoming_approach=False,
        opponent_opportunities=0,opponent_returns=0,post_own_opponent_opportunities=0,
        post_own_opponent_returns=0,natural_serve_win=False)
    monkeypatch.setattr(study.prior.physical,'annotate',lambda *args:{j['id']:dict(rallies=[row],verified_camera_rows=0) for j in jobs})
    result=study.supervise(tmp_path)
    assert result['status']=='partial' and not result['evidence_eligible'] and result['wall_seconds']==1201.
    monkeypatch.setattr(study.prior.engine,'execute_jobs',lambda *args,**kwargs:pytest.fail('retained outcome must not execute'))
    monkeypatch.setattr(study,'evidence',lambda *args:pytest.fail('retained outcome must not rescore'))
    assert study.supervise(tmp_path)==result
