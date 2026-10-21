"""Metadata-only continuation lifecycle, argument handoff and paired statistics."""
import copy,json,sys,time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pendulum_continuation as study


def test_seed_contract_uses_test_roots_and_full_width_exclusions(monkeypatch):
    monkeypatch.setattr(study,'ROOTS',dict(environment=104729,learner=104743,development=104759,control_actions=104761))
    monkeypatch.setattr(study,'TAG',104773);monkeypatch.setattr(study,'BOOTSTRAP',104779)
    a=study.schedule([2**31+17]);b=study.schedule([*a['development'][:5],2**31+17])
    assert len(set(b['development']))==len(set(b['control_actions']))==64
    assert b['development'][:59]==a['development'][5:]
    assert not set(b['control_actions'])&set(study.ROOTS.values())
    with pytest.raises(ValueError,match='root collision'):study.schedule([104729])


@pytest.mark.parametrize('enabled',[True,False])
def test_worker_loads_owned_projected_weights_and_mode_before_construction(tmp_path,monkeypatch,enabled):
    w=study.worker;theta=np.arange(3*214,dtype=float).reshape(3,214)*.0001
    np.save(tmp_path/'initial.npy',theta,allow_pickle=False)
    for n in ('vision.pkl','search-transform.npy'):(tmp_path/n).write_bytes(b'fixture')
    job=dict(id='test_train',kind='train',phase='training',snapshot_from='vision.pkl',
        initial_weights_from='initial.npy',learn_enabled=enabled,
        input_hashes={n:study.sha(tmp_path/n) for n in ('initial.npy','vision.pkl','search-transform.npy')},
        config=dict(seed=104729,learner_seed=104743))
    study.atomic_json(tmp_path/'manifest.json',dict(source=w.source_signature(),runtime=w.runtime_signature(),
        pinned_sources={},projection={},jobs=[job]))
    study.atomic_json(tmp_path/'test_train-spec.json',job)
    ns=time.time_ns()+30_000_000_000;study.atomic_json(tmp_path/'training-boundary.json',dict(deadline_wall_ns=ns))
    monkeypatch.setattr(w,'load_snapshot',lambda *args:SimpleNamespace(content_hash='fixture',learning_observations=22282))
    class ReachedConstructor(Exception):pass
    def intercept(**kwargs):
        np.testing.assert_array_equal(kwargs['weights'],theta)
        assert kwargs['learn_enabled'] is enabled and kwargs['learner_seed']==104743
        raise ReachedConstructor
    monkeypatch.setattr(w,'PendulumLearningSession',intercept)
    receipt=w.execute_job(tmp_path,job,ns/1e9)
    assert not receipt['ok'] and 'ReachedConstructor' in receipt['traceback'] and receipt['wall_seconds']>=0
    assert not list(tmp_path.glob('*.db'))
    del job['input_hashes']['initial.npy']
    m=json.loads((tmp_path/'manifest.json').read_text());m['jobs']=[job]
    study.atomic_json(tmp_path/'manifest.json',m);study.atomic_json(tmp_path/'test_train-spec.json',job)
    with pytest.raises(ValueError,match='referenced pendulum inputs'):w.validate_inputs(tmp_path,job)


def test_original_competence_and_incremental_mean_cap_contrasts_are_distinct():
    def rows(seconds,cap=False,count=64):
        return [dict(seed=1000+i,serve_index=i,survival_seconds=seconds,time_limit=cap) for i in range(count)]
    groups={n:rows(1.) for n in study.metrics.CONTROLS}
    groups.update(on_endpoint=rows(5.,True),off_endpoint=rows(4.),start=rows(4.9))
    groups.update(on_at5000=rows(4.5,count=32),off_at5000=rows(3.5,count=32),
                  on_at10000=rows(4.7,count=32),off_at10000=rows(3.9,count=32))
    r=study.survival_evidence(groups)
    assert r['arms']['on']['competence_screen'] and not r['arms']['off']['competence_screen']
    assert r['on_minus_off']['point']==r['on_minus_off_cap_fraction']['point']==1.
    assert r['incremental_vs_start']['on']['mean_seconds']['point']==pytest.approx(.1)
    assert r['incremental_vs_start']['off']['mean_seconds']['point']==pytest.approx(-.9)
    assert r['fixed32_curves_descriptive']['on']['start20k']['mean_survival_seconds']==pytest.approx(4.9)
    bad=copy.deepcopy(groups);bad['start'][0]['seed']+=1
    with pytest.raises(ValueError,match='starting-head cohort'):study.survival_evidence(bad)
    bad=copy.deepcopy(groups);bad['off_at5000'].reverse()
    with pytest.raises(ValueError,match='curve cohort order'):study.survival_evidence(bad)
    groups['start'].pop();partial=study.survival_evidence(groups)
    assert not partial['complete'] and partial['arms']['on']['competence_screen']


@pytest.mark.parametrize('mismatch',[None,'config','snapshot'])
def test_off_a_reuse_requires_full_config_snapshot_and_raw_evidence(tmp_path,monkeypatch,mismatch):
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir();monkeypatch.setattr(study,'OLD',old)
    (old/'vision.pkl').write_bytes(b'old retained bytes');(old/'weights.npy').write_bytes(b'old fixed A head')
    cap={};study.atomic_json(new/'a-cap.json',cap)
    jobs=[]
    for i,seed in enumerate(range(1,65)):
        name=f'a_experienced_s{i:02d}'
        config=asdict(study.PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap))
        if mismatch=='config' and i==0:config['blank_camera']=True
        jobs.append(dict(id=name,kind='pong',config=config,learn_enabled=False,episode_seeds=[seed],
            weights_from='weights.npy',snapshot_from='vision.pkl',
            input_hashes={'vision.pkl':study.sha(old/'vision.pkl') if mismatch!='snapshot' else 'different'}))
        study.atomic_json(old/(name+'.json'),dict(complete=True))
        study.atomic_json(old/(name+'-completion.json'),dict(report_sha256=study.sha(old/(name+'.json')),journal_hash='journal'))
    study.atomic_json(old/'manifest.json',dict(jobs=jobs));study.atomic_json(old/'a-after-annotations.json',{})
    monkeypatch.setattr(study.worker,'load_snapshot',lambda *args:SimpleNamespace(content_hash='same',learning_observations=22282))
    monkeypatch.setattr(study.worker,'journal_hash',lambda *args:'journal')
    m=dict(a_reused_development_seeds=list(range(1,65)),fixed_artifacts={'a-weights.npy':study.sha(old/'weights.npy')},
        source_bridges={'initial':dict(ancestor_file_sha256=study.sha(old/'vision.pkl'))})
    proof=study.off_a_reuse(new,m)
    assert proof['reuse']==(mismatch is None) and proof['full_reset_vision_content_equal']
    assert proof['reused_trials']==(64 if mismatch is None else 0)


def test_custom_phase_caps_and_frozen_budget_survive_resume(tmp_path,monkeypatch):
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(study.base,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(study.base.time,'time_ns',lambda:1500)
    m=dict(started_wall_ns=1000)
    train=study.base.execute_phase(tmp_path,m,'training',[],2,phase_seconds=20,total_seconds=50)
    assert train['deadline_wall_ns']==1000+20_000_000_000
    evaluation=study.base.execute_phase(tmp_path,m,'evaluation',[],4,phase_seconds=30,total_seconds=50)
    assert evaluation['started_wall_ns']==1500 and evaluation['deadline_wall_ns']==1500+30_000_000_000
    monkeypatch.setattr(study.base.time,'time_ns',lambda:1_000_000_000_000)
    assert study.base.execute_phase(tmp_path,m,'evaluation',[],4,phase_seconds=30,total_seconds=50)==evaluation
    train['failed']=True;study.atomic_json(tmp_path/'training-boundary.json',train)
    with pytest.raises(ValueError,match='completion changed'):
        study.base.execute_phase(tmp_path,m,'training',[],2,phase_seconds=20,total_seconds=50)
