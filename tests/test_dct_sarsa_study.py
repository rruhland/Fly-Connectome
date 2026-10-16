import importlib.util
import json,sqlite3,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))

spec=importlib.util.spec_from_file_location('dct_sarsa_study',Path(__file__).parents[1]/'scripts/experimental/run_dct_sarsa_pong.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

def controls():
    return [dict(complete=True,episodes=24,wins=w,own_hit_followed_wins=h)
            for w,h in [(9,1),(10,2),(9,1),(10,2)]]

def test_fixed_condition_has_three_independent_zero_initial_training_seeds():
    jobs=study.declared_jobs();learn=[j for j in jobs if j['kind']=='learn'];evaluation=[j for j in jobs if j['kind']=='evaluate']
    assert len(learn)==len(evaluation)==3 and sum(j['transitions'] for j in learn)==90000
    assert tuple(j['seed'] for j in learn)==study.TRAINING_SEEDS
    assert not set(study.TRAINING_SEEDS).intersection(study.FINAL)
    for job in learn:
        assert job['alpha']==.001 and job['lam']==.995 and job['opponent_speed']==.25
        assert job['checkpoints']==[30000] and job['projection_from']=='projection.json'
        assert 'weights_from' not in job and job.get('action_hold_frames',1)==1
    for job in evaluation:
        assert job['config']['seed']==904271833 and job['config']['episodes']==24
        assert job['scaling_from'].endswith('_at30000.cap.json')

def test_all_three_endpoints_must_pass_independent_control_maxima():
    reports={f'eval_dct_{seed}':dict(complete=True,episodes=24,wins=14,own_hit_followed_wins=6)
             for seed in study.TRAINING_SEEDS}
    assert study.condition_decision(reports,controls())['qualifies']
    reports[f'eval_dct_{study.TRAINING_SEEDS[1]}']['wins']=13
    assert not study.condition_decision(reports,controls())['qualifies']
    reports.pop(f'eval_dct_{study.TRAINING_SEEDS[1]}')
    assert not study.condition_decision(reports,controls())['qualifies']
    changed=controls();changed[0].update(wins=13,own_hit_followed_wins=1);changed[1].update(wins=9,own_hit_followed_wins=5)
    assert study.thresholds(changed)==dict(total=17,own_hit=9)
    changed[0]['complete']=False
    with pytest.raises(ValueError):study.thresholds(changed)

def test_completion_pin_rejects_stale_report_and_partial_training_journal(tmp_path,monkeypatch):
    job=dict(id='learn',kind='learn',transitions=2)
    with sqlite3.connect(tmp_path/'learn.db') as c:
        c.execute('CREATE TABLE replay_samples (seq INTEGER)');c.execute('INSERT INTO replay_samples VALUES (0)')
    path=tmp_path/'learn.json';path.write_text(json.dumps(dict(complete=True,transitions=2)))
    assert study.complete_report(tmp_path,job) is None
    with pytest.raises(ValueError,match='journal row count'):study.completed_count(tmp_path,job)
    path.write_text(json.dumps(dict(complete=True,transitions=1)))
    with pytest.raises(ValueError,match='declared endpoint rows'):study.completed_count(tmp_path,job)
    with sqlite3.connect(tmp_path/'learn.db') as c:c.execute('INSERT INTO replay_samples VALUES (1)')
    path.write_text(json.dumps(dict(complete=True,transitions=2,wins=1)))
    # This unit fixture isolates pin/report integrity; artifact mapping and real
    # durable recovery are exercised in the projection worker integration test.
    monkeypatch.setattr(study,'journal_hash',lambda p:'semantic-journal-hash')
    monkeypatch.setattr(study,'endpoint_artifacts',lambda directory,job:{'weights':'fixed-artifact-hash'})
    study.pin_completion(tmp_path,job);assert study.complete_report(tmp_path,job)['wins']==1
    path.write_text(json.dumps(dict(complete=True,transitions=2,wins=2)))
    with pytest.raises(ValueError,match='artifacts changed'):study.complete_report(tmp_path,job)
