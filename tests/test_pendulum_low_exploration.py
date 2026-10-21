"""Zero-camera fixtures; actual candidate roots stay unset during static work."""
import copy,json,sys
from dataclasses import asdict
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pendulum_low_exploration as study


def test_draft_guard_does_not_create_or_expand(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'DRAFT_ONLY',True)
    for name in ('NEW_ROOTS','TAG','BOOTSTRAP'):monkeypatch.setattr(study,name,None)
    with pytest.raises(ValueError,match='draft only'):study.stage(tmp_path/'unallocated')
    with pytest.raises(ValueError,match='draft only'):study.supervise(tmp_path/'unallocated')
    assert not (tmp_path/'unallocated').exists()


def test_seed_exception_is_only_old_training_roots(monkeypatch):
    monkeypatch.setattr(study,'NEW_ROOTS',dict(development=104729,control_actions=104743))
    monkeypatch.setattr(study,'TAG',104759);monkeypatch.setattr(study,'BOOTSTRAP',104761)
    first=study.schedule([*study.REUSED_ROOTS.values(),2**31+13])
    second=study.schedule([*study.REUSED_ROOTS.values(),*first['development'][:3],2**31+13])
    assert len(set(second['development']))==len(set(second['control_actions']))==64
    assert not set(second['development'])&set(first['development'][:3])
    assert second['development'][:61]==first['development'][3:]
    with pytest.raises(ValueError,match='root collision'):study.schedule([104729])


def test_only_epsilon_changes_from_fixed_control_config():
    m=dict(fixed_artifacts={n:'hash' for n in ('initial-vision.pkl','initial-theta.npy','search-transform.npy')})
    job=study.training_jobs(m)[0]
    expected=dict(seed=1083294751,learner_seed=1647809237,max_episode_ticks=600,
                  alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95)
    assert {k for k,v in job['config'].items() if v!=expected[k]}=={'epsilon'}
    assert job['config']['epsilon']==.02 and job['learn_enabled'] is True
    assert job['camera_dose']==20000 and job['checkpoints']==[5000,10000,20000]
    assert job['initial_weights_from']=='initial-theta.npy'


def test_new_evaluation_has_704_jobs_and_reuses_no_hidden_compute(tmp_path):
    m=dict(schedules=dict(development=list(range(1000,1064)),control_actions=list(range(2000,2064))),
           a_reused_development_seeds=list(range(3000,3064)))
    for name in ('initial','a',*(f'{arm}_at{dose}' for arm in ('low','control') for dose in (5000,10000,20000))):
        study.atomic_json(tmp_path/(name+'-cap.json'),{})
        for suffix in ('-vision.pkl','-weights.npy'):(tmp_path/(name+suffix)).write_bytes(name.encode())
    (tmp_path/'generic-vision.pkl').write_bytes(b'generic')
    jobs=study.evaluation_jobs(tmp_path,m)
    assert len(jobs)==704 and len({j['id'] for j in jobs})==704
    assert sum(j['kind']=='pong' for j in jobs)==64
    assert all(j['group']=='low_a_after' for j in jobs if j['kind']=='pong')
    assert sum(j['group']=='control_endpoint' for j in jobs)==64
    assert sum(j['group']=='control_at5000' for j in jobs)==32
    for j in jobs:
        assert j['snapshot_from'] in j['input_hashes']
        if j['weights_from'] is not None:assert j['weights_from'] in j['input_hashes']


def test_competence_incremental_and_mechanism_are_distinct(monkeypatch):
    monkeypatch.setattr(study,'BOOTSTRAP',104729)
    def rows(value,cap=False,count=64):
        return [dict(seed=1000+i,serve_index=i,survival_seconds=value,time_limit=cap) for i in range(count)]
    groups={name:rows(1.) for name in study.metrics.CONTROLS}
    groups.update(low_endpoint=rows(4.,True),control_endpoint=rows(3.5),start=rows(3.9))
    for arm in ('low','control'):
        for dose in (5000,10000):groups[f'{arm}_at{dose}']=rows(3.,count=32)
    result=study.survival_evidence(groups)
    assert result['complete'] and result['arms']['low']['competence_screen']
    assert not result['arms']['control']['competence_screen']
    assert result['lower_exploration_development_signal']
    assert result['incremental_vs_start']['low']['mean_seconds']['point']==pytest.approx(.1)
    assert result['low_minus_control_cap_fraction']['point']==1.
    bad=copy.deepcopy(groups);bad['control_at5000'].reverse()
    with pytest.raises(ValueError,match='curve cohort order'):study.survival_evidence(bad)
    groups['low_endpoint']=rows(2.9,True)
    result=study.survival_evidence(groups)
    assert not result['arms']['low']['competence_screen']
    assert not result['lower_exploration_development_signal']


def test_reused_control_configuration_and_ancestor_tamper_rejected(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();ancestor=tmp_path/'ancestor';ancestor.mkdir();audit=tmp_path/'audit';audit.mkdir()
    monkeypatch.setattr(study,'OLD',old);monkeypatch.setattr(study.prior,'OLD',ancestor);monkeypatch.setattr(study,'AUDIT',audit)
    monkeypatch.setattr(study,'source_signature',lambda:{'fixture':'source'})
    monkeypatch.setattr(study,'runtime_signature',lambda:{'fixture':'runtime'})
    monkeypatch.setattr(study.worker,'complete_report',lambda *args:dict(exports={}))
    monkeypatch.setattr(study,'archived_a_reference',lambda files:dict(separately_tested=True))
    study.atomic_json(ancestor/'a-after-annotations.json',{})
    config=dict(seed=1083294751,learner_seed=1647809237,max_episode_ticks=600,
                alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95)
    train=dict(id='on_train',config=config,learn_enabled=True,camera_dose=20000,checkpoints=[5000,10000,20000],
        snapshot_from='initial-vision.pkl',initial_weights_from='initial-theta.npy')
    jobs=[train]
    for i,seed in enumerate(range(1,65)):
        name=f'a_on_s{i:02d}'
        jobs.append(dict(id=name,config=asdict(study.PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,
            opponent_speed=.65,policy_scaling={})),learn_enabled=False,episode_seeds=[seed],
            snapshot_from='on_at20000-vision.pkl',weights_from='a-weights.npy'))
        for suffix in ('.json','-completion.json','-spec.json'):study.atomic_json(old/(name+suffix),{})
    m=dict(source={'fixture':'source'},runtime={'fixture':'runtime'},jobs=jobs,a_reused_development_seeds=list(range(1,65)),
        ancestor_evidence={str(ancestor/'a-after-annotations.json'):study.sha(ancestor/'a-after-annotations.json')})
    study.atomic_json(old/'manifest.json',m);study.atomic_json(old/'summary.json',dict(status='complete'))
    for path in (old/'completion-verification.json',audit/'summary.json'):study.atomic_json(path,dict(passed=True))
    for name in ('study-completion.json','a-after-annotations.json','a-off-reuse-proof.json','on_train.json','on_train-completion.json','a-cap.json'):
        study.atomic_json(old/name,{})
    assert study.reused_control_proof()['inherited_control_training_transitions']==20000
    jobs[0]['config']['epsilon']=.1;study.atomic_json(old/'manifest.json',m)
    with pytest.raises(ValueError,match='epsilon-only'):study.reused_control_proof()
    jobs[0]['config']['epsilon']=.2;jobs[1]['config']['blank_camera']=True;study.atomic_json(old/'manifest.json',m)
    with pytest.raises(ValueError,match='A configuration'):study.reused_control_proof()
    jobs[1]['config']['blank_camera']=False;study.atomic_json(old/'manifest.json',m)
    study.atomic_json(ancestor/'a-after-annotations.json',dict(tampered=True))
    with pytest.raises(ValueError,match='audited ancestor'):study.reused_control_proof()


def test_archived_a_bytes_and_audited_semantic_pins_are_bound(tmp_path,monkeypatch):
    ancestor=tmp_path/'ancestor';ancestor.mkdir();old=tmp_path/'old';old.mkdir()
    monkeypatch.setattr(study.prior,'OLD',ancestor);monkeypatch.setattr(study,'OLD',old)
    entries={};audited={}
    for i in range(64):
        name=f'a_experienced_s{i:02d}';report=ancestor/(name+'.json');compressed=ancestor/(name+'.db.zst')
        study.atomic_json(report,dict(fixture=i));compressed.write_bytes(('compressed fixture '+name).encode())
        pin=dict(journal_hash='audited-semantic-'+name,report_sha256=study.sha(report))
        study.atomic_json(ancestor/(name+'-completion.json'),pin)
        audited[name]=dict(completion_sha256=study.sha(ancestor/(name+'-completion.json')),**pin)
        entries[name+'.db']=dict(source_sha256='archival-original-byte-hash-'+name,source_bytes=4096,
            archive_bytes=compressed.stat().st_size,archive_sha256=study.sha(compressed),fully_stream_verified_before_unlink=True)
    study.atomic_json(ancestor/'journal-archive-manifest.json',dict(root=str(ancestor),entries=entries))
    study.atomic_json(ancestor/'journal-archive-report.json',dict(passed=True,all_original_bytes_reconstruct_verified=True))
    study.atomic_json(old/'a-off-reuse-proof.json',dict(old_jobs=audited))
    files={};proof=study.archived_a_reference(files)
    assert len(proof['entries'])==64 and proof['decompressions']==0
    assert str(ancestor/'a_experienced_s00.db.zst') in files
    (ancestor/'a_experienced_s00.db.zst').write_bytes(b'tampered compressed bytes')
    with pytest.raises(ValueError,match='compressed journal'):study.archived_a_reference({})


def test_reused_control_journals_are_revalidated_on_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OLD',tmp_path)
    study.atomic_json(tmp_path/'manifest.json',dict(jobs=[dict(id='on_train'),dict(id='a_on_s00')]))
    proof=dict(training_pin={'rows':20000},a_pins={'a_on_s00':{'rows':100}})
    def payload(root,job):return proof['training_pin'] if job['id']=='on_train' else proof['a_pins'][job['id']]
    monkeypatch.setattr(study.worker,'pin_payload',payload)
    study.validate_reused_journals(proof)
    monkeypatch.setattr(study.worker,'pin_payload',lambda root,job:{'rows':0})
    with pytest.raises(ValueError,match='journal contents'):study.validate_reused_journals(proof)
