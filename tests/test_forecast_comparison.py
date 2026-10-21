"""Generic cached algebra/metadata tests: no environment or policy observations."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_forecast_comparison as study
from fly_connectome.vision import VisualStateEncoder


def test_basis_mask_before_fit_retains_bias_and_exact_zero_forecasts():
    encoder=VisualStateEncoder(64,64,.02);rng=np.random.default_rng(19827)
    raw=rng.normal(size=(96,4389));raw[:,4384]=1.
    transform,z=study.fit_state_only(raw,encoder.layout,encoder.dimension)
    matrix=transform.matrix
    assert matrix.shape==(214,4389) and np.count_nonzero(matrix[:,1792:3848])==0
    assert transform.diagnostics['basis_dimension']==214
    assert transform.diagnostics['projected_bias_energy']==1.
    assert transform.diagnostics['free_action_contrasts']==2*transform.diagnostics['supported_rank']
    assert np.array_equal(matrix@np.zeros(4389),np.zeros(214))
    theta=rng.normal(size=(3,214));weights=transform.mapped_weights(theta)
    assert np.count_nonzero(weights[:,1792:3848])==0
    projected=z@theta.T;mapped=raw@weights.T
    np.testing.assert_allclose(projected,mapped,atol=1e-12,rtol=1e-12)
    assert np.array_equal(projected.argmax(1),mapped.argmax(1))
    capped=raw*np.minimum(1.,np.sqrt(256/np.sum(raw*raw,axis=1)))[:,None]
    assert np.array_equal((capped@weights.T).argmax(1),mapped.argmax(1))
    altered=raw.copy();altered[:,1792:3848]=rng.normal(100,1000,size=(96,2056))
    assert np.array_equal(altered@matrix.T,z)


def test_same_214_storage_keeps_raw_rng_draw_layout():
    rng1=np.random.default_rng(741);rng2=np.random.default_rng(741)
    z1=np.random.default_rng(231).normal(size=(50,214));z2=z1.copy();z2[:,75:150]=0.
    study.full.prior.sample_directions(rng1,z1,8);study.full.prior.sample_directions(rng2,z2,8)
    assert rng1.bit_generator.state==rng2.bit_generator.state


def rally(success,returns=None,opportunities=1):
    return dict(own_returns=int(success) if returns is None else returns,own_opportunities=opportunities,
        first_incoming=dict(success=bool(success),tick=101,pre_ball=[.1,.4],pre_velocity=[-.5,.1]))


def test_comparison_is_paired_not_new_acquisition_screen():
    positive=[rally(True) for _ in range(64)];negative=[rally(False) for _ in range(64)]
    result=study.comparison(positive,negative)
    assert result['positive_development_signal'] and not result['reverse_development_signal']
    assert result['first_incoming_lift']==result['all_opportunity_lift']==1.
    assert result['first_incoming_lift_ci95']==[1.,1.] and result['bootstrap_seed']==194870295
    assert study.comparison(negative,positive)['reverse_development_signal']
    assert not study.comparison(positive,positive)['positive_development_signal']
    assert not study.comparison(positive[:-1],negative)['complete']
    changed=copy.deepcopy(negative);changed[0]['first_incoming']['tick']=102
    with pytest.raises(ValueError,match='geometry differs'):study.comparison(positive,changed)


def test_state_eval_roster_executes_own_four_and_references_five_simples():
    manifest=dict(serve_seeds=list(range(1,65)),random_action_seed=719456287,development_seed=1662073981)
    all_jobs=study.full.evaluation_jobs(manifest)
    executed=[j for j in all_jobs if j['policy'] in study.EXECUTED]
    reused=[j for j in all_jobs if j['policy'] in study.SIMPLES]
    assert len(executed)==256 and len(reused)==320
    assert set(study.EXECUTED).isdisjoint(study.SIMPLES)
    assert all(j['weights_from'] is not None and not j['config']['skip_baseline_perception'] for j in executed)
    assert all(j['weights_from'] is None and j['config']['skip_baseline_perception'] for j in reused)
    assert all(j['config']['opponent_speed']==.65 and j['config']['max_episode_ticks']==1200 for j in all_jobs)


def test_full_coordinator_math_worker_and_core_match_qualified_source():
    ancestor=study.ANCESTOR
    if not (ancestor/'manifest.json').exists():pytest.skip('Qualified external lineage unavailable; ordinary algebra tests still run')
    old=json.loads((ancestor/'manifest.json').read_text())
    assert study.full.prior.source_signature()==old['source']
    # Original seed-parameterized coordinator is replica886d05d; its math/worker
    # are identical to both independently verified training runs.
    replica=ancestor.with_name('pong-first-return-replication-20261021-r1')
    if not (replica/'manifest.json').exists():pytest.skip('Replica provenance fixture unavailable')
    original=json.loads((replica/'manifest.json').read_text())
    assert study.sha(study.full.__file__)==original['coordinator_hash']
    assert study.sha(study.full.engine.MATH)==original['math_hash']
    assert study.sha(study.full.engine.WORKER)==original['driver_hash']


def test_reuse_rejects_incomplete_full_before_any_state_clock(monkeypatch,tmp_path):
    d=tmp_path/'full';d.mkdir()
    (d/'manifest.json').write_text('{}');(d/'summary.json').write_text(json.dumps(dict(status='partial')))
    with pytest.raises(ValueError,match='full completion required'):study.reuse_controls(tmp_path)
    assert not (tmp_path/'state-only'/'manifest.json').exists()


def synthetic_joint(tmp_path,monkeypatch):
    roots=dict(training_seed=101,optimizer_seed=202,development_seed=303,development_tag=404,random_action_seed=505)
    monkeypatch.setattr(study,'ROOTS',roots)
    monkeypatch.setattr(study,'recipe',lambda replicate:dict(study.full.DEFAULT_RECIPE,**roots,replication_of=None))
    cache=tmp_path/'raw-cache';cache.write_bytes(b'generic-unlabeled-fixture')
    monkeypatch.setattr(study.full.engine,'CACHE',cache)
    raw=np.zeros((8,4389));raw[:,4384]=1.;raw[:,0]=np.arange(8)
    monkeypatch.setattr(study.full.engine,'read_calibration',lambda path:raw)
    def snapshot(d):
        if not (d/'seed-exclusions.json').exists():study.atomic_json(d/'seed-exclusions.json',dict(ids=[719]))
    monkeypatch.setattr(study.full.seeds,'snapshot_exclusions',snapshot)
    def stage_full(d):
        matrix=np.zeros((214,4389));matrix[:214,:214]=np.eye(214)
        study.full.immutable_array(d/'search-transform.npy',matrix)
        study.atomic_json(d/'transform-fit.json',{})
        study.atomic_json(d/'radial-cap.json',{})
        study.atomic_json(d/'projection.json',dict(version='fixed-linear-policy-projection-v1:cap-after',
            shape=[214,4389],matrix_file=str(d/'search-transform.npy'),file_sha256=study.sha(d/'search-transform.npy'),
            array_sha256=study.full.array_hash(matrix),max_norm2=256.,calibration=dict(fit_file='fixture',fit_sha256='0'*64)))
    monkeypatch.setattr(study.full,'stage',stage_full)
    return tmp_path/'joint'


def test_joint_staging_resume_state_clock_and_phase_carry(tmp_path,monkeypatch):
    root=synthetic_joint(tmp_path,monkeypatch);study.stage(root)
    before=(root/'comparison-registration.json').read_bytes()
    assert not (root/'full'/'manifest.json').exists() and not (root/'state-only'/'manifest.json').exists()
    study.stage(root);assert (root/'comparison-registration.json').read_bytes()==before
    assert (root/'full'/'seed-exclusions.json').read_bytes()==(root/'state-only'/'seed-exclusions.json').read_bytes()
    manifest,t,z,theta,rng=study.prepare_state(root)
    clock=manifest['started_wall_ns'];raw=(root/'state-only'/'manifest.json').read_bytes()
    assert study.prepare_state(root)[0]==manifest and (root/'state-only'/'manifest.json').read_bytes()==raw
    assert manifest['evaluation_camera_cap']==128000 and manifest['reused_control_jobs']==320
    assert manifest['serve_seeds']==study.validate_joint(root)['serve_seeds']
    directory,state,current=study.full.initialize_phase(root/'state-only',study.full.PHASES[1],theta,rng,7,clock)
    assert state['rng_state']==rng and np.array_equal(current,theta)
    assert np.count_nonzero(np.load(directory/'policy_at0.npy')[:,1792:3848])==0
    with pytest.raises(ValueError,match='phase lineage/clock'):
        study.full.initialize_phase(root/'state-only',study.full.PHASES[1],theta,rng,7,clock+1)
    changed=json.loads((root/'state-only'/'seed-exclusions.json').read_text());changed['ids'].append(809)
    study.atomic_json(root/'state-only'/'seed-exclusions.json',changed)
    with pytest.raises(ValueError,match='exclusion snapshot changed'):study.prepare_state(root)


def test_joint_rejects_changed_fit_and_root_redeclaration(tmp_path,monkeypatch):
    root=synthetic_joint(tmp_path,monkeypatch);study.stage(root)
    (root/'state-only'/'transform-fit.json').write_text('{}')
    with pytest.raises(ValueError,match='joint source/inputs'):study.validate_joint(root)
    with pytest.raises(ValueError,match='fit changed'):study.stage(root)
    study.atomic_json(root/'seed-exclusions.json',dict(ids=[study.ROOTS['training_seed']]))
    with pytest.raises(ValueError,match='fresh joint root'):study.stage(root)


def test_reused_control_evidence_is_pinned_and_tamper_rejected(tmp_path,monkeypatch):
    d=tmp_path/'full';d.mkdir();audit=tmp_path/'audit';audit.mkdir()
    monkeypatch.setattr(study,'FULL_AUDIT',audit)
    manifest=dict(source=study.full.prior.source_signature(),serve_seeds=[11,12],
                  development_seed=91,random_action_seed=193)
    manifest['jobs']=study.full.evaluation_jobs(manifest)
    study.atomic_json(d/'manifest.json',manifest);study.atomic_json(d/'summary.json',dict(status='complete'))
    annotations={}
    for job in manifest['jobs']:
        annotations[job['id']]=dict(rallies=[rally(False)])
        study.atomic_json(d/(job['id']+'.json'),dict(complete=True,transitions=5))
        study.atomic_json(d/(job['id']+'-completion.json'),dict(fixture=True))
    study.atomic_json(d/'return-annotations.json',annotations)
    study.atomic_json(d/'completion-verification.json',dict(passed=True,evaluation_evidence_hashes={
        n:study.sha(d/n) for n in ('manifest.json','summary.json','return-annotations.json')}))
    study.atomic_json(audit/'summary.json',dict(passed=True,evidence_file_hashes={
        n:study.sha(d/n) for n in ('manifest.json','summary.json','return-annotations.json')}))
    monkeypatch.setattr(study,'validate_joint',lambda root:dict(serve_seeds=[11,12]))
    monkeypatch.setattr(study.full.completion,'complete_report',lambda directory,job:dict(complete=True,transitions=5))
    reused,proof=study.reuse_controls(tmp_path)
    assert len(proof)==10 and all(len(v)==2 for v in reused.values())
    assert all(p['annotations_hash']==study.sha(d/'return-annotations.json') for p in proof)
    annotations[next(iter(annotations))]['rallies'][0]['own_returns']=99
    study.atomic_json(d/'return-annotations.json',annotations)
    with pytest.raises(ValueError,match='audited shared evaluation evidence changed'):study.reuse_controls(tmp_path)
