"""Synthetic configuration/resume tests; no new camera trajectories."""
import copy,json,sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_frozen_hit_confirmation as study


def test_all_eight_cohorts_share_serves_and_only_blank_masks_inputs():
    serves=list(range(1,65));jobs=study.declared_jobs(serves)
    assert len(jobs)==512
    for policy in study.POLICIES:
        group=[j for j in jobs if j['policy']==policy]
        assert [j['config']['seed'] for j in group]==serves
        assert all(j['config']['blank_camera']==(policy=='blank') for j in group)
        assert all(j['config']['opponent_speed']==.65 and j['config']['max_episode_ticks']==1200 for j in group)
        assert all(j['config']['action_hold_frames']==(10 if policy=='random10' else 1) for j in group)
    for index in range(64):
        pair=[j for j in jobs if j['serve_index']==index and j['policy'] in ('random1','random10')]
        assert pair[0]['config']['action_seed']==pair[1]['config']['action_seed']
    assert all(j['weights_from']=='endpoint.npy' and not j['config']['skip_baseline_perception']
               for j in jobs if j['policy'] in ('endpoint','blank'))
    assert all(j['weights_from']=='initial-restricted.npy' for j in jobs if j['policy']=='initial_restricted')


def test_metadata_seed_snapshot_is_frozen_and_does_not_start_clock(tmp_path):
    experiments=tmp_path/'old';experiments.mkdir();directory=tmp_path/'new';directory.mkdir()
    metadata=experiments/'manifest.json'
    metadata.write_text(json.dumps(dict(seed=17,jobs=[dict(config=dict(seed=31,action_seed=43))],reward=999)))
    (experiments/'g00-definition.json').write_text(json.dumps(dict(training_serve_seeds=[59,61])))
    study.snapshot_exclusions(directory,experiments)
    path=directory/'seed-exclusions.json';before=path.read_bytes();a=json.loads(before)
    assert {17,31,43,59,61,849375169,686753009,*study.prior.engine.FINAL}<=set(a['ids'])
    assert 999 not in a['ids'] and a['metadata_sources'][str(metadata)]==study.sha(metadata)
    metadata.write_text(json.dumps(dict(seed=101)));study.snapshot_exclusions(directory,experiments)
    assert path.read_bytes()==before and not (directory/'manifest.json').exists()


def test_serve_recipe_rejects_exclusions_without_expanding_final_streams(monkeypatch):
    monkeypatch.setattr(study,'DEV_SEED',12345)
    first=study.serve_schedule([]);excluded=first[:7]
    revised=study.serve_schedule(excluded)
    assert len(set(revised))==64 and not set(revised)&set(excluded)
    assert not set(revised)&set(study.prior.engine.FINAL)
    assert revised[:57]==first[7:] and revised==study.serve_schedule(excluded)


def rally(success):
    return dict(own_returns=int(success),own_opportunities=1,
        first_incoming=dict(success=bool(success),tick=17,pre_ball=[.1,.4],pre_velocity=[-.4,.1]))


def test_blank_is_secondary_only_and_geometry_is_verified(monkeypatch):
    monkeypatch.setattr(study,'SERVES',4)
    data={p:[rally(p in ('endpoint','blank')) for _ in range(4)] for p in study.POLICIES}
    primary,secondary=study.evidence(data)
    assert primary['arms']['endpoint']['promising_development_signal']
    assert secondary['first_incoming_lift']==0 and secondary['all_opportunity_lift']==0
    changed=copy.deepcopy(data);changed['blank']=[rally(False) for _ in range(4)]
    again,difference=study.evidence(changed)
    assert primary==again and difference['first_incoming_lift']==difference['all_opportunity_lift']==1
    changed['blank'][0]['first_incoming']['tick']=18
    with pytest.raises(ValueError,match='trajectory differs'):study.evidence(changed)


def test_partial_secondary_remains_partial_without_changing_primary(monkeypatch):
    monkeypatch.setattr(study,'SERVES',4)
    data={p:[rally(p=='endpoint') for _ in range(4)] for p in study.POLICIES}
    data['blank'].pop();primary,secondary=study.evidence(data)
    assert primary['complete'] and not secondary['complete']


def test_manifest_resume_keeps_clock_and_rejects_frozen_file_tampering(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'SERVES',2);monkeypatch.setattr(study,'DEV_SEED',12345)
    files=('endpoint.npy','initial-restricted.npy','radial-cap.json','projection.json',
           'search-transform.npy','transform-fit.json')
    for name in files:(tmp_path/name).write_bytes(b'fixed-fixture')
    study.atomic_json(tmp_path/'lineage-proof.json',dict(source=study.prior.source_signature(),
        runtime=study.prior.runtime_signature(),model_hash=study.prior.MODEL_HASH,
        vision_configuration=study.prior.VISION_CONFIGURATION,mapped_files={n:study.sha(tmp_path/n) for n in files}))
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=list(study.prior.engine.FINAL)))
    manifest=study.prepare(tmp_path);before=(tmp_path/'manifest.json').read_bytes()
    assert study.prepare(tmp_path)==manifest and (tmp_path/'manifest.json').read_bytes()==before
    assert manifest['primary_controls']==list(study.CONTROLS) and manifest['secondary_only']==['blank']
    assert manifest['new_training_observations']==0 and len(manifest['jobs'])==16
    (tmp_path/'endpoint.npy').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='frozen readout'):study.prepare(tmp_path)
