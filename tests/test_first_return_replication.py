"""Seed-only replication plumbing; no new policy/vision observations."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_first_return_curriculum as study

REPLICA=dict(study.DEFAULT_RECIPE,training_seed=2093651519,optimizer_seed=837263411,
    development_seed=1179033779,development_tag=29108,random_action_seed=668291057,
    protocol_file='docs/reconstruction/2026-10-21-first-return-replication-protocol.md')


def test_default_legacy_first_draw_exact_known_fixture():
    # Frozen before the registration parameterization, not a second import of
    # the updated implementation. Small generic calibration avoids artifacts.
    z=np.random.default_rng(8193).standard_normal((32,4))
    rng=np.random.default_rng(study.OPTIMIZER_SEED)
    theta=.1*study.prior.sample_directions(rng,z,1)[0]
    delta,candidates,serves,after=study.generation_spec(theta,rng.bit_generator.state,0,z,
        training_seed=study.TRAINING_SEED,serves_per_generation=12)
    assert study.array_hash(theta)=='ba31673449b50a49022c8acf0409d782e60dfd515cab39c1243f2acc91563774'
    assert study.array_hash(delta)=='ae1de785fa46409f95dae6221c960d9f4b567bea67c95316dcf1b9a14aa0a38f'
    assert study.array_hash(candidates)=='87062ff7c295b368318e3512fe8078662d10dd38692c4c09eaf8015222373c89'
    assert serves==[1915570759,766259534,1024648637,1927825492,78506446,1096690980,
                    254798974,1452478181,675457801,1068266130,1359011726,144482273]
    assert after==dict(bit_generator='PCG64',state=dict(state=339015255771708679801826546377330561998,
        inc=185211427201749367027179918158382254943),has_uint32=0,uinteger=0)


def test_registration_rejects_bad_seed_tag_protocol_and_lineage(tmp_path):
    assert study.registration(tmp_path)==(study.DEFAULT_RECIPE,study.PROTOCOL)
    for change in (dict(training_seed=True),dict(training_seed=study.engine.FINAL[0]),
                   dict(training_seed=REPLICA['optimizer_seed']),dict(development_tag=0),
                   dict(protocol_file='../../etc/passwd'),dict(replication_of={}),dict(extra=1)):
        study.atomic_json(tmp_path/'seed-recipe.json',dict(REPLICA,**change))
        with pytest.raises(ValueError):study.registration(tmp_path)


def verified_ancestor(directory,monkeypatch):
    directory.mkdir();audit=directory/'parent';audit.mkdir()
    monkeypatch.setattr(study.prior,'source_signature',lambda:{'source':'fixed'})
    monkeypatch.setattr(study.prior,'runtime_signature',lambda:{'runtime':'fixed'})
    study.atomic_json(directory/'manifest.json',dict(source=study.prior.source_signature(),
        runtime=study.prior.runtime_signature(),model_hash=study.prior.MODEL_HASH,
        vision_configuration=study.prior.VISION_CONFIGURATION,
        **{k:study.DEFAULT_RECIPE[k] for k in ('training_seed','optimizer_seed','development_seed','random_action_seed')}))
    study.atomic_json(directory/'summary.json',dict(status='complete',paired_evidence=dict(arms=dict(
        endpoint=dict(promising_development_signal=True)))))
    study.atomic_json(directory/'completion-verification.json',dict(passed=True))
    study.atomic_json(audit/'summary.json',dict(passed=True))
    study.atomic_json(directory/'launch.json',dict(source_commit='fixed-source'))
    return dict(directory=str(directory),source_commit='fixed-source',parent_audit_directory=str(audit),
        manifest_sha256=study.sha(directory/'manifest.json'),summary_sha256=study.sha(directory/'summary.json'),
        completion_verification_sha256=study.sha(directory/'completion-verification.json'),
        parent_audit_summary_sha256=study.sha(audit/'summary.json'))


def test_replica_requires_verified_same_condition_ancestor(tmp_path,monkeypatch):
    lineage=verified_ancestor(tmp_path/'ancestor',monkeypatch)
    recipe=dict(REPLICA,replication_of=lineage);study.atomic_json(tmp_path/'seed-recipe.json',recipe)
    assert study.registration(tmp_path)[0]==recipe
    changed=copy.deepcopy(recipe);changed['replication_of']['source_commit']='wrong'
    study.atomic_json(tmp_path/'seed-recipe.json',changed)
    with pytest.raises(ValueError,match='same-condition'):study.registration(tmp_path)
    study.atomic_json(tmp_path/'seed-recipe.json',recipe)
    (tmp_path/'ancestor/summary.json').write_text('{}')
    with pytest.raises(ValueError,match='lineage bytes changed'):study.registration(tmp_path)


def test_manifest_pins_recipe_protocol_and_carries_training_root(tmp_path,monkeypatch):
    # Small real matrix, durable metadata and real score mapping; no environment.
    lineage=tmp_path/'fixed';lineage.mkdir()
    np.save(lineage/'search-transform.npy',np.eye(4))
    (lineage/'transform-fit.json').write_text('{}');(lineage/'radial-cap.json').write_text('{}')
    for name in ('search-transform.npy','transform-fit.json','radial-cap.json'):
        (tmp_path/name).write_bytes((lineage/name).read_bytes())
    study.atomic_json(tmp_path/'projection.json',{})
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=[123]))
    study.atomic_json(tmp_path/'seed-recipe.json',REPLICA)
    monkeypatch.setattr(study.engine,'LINEAGE',lineage)
    monkeypatch.setattr(study.engine,'read_calibration',lambda path:np.eye(4))
    class Projection:
        def __init__(self,*args):pass
        def validate_embedded(self):pass
    monkeypatch.setattr(study,'FixedPolicyProjection',Projection)
    manifest,t,z,theta,rng=study.prepare(tmp_path)
    assert manifest['seed_recipe']==REPLICA and manifest['seed_recipe_hash']==study.sha(tmp_path/'seed-recipe.json')
    assert manifest['protocol_hash']==study.sha(study.ROOT/REPLICA['protocol_file'])
    assert manifest['development_tag']==29108
    assert study.prepare(tmp_path)[0]==manifest
    phase,state,current=study.initialize_phase(tmp_path,study.PHASES[0],theta,rng,0,manifest['started_wall_ns'])
    definition,delta=study.prepare_population(phase,t,z,state,current,0,.65)
    expected=study.generation_spec(theta,rng,0,z,training_seed=REPLICA['training_seed'],
        serves_per_generation=12,excluded_seeds=manifest['serve_seeds']+[123])
    np.testing.assert_array_equal(delta,expected[0]);assert definition['training_serve_seeds']==expected[2]
    jobs=study.evaluation_jobs(manifest);random=next(j for j in jobs if j['policy']=='random1')
    expected_seed=int(np.random.SeedSequence([668291057,1179033779,0]).generate_state(1)[0])
    assert random['config']['action_seed']==expected_seed
    study.atomic_json(tmp_path/'seed-recipe.json',dict(REPLICA,development_tag=29109))
    with pytest.raises(ValueError,match='registered curriculum changed'):study.prepare(tmp_path)


def test_prospective_own_initial_secondary_cannot_change_screen(monkeypatch):
    monkeypatch.setattr(study,'SERVES',4)
    def rally(success):return dict(own_returns=int(success),own_opportunities=1,
        first_incoming=dict(success=bool(success),tick=17,pre_ball=[.1,.4],pre_velocity=[-.4,.1]))
    data={p:[rally(p=='endpoint') for _ in range(4)] for p in study.POLICIES}
    old,secondary=study.evidence(data);new,extra=study.evidence(data,include_owninitial=True)
    assert old==new and set(extra)==set(secondary)|{'endpoint_minus_initial'}
    assert extra['endpoint_minus_initial']['first_incoming_lift']==1
