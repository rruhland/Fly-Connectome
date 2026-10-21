"""Feedback allocation/statistics checks without any camera observations."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pong_camera_feedback as study


def rally(success,returns=None,opportunities=1):
    return dict(own_returns=int(success) if returns is None else returns,own_opportunities=opportunities,
        first_incoming=dict(success=bool(success),tick=101,pre_ball=[.07,.3],pre_velocity=[-.5,.1]))


def test_exact_six_cohort_roster_and_sensor_only_difference():
    serves=list(range(1,33));jobs=study.declared_jobs(serves)
    assert len(jobs)==192 and study.CAMERA_CAP==192*500
    for policy in study.LINEAGES:
        groups={m:[j for j in jobs if j['policy']==policy+'_'+m] for m in study.MODES}
        for mode,group in groups.items():
            assert [j['config']['seed'] for j in group]==serves
            assert all(j['weights_from']==policy+'.npy' and j['config']['opponent_speed']==.65
                and j['config']['max_episode_ticks']==1200 and j['config']['action_hold_frames']==1
                and not j['config']['blank_camera'] and not j['config']['skip_baseline_perception']
                and j['purpose']=='fresh_evaluation' for j in group)
            if mode=='live':assert all('camera_intervention' not in j for j in group)
            else:assert all(j['camera_intervention']==dict(kind=mode,warmup_observations=20) for j in group)
        assert [j['config'] for j in groups['live']]==[j['config'] for j in groups['warm_missing']]


def test_seed_recipe_exclusions_and_root_collision(monkeypatch):
    monkeypatch.setattr(study,'DEV_SEED',12345)
    first=study.serve_schedule([])
    revised=study.serve_schedule(first[:5])
    assert revised[:27]==first[5:] and len(set(revised))==32
    assert not set(revised)&set((*first[:5],*study.prior.engine.FINAL))
    with pytest.raises(ValueError,match='root already used'):study.serve_schedule([12345])


def test_joint_claim_geometry_partial_and_fixed_policy_average(monkeypatch):
    monkeypatch.setattr(study,'SERVES',4)
    data={p+'_'+m:[rally(m=='live') for _ in range(4)] for p in study.LINEAGES for m in study.MODES}
    result=study.evidence(data)
    assert result['joint_sensitivity_finding'] and result['serve_clusters']==4 and result['policy_count']==2
    for contrast in result['contrasts'].values():
        a=contrast['equal_policy_average']
        assert a['first_incoming_lift']==a['all_opportunity_lift']==1
        assert a['first_incoming_lift_ci95']==[1.,1.]
    # Equal policy weight, never pool policy-dependent contact denominators.
    data['seed1_live']=[rally(True,returns=1,opportunities=1) for _ in range(4)]
    data['seed2_live']=[rally(True,returns=1,opportunities=4) for _ in range(4)]
    result=study.evidence(data)
    assert result['contrasts']['live_minus_warm_hold']['equal_policy_average']['all_opportunity_lift']==.625
    changed=copy.deepcopy(data);changed['seed2_live']=[rally(False,returns=0,opportunities=4) for _ in range(4)]
    assert not study.evidence(changed)['joint_sensitivity_finding']
    changed['seed2_warm_hold'][0]['first_incoming']['tick']=102
    with pytest.raises(ValueError,match='trajectory differs'):study.evidence(changed)
    data['seed1_warm_missing'].pop();assert not study.evidence(data)['complete']


def test_source_bridge_rejects_any_other_core_change():
    old={'control/pong_evaluation.py':'old','vision/state.py':'unchanged'}
    new=dict(old,**{'control/pong_evaluation.py':'new'})
    assert study.source_bridge(old,new)==['control/pong_evaluation.py']
    new['vision/state.py']='different'
    with pytest.raises(ValueError,match='only the explicitly tested'):study.source_bridge(old,new)


def test_manifest_resume_keeps_original_clock_and_rejects_artifact_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'DEV_SEED',12345);monkeypatch.setattr(study,'SERVES',2)
    files=('seed1.npy','seed1.json','seed2.npy','seed2.json','radial-cap.json','projection.json',
           'search-transform.npy','transform-fit.json')
    for name in files:(tmp_path/name).write_bytes(b'fixed synthetic input')
    study.atomic_json(tmp_path/'lineage-proof.json',dict(current_source=study.prior.source_signature(),
        runtime=study.prior.runtime_signature(),mapped_files={n:study.sha(tmp_path/n) for n in files}))
    study.atomic_json(tmp_path/'seed-exclusions.json',dict(ids=list(study.prior.engine.FINAL)))
    study.atomic_json(tmp_path/'default-bridge.json',dict(passed=True,current_source=study.prior.source_signature()))
    study.atomic_json(tmp_path/'focused-checks.json',dict(passed=True))
    manifest=study.prepare(tmp_path);before=(tmp_path/'manifest.json').read_bytes()
    assert study.prepare(tmp_path)==manifest and (tmp_path/'manifest.json').read_bytes()==before
    assert len(manifest['jobs'])==12 and manifest['new_training_observations']==0
    (tmp_path/'seed2.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen mapped files changed'):study.prepare(tmp_path)


def test_real_worker_handoff_accepts_live_and_modes_before_construction(tmp_path,monkeypatch):
    worker=study.prior.engine.base
    study.prior.weights_file(tmp_path,'seed1.npy',np.ones((3,4389))*.001)
    (tmp_path/'radial-cap.json').write_text('{}')
    class ReachedConstructor(Exception):pass
    for job in study.declared_jobs([111])[:3]:
        study.atomic_json(tmp_path/'manifest.json',dict(source=worker.source_signature(),runtime=worker.runtime_signature(),
            driver_hash=study.sha(worker.__file__),jobs=[job]))
        def intercept(weights,config,**kwargs):
            assert config.seed==111 and weights.shape==(3,4389)
            assert kwargs==({} if job['camera_mode']=='live' else dict(camera_intervention=job['camera_intervention']))
            raise ReachedConstructor
        monkeypatch.setattr(worker,'FreshPongEvaluator',intercept)
        with pytest.raises(ReachedConstructor):worker.worker(tmp_path,job['id'],emit=False)
    assert not list(tmp_path.glob('*.db'))
