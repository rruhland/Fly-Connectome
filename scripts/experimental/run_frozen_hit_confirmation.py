"""Fresh development confirmation of the frozen theta9 hit-return readout."""
import argparse,fcntl,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_direct_hit_policy_search as prior
from return_metrics import aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-frozen-hit-confirmation-protocol.md'
LINEAGE=Path('/workspace/m1b-experiments/pong-direct-hit-policy-search-20261021-r1')
EXPERIMENTS=LINEAGE.parent
SERVES=64;DEV_SEED=2037041129;ACTION_SEED=1063729411
EVALUATION_SECONDS=900.;TOTAL_SECONDS=1080.;CAMERA_CAP=256000
CONTROLS=prior.CONTROLS
POLICIES=('endpoint',*CONTROLS,'blank')
sha=prior.sha;atomic_json=prior.atomic_json


def collect_seed_ids(value):
    """Metadata only; never inspect reward, label or coefficient data."""
    found=set()
    def integers(v):
        if type(v) is int and 0<v<2**32:found.add(v)
        elif isinstance(v,list):
            for item in v:integers(item)
        elif isinstance(v,dict):
            for item in v.values():integers(item)
    def visit(v):
        if isinstance(v,dict):
            for key,item in v.items():
                if 'seed' in key.lower():integers(item)
                else:visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
    visit(value)
    return found


def snapshot_exclusions(directory,experiments=EXPERIMENTS):
    path=directory/'seed-exclusions.json'
    if path.exists():return
    sources={};ids=set(prior.engine.FINAL)|{849375169,686753009,904271833}
    paths=set()
    for pattern in ('manifest.json','*-definition.json','protocol.json','*calibration*.json'):
        paths.update(experiments.rglob(pattern))
    for source in sorted(paths):
        if source.is_relative_to(directory):continue
        value=json.loads(source.read_text());extracted=collect_seed_ids(value)
        if extracted:ids.update(extracted);sources[str(source)]=sha(source)
    atomic_json(path,dict(version='prior-declared-seed-metadata-snapshot-v1',ids=sorted(ids),
        metadata_sources=sources,reserved_final_ids=list(prior.engine.FINAL),
        final_streams_expanded=False,known_pong_calibration_ids=[849375169,686753009]))


def stage(directory):
    """Copy immutable readouts and metadata without starting any allocation clock."""
    directory.mkdir(parents=True,exist_ok=True)
    old=json.loads((LINEAGE/'manifest.json').read_text())
    summary=json.loads((LINEAGE/'summary.json').read_text())
    if (summary['status']!='complete' or summary['training_endpoint']['completed_generations']!=9
            or not summary['training_endpoint']['valid'] or old['source']!=prior.source_signature()
            or old['runtime']!=prior.runtime_signature()):
        raise ValueError('exact completed theta9 lineage/runtime required')
    matrix=np.load(LINEAGE/'search-transform.npy',allow_pickle=False)
    for count in (0,9):
        if not np.array_equal(np.load(LINEAGE/f'policy_at{count}.npy'),
                              np.load(LINEAGE/f'theta_at{count}.npy')@matrix):
            raise ValueError('frozen mapped readout differs from its theta')
    copies={'endpoint.npy':'policy_at9.npy','endpoint.json':'policy_at9.json',
        'initial-restricted.npy':'policy_at0.npy','initial-restricted.json':'policy_at0.json',
        **{name:name for name in ('radial-cap.json','projection.json','search-transform.npy','transform-fit.json')}}
    for target,source in copies.items():
        if (directory/target).exists():
            if sha(directory/target)!=sha(LINEAGE/source):raise ValueError('staged frozen bytes differ')
        else:shutil.copyfile(LINEAGE/source,directory/target)
    proof=dict(lineage=str(LINEAGE),source=old['source'],runtime=old['runtime'],
        model_hash=old['model_hash'],vision_configuration=old['vision_configuration'],
        manifest_hash=sha(LINEAGE/'manifest.json'),
        summary_hash=sha(LINEAGE/'summary.json'),source_archive_hash=sha(LINEAGE/'source.tar.gz'),
        selected_checkpoint='theta9; original theta0 retained',
        mapped_files={target:sha(directory/target) for target in copies},
        original_screen_passed=False,no_new_training=True)
    path=directory/'lineage-proof.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('frozen lineage proof changed')
    atomic_json(path,proof);snapshot_exclusions(directory)


def serve_schedule(excluded):
    rng=np.random.default_rng(np.random.SeedSequence([DEV_SEED,29105]));result=[]
    excluded=set(excluded)|set(prior.engine.FINAL)
    while len(result)<SERVES:
        value=int(rng.integers(1,2147483647))
        if value not in excluded and value not in result:result.append(value)
    return result


def declared_jobs(serves):
    jobs=[]
    for policy in POLICIES:
        for i,seed in enumerate(serves):
            trained=policy in ('endpoint','initial_restricted','blank')
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            action_seed=int(np.random.SeedSequence([ACTION_SEED,DEV_SEED,i]).generate_state(1)[0]) if kind=='random' else None
            if action_seed in prior.engine.FINAL:raise ValueError('reserved action seed collision')
            weights=('initial-restricted.npy' if policy=='initial_restricted' else 'endpoint.npy') if trained else None
            job=prior.rollout_job(f'eval_{policy}_s{i:02d}',seed,weights,'fresh_evaluation',
                kind=kind,action_seed=action_seed,hold=10 if policy=='random10' else 1)
            job['config']['blank_camera']=policy=='blank'
            job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def prepare(directory):
    # stage() is explicitly separate so review/tests cannot consume the real clock.
    required=('seed-exclusions.json','lineage-proof.json','endpoint.npy','endpoint.json',
              'initial-restricted.npy','initial-restricted.json',
              'radial-cap.json','projection.json','search-transform.npy','transform-fit.json')
    if any(not (directory/name).exists() for name in required):raise ValueError('stage frozen inputs before launch')
    proof=json.loads((directory/'lineage-proof.json').read_text())
    if (proof['source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature()
            or proof['model_hash']!=prior.MODEL_HASH or proof['vision_configuration']!=prior.VISION_CONFIGURATION):
        raise ValueError('frozen lineage source/runtime/model differs')
    if any(sha(directory/name)!=digest for name,digest in proof['mapped_files'].items()):
        raise ValueError('staged frozen readout/transform bytes changed')
    excluded=json.loads((directory/'seed-exclusions.json').read_text())['ids'];serves=serve_schedule(excluded)
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,coordinator_hash=sha(__file__),protocol_hash=sha(PROTOCOL),
        driver_hash=sha(prior.engine.WORKER),prior_helper_hash=sha(prior.__file__),
        engine_hash=sha(prior.engine.__file__),completion_helper_hash=sha(prior.completion.__file__),
        physical_helper_hash=sha(prior.physical.__file__),metrics_hash=sha(Path(prior.physical.__file__).with_name('return_metrics.py')),
        lineage_proof_hash=sha(directory/'lineage-proof.json'),seed_exclusions_hash=sha(directory/'seed-exclusions.json'),
        frozen_file_hashes={name:sha(directory/name) for name in required},serve_seeds=serves,
        development_seed=DEV_SEED,random_action_seed=ACTION_SEED,jobs=declared_jobs(serves),
        primary_controls=list(CONTROLS),secondary_only=['blank'],new_training_observations=0,
        evaluation_wall_seconds=EVALUATION_SECONDS,total_wall_seconds=TOTAL_SECONDS,camera_cap=CAMERA_CAP,
        workers=4,gc_worker_mode=prior.GC_WORKER_MODE,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(prior.engine.FINAL),purpose='fresh DEV confirmation; original screen remains failed')
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered frozen confirmation changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns());atomic_json(path,manifest)
    return manifest


def evidence(by_policy):
    primary=paired_evidence(by_policy,arms=('endpoint',),expected_serves=SERVES,controls=CONTROLS)
    all_evidence=paired_evidence(by_policy,arms=('endpoint','blank'),expected_serves=SERVES,
        controls=CONTROLS,comparison=('endpoint','blank'))
    secondary=all_evidence.get('paired_arm_comparison',dict(complete=False))
    secondary.update(scope='Secondary whole visual-input blanking; OOD; not object/forecast-specific or primary qualification')
    return primary,secondary


def supervise(directory):
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=prepare(directory);start=manifest['started_wall_ns']/1e9;jobs=manifest['jobs']
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
            evaluated=prior.engine.execute_jobs(directory,jobs,pool,deadline=start+EVALUATION_SECONDS,
                camera_cap=CAMERA_CAP,initial_consumed=prior.engine.counts(directory,manifest)['fresh_evaluation'],
                completion_report=prior.completion.complete_report,completion_pin=prior.completion.pin_completion)
        annotations=prior.physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        by_policy={};groups={}
        for policy in POLICIES:
            rallies=[r for j in jobs if j['policy']==policy and j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        primary,secondary=evidence(by_policy);counts=prior.engine.counts(directory,manifest)
        assert counts['training_fitness']==0 and counts['fresh_evaluation']<=CAMERA_CAP
        elapsed=time.time()-start;overall_budget_met=elapsed<=TOTAL_SECONDS
        summary=dict(status='complete' if evaluated['complete'] and overall_budget_met else 'partial',
            overall_budget_met=overall_budget_met,counts=counts,evaluation=evaluated,
            groups=groups,paired_evidence=primary,secondary_blank=secondary,selection='unchanged frozen theta9; no training/selection',
            original_screen_remains_failed=True,wall_seconds=elapsed,final_holdout_unused=list(prior.engine.FINAL),
            scope='New64-serve DEV confirmation only; no independent training replication or completed M1B')
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:supervise(args.directory)
