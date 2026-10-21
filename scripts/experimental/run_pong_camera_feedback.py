"""Paired later-camera interventions on two independently acquired fixed policies."""
import argparse,fcntl,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_direct_hit_policy_search as prior
from run_frozen_hit_confirmation import snapshot_exclusions
from return_metrics import aggregate

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-camera-feedback-protocol.md'
EXPERIMENTS=Path('/workspace/m1b-experiments')
LINEAGES={
    'seed1':EXPERIMENTS/'pong-first-return-curriculum-20261021-r1',
    'seed2':EXPERIMENTS/'pong-first-return-replication-20261021-r1'}
AUDITS={
    'seed1':EXPERIMENTS/'first-return-curriculum-parent-audit-20261021'/'summary.json',
    'seed2':EXPERIMENTS/'first-return-replication-parent-audit-20261021'/'summary.json'}
MODES=('live','warm_hold','warm_missing')
SERVES=32;DEV_SEED=1438291051;DEV_TAG=29109;BOOTSTRAP_SEED=194870293
EVALUATION_SECONDS=720.;TOTAL_SECONDS=900.;CAMERA_CAP=96000
sha=prior.sha;atomic_json=prior.atomic_json


def source_bridge(old,new):
    changed=sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k))
    if changed!=['control/pong_evaluation.py']:
        raise ValueError('only the explicitly tested optional evaluator intervention may differ')
    return changed


def copy_exact(source,target):
    if target.exists():
        if sha(target)!=sha(source):raise ValueError('staged immutable file changed')
    else:shutil.copyfile(source,target)


def stage(directory):
    """Copy only verified final endpoints; stage never starts the allocation clock."""
    directory.mkdir(parents=True,exist_ok=True);proofs={}
    for name,lineage in LINEAGES.items():
        manifest=json.loads((lineage/'manifest.json').read_text())
        summary=json.loads((lineage/'summary.json').read_text())
        verification=json.loads((lineage/'completion-verification.json').read_text())
        audit=json.loads(AUDITS[name].read_text())
        if (len(summary['phases'])!=2 or not all(p['valid'] and not p['failed']
                and p['completed_generations']>=2 for p in summary['phases'])
                or not verification['passed'] or not audit.get('passed',audit.get('ok',False))
                or manifest['runtime']!=prior.runtime_signature()
                or manifest['model_hash']!=prior.MODEL_HASH
                or manifest['vision_configuration']!=prior.VISION_CONFIGURATION):
            raise ValueError('both integrity-verified two-phase endpoints required')
        changed=source_bridge(manifest['source'],prior.source_signature())
        matrix=np.load(lineage/'search-transform.npy',allow_pickle=False)
        phase=summary['phases'][1]
        theta=np.load(lineage/phase['phase']/phase['theta_file'],allow_pickle=False)
        weights=np.load(lineage/'policy_endpoint.npy',allow_pickle=False)
        if not np.array_equal(theta@matrix,weights):raise ValueError('endpoint is not the last complete phase1 mapping')
        for suffix in ('.npy','.json'):
            copy_exact(lineage/('policy_endpoint'+suffix),directory/(name+suffix))
        copy_exact(lineage/'projection.json',directory/(name+'-projection.json'))
        if name==next(iter(LINEAGES)):
            copy_exact(lineage/'projection.json',directory/'projection.json')
        for filename in ('radial-cap.json','search-transform.npy','transform-fit.json'):
            copy_exact(lineage/filename,directory/filename)
        proofs[name]=dict(lineage=str(lineage),manifest_hash=sha(lineage/'manifest.json'),
            summary_hash=sha(lineage/'summary.json'),verification_hash=sha(lineage/'completion-verification.json'),
            parent_audit=str(AUDITS[name]),parent_audit_hash=sha(AUDITS[name]),
            source_archive_hash=sha(lineage/'source.tar.gz'),launch_hash=sha(lineage/'launch.json'),
            projection_descriptor_hash=sha(lineage/'projection.json'),
            source=manifest['source'],source_changed=changed,theta_hash=phase['theta_hash'],
            theta_file=str(lineage/phase['phase']/phase['theta_file']),
            selected='last complete phase1 endpoint; both seeds retained regardless acquisition screen')
    files=('seed1.npy','seed1.json','seed2.npy','seed2.json','seed1-projection.json','seed2-projection.json','radial-cap.json',
           'projection.json','search-transform.npy','transform-fit.json')
    proof=dict(policies=proofs,current_source=prior.source_signature(),runtime=prior.runtime_signature(),
        model_hash=prior.MODEL_HASH,vision_configuration=prior.VISION_CONFIGURATION,
        mapped_files={n:sha(directory/n) for n in files},new_training_observations=0)
    path=directory/'lineage-proof.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('frozen lineage proof changed')
    atomic_json(path,proof);snapshot_exclusions(directory)


def serve_schedule(excluded):
    excluded=set(excluded)|set(prior.engine.FINAL);result=[]
    if DEV_SEED in excluded:raise ValueError('fresh development root already used')
    rng=np.random.default_rng(np.random.SeedSequence([DEV_SEED,DEV_TAG]))
    while len(result)<SERVES:
        value=int(rng.integers(1,2147483647))
        if value not in excluded and value not in result:result.append(value)
    return result


def declared_jobs(serves):
    jobs=[]
    for policy in LINEAGES:
        for mode in MODES:
            for i,seed in enumerate(serves):
                job=prior.rollout_job(f'eval_{policy}_{mode}_s{i:02d}',seed,policy+'.npy','fresh_evaluation')
                if mode!='live':job['camera_intervention']=dict(kind=mode,warmup_observations=20)
                job.update(policy=policy+'_'+mode,trained_seed=policy,camera_mode=mode,serve_index=i)
                jobs.append(job)
    return jobs


def prepare(directory):
    required=('lineage-proof.json','seed-exclusions.json','default-bridge.json','focused-checks.json')
    if any(not (directory/n).exists() for n in required):raise ValueError('stage and validate frozen inputs before launch')
    proof=json.loads((directory/'lineage-proof.json').read_text())
    bridge=json.loads((directory/'default-bridge.json').read_text())
    checks=json.loads((directory/'focused-checks.json').read_text())
    if (proof['current_source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature()
            or not bridge['passed'] or bridge['current_source']!=prior.source_signature()
            or not checks['passed']):raise ValueError('frozen source/default bridge/checks differ')
    if any(sha(directory/n)!=h for n,h in proof['mapped_files'].items()):raise ValueError('frozen mapped files changed')
    serves=serve_schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,coordinator_hash=sha(__file__),protocol_hash=sha(PROTOCOL),
        driver_hash=sha(prior.engine.WORKER),prior_helper_hash=sha(prior.__file__),
        engine_hash=sha(prior.engine.__file__),completion_helper_hash=sha(prior.completion.__file__),
        physical_helper_hash=sha(prior.physical.__file__),seed_helper_hash=sha(Path(__file__).with_name('run_frozen_hit_confirmation.py')),
        metrics_hash=sha(Path(__file__).with_name('return_metrics.py')),
        frozen_file_hashes={n:sha(directory/n) for n in (*required,*proof['mapped_files'])},
        development_seed=DEV_SEED,development_tag=DEV_TAG,serve_seeds=serves,jobs=declared_jobs(serves),
        policies=list(LINEAGES),camera_modes=list(MODES),warmup_observations=20,
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=10000,
        new_training_observations=0,evaluation_wall_seconds=EVALUATION_SECONDS,total_wall_seconds=TOTAL_SECONDS,
        camera_cap=CAMERA_CAP,workers=4,gc_worker_mode=prior.GC_WORKER_MODE,
        long_bank_entries=0,context_fallback_steps=4,reserved_final_seeds=list(prior.engine.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered feedback study changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns());atomic_json(path,manifest)
    return manifest


def evidence(by_policy,*,draws=10000):
    names=[p+'_'+m for p in LINEAGES for m in MODES]
    if any(len(by_policy.get(n,[]))!=SERVES for n in names):return dict(complete=False)
    for i in range(SERVES):
        first=[by_policy[n][i]['first_incoming'] for n in names]
        geometry=[None if f is None else {k:v for k,v in f.items() if k!='success'} for f in first]
        if any(g!=geometry[0] for g in geometry[1:]):raise ValueError('first incoming trajectory differs before own contact')
    index=np.random.default_rng(BOOTSTRAP_SEED).integers(SERVES,size=(draws,SERVES))
    rates={};points={}
    for n in names:
        data=by_policy[n];hits=np.array([r['own_returns'] for r in data]);opp=np.array([r['own_opportunities'] for r in data])
        present=np.array([r['first_incoming'] is not None for r in data]);first=np.array([bool(r['first_incoming'] and r['first_incoming']['success']) for r in data])
        with np.errstate(divide='ignore',invalid='ignore'):
            rates[n]=dict(first=first[index].sum(1)/present[index].sum(1),all=hits[index].sum(1)/opp[index].sum(1))
        points[n]=dict(first=float(first.sum()/present.sum()) if present.any() else None,
            all=float(hits.sum()/opp.sum()) if opp.sum() else None)
    report=dict(complete=True,serve_clusters=SERVES,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=draws,
        first_incoming_geometry_identical=True,policy_count=len(LINEAGES),contrasts={},joint_sensitivity_finding=False)
    if any(v is None for p in points.values() for v in p.values()):
        return dict(**report,analyzable=False,reason='a full cohort has no incoming opportunity')
    def describe(first,all_rates,point_first,point_all):
        valid=np.isfinite(first)&np.isfinite(all_rates)
        return dict(first_incoming_lift=point_first,all_opportunity_lift=point_all,
            first_incoming_lift_ci95=np.quantile(first[valid],[.025,.975]).tolist() if valid.any() else None,
            all_opportunity_lift_ci95=np.quantile(all_rates[valid],[.025,.975]).tolist() if valid.any() else None,
            invalid_bootstrap_draws=int((~valid).sum()))
    for mode in MODES[1:]:
        per={};delta_first=[];delta_all=[]
        for policy in LINEAGES:
            live=policy+'_live';masked=policy+'_'+mode
            df=rates[live]['first']-rates[masked]['first'];da=rates[live]['all']-rates[masked]['all']
            per[policy]=describe(df,da,points[live]['first']-points[masked]['first'],points[live]['all']-points[masked]['all'])
            delta_first.append(df);delta_all.append(da)
        combined=describe(np.mean(delta_first,axis=0),np.mean(delta_all,axis=0),
            float(np.mean([v['first_incoming_lift'] for v in per.values()])),
            float(np.mean([v['all_opportunity_lift'] for v in per.values()])))
        report['contrasts']['live_minus_'+mode]=dict(equal_policy_average=combined,per_policy=per)
    report['joint_sensitivity_finding']=all(
        v['equal_policy_average']['invalid_bootstrap_draws']==0
        and v['equal_policy_average']['first_incoming_lift_ci95'][0]>0
        and all(p['first_incoming_lift']>0 for p in v['per_policy'].values())
        for v in report['contrasts'].values())
    report['scope']='Fixed-policy-set average; secondary/per-policy intervals descriptive; OOD interventions, not optimal opening-only control or forecast benefit'
    return report


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
        for name in (p+'_'+m for p in LINEAGES for m in MODES):
            rallies=[r for j in jobs if j['policy']==name and j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[name]=rallies;groups[name]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        counts=prior.engine.counts(directory,manifest)
        assert counts['training_fitness']==0 and counts['fresh_evaluation']<=CAMERA_CAP
        paired=evidence(by_policy)
        elapsed=time.time()-start;overall_budget_met=elapsed<=TOTAL_SECONDS
        summary=dict(status='complete' if evaluated['complete'] and overall_budget_met else 'partial',
            overall_budget_met=overall_budget_met,counts=counts,evaluation=evaluated,groups=groups,
            paired_evidence=paired,wall_seconds=elapsed,
            within_allocation_sensitivity_evidence=bool(evaluated['complete'] and overall_budget_met
                and paired.get('joint_sensitivity_finding',False)),
            selection='Both verified final endpoints, fixed before fresh data; no new training/selection',
            final_holdout_unused=list(prior.engine.FINAL))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:supervise(args.directory)
