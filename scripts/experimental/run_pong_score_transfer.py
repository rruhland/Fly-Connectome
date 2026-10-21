"""One frozen .35 score-transfer cohort; no training or opponent selection."""
import argparse,fcntl,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_direct_hit_policy_search as prior
from run_frozen_hit_confirmation import collect_seed_ids
from return_metrics import aggregate

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS=Path('/workspace/m1b-experiments')
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-pong-score-transfer-protocol.md'
LINEAGES={p:EXPERIMENTS/n for p,n in (
    ('seed1','pong-first-return-curriculum-20261021-r1'),
    ('seed2','pong-first-return-replication-20261021-r1'))}
AUDITS={p:EXPERIMENTS/n/'summary.json' for p,n in (
    ('seed1','first-return-curriculum-parent-audit-20261021'),
    ('seed2','first-return-replication-parent-audit-20261021'))}
ARMS=('seed1','seed2')
CONTROLS=('seed1_initial','seed2_initial','initial','noop','up','random1','random10')
POLICIES=(*ARMS,*CONTROLS)
DEV_SEED=1492093789;ACTION_SEED=836917243;TAG=29115;BOOTSTRAP=1236702941
SERVES=64;ROLLOUT_SECONDS=1080.;TOTAL_SECONDS=1200.;CAMERA_CAP=288000
ALLOWED_SOURCE_CHANGES={'control/codec.py','control/durable.py','control/online_vision.py',
    'control/pendulum.py','control/pendulum_session.py','control/pendulum_evaluation.py',
    'control/pong_evaluation.py','control/vision_transfer.py'}
sha=prior.sha;atomic_json=prior.atomic_json


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable registration/evidence changed')
    else:atomic_json(path,value)


def snapshot_exclusions(directory,experiments=EXPERIMENTS):
    path=directory/'seed-exclusions.json'
    if path.exists():return
    ids=set(prior.engine.FINAL)|{849375169,686753009,904271833};sources={}
    def integers(v):
        if type(v) is int and 0<v<2147483647:ids.add(v)
        elif isinstance(v,list):
            for x in v:integers(x)
        elif isinstance(v,dict):
            for x in v.values():integers(x)
    paths=set()
    for pattern in ('manifest.json','*-definition.json','protocol.json','*calibration*.json',
                    'seed-exclusions.json','*registration.json','*recipe.json'):
        paths.update(experiments.rglob(pattern))
    for source in sorted(paths):
        if source.is_relative_to(directory):continue
        value=json.loads(source.read_text());ids.update(collect_seed_ids(value))
        # Newer protocols keep expanded seeds under domain names rather than
        # keys containing "seed"; inherited exclusion snapshots are cumulative.
        if source.name=='seed-exclusions.json':integers(value.get('ids',[]))
        if isinstance(value,dict):
            for key in ('schedules','seed_roots','roots'):integers(value.get(key,{}))
        sources[str(source)]=sha(source)
    immutable_json(path,dict(version='all-prior-expanded-development-domains-v2',ids=sorted(ids),
        metadata_sources=sources,reserved_final_ids=list(prior.engine.FINAL),final_streams_expanded=False))


def schedule(excluded):
    excluded=set(excluded)|set(prior.engine.FINAL)
    if {DEV_SEED,ACTION_SEED,BOOTSTRAP}&excluded:raise ValueError('fresh root collision')
    used=excluded|{DEV_SEED,ACTION_SEED,BOOTSTRAP};seeds=[]
    rng=np.random.default_rng(np.random.SeedSequence([DEV_SEED,TAG]))
    while len(seeds)<SERVES:
        value=int(rng.integers(1,2147483647))
        if value not in used:seeds.append(value);used.add(value)
    actions=[int(np.random.SeedSequence([ACTION_SEED,TAG,i]).generate_state(1)[0]) for i in range(SERVES)]
    if (len(set(actions))!=SERVES
            or set(actions)&(set(prior.engine.FINAL)|set(seeds)|excluded|{DEV_SEED,ACTION_SEED,BOOTSTRAP})):
        raise ValueError('action seed collision')
    return dict(serves=seeds,control_actions=actions)


def declared_jobs(schedules):
    jobs=[]
    for policy in POLICIES:
        trained=policy in (*ARMS,'seed1_initial','seed2_initial')
        for i,seed in enumerate(schedules['serves']):
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            job=prior.rollout_job(f'eval_{policy}_s{i:02d}',seed,policy+'.npy' if trained else None,
                'fresh_evaluation',kind=kind,action_seed=schedules['control_actions'][i] if kind=='random' else None,
                hold=10 if policy=='random10' else 1)
            job['config']['opponent_speed']=.35
            job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def stage(directory):
    """Stage exact qualified heads and seeds without starting a budget clock."""
    directory.mkdir(parents=True,exist_ok=True);current=prior.source_signature();proofs={};fixed={}
    for policy,lineage in LINEAGES.items():
        m=json.loads((lineage/'manifest.json').read_text());s=json.loads((lineage/'summary.json').read_text())
        verification=json.loads((lineage/'completion-verification.json').read_text())
        audit=json.loads(AUDITS[policy].read_text())
        if (s['status']!='complete' or not s['paired_evidence']['arms']['endpoint']['promising_development_signal']
                or not all(p['valid'] and not p['failed'] for p in s['phases'])
                or not verification['passed'] or not audit.get('passed',audit.get('ok',False))
                or m['runtime']!=prior.runtime_signature() or m['model_hash']!=prior.MODEL_HASH
                or m['vision_configuration']!=prior.VISION_CONFIGURATION):
            raise ValueError('both original integrity-verified qualified endpoints required')
        delta={k:dict(before=m['source'].get(k),after=current.get(k)) for k in sorted(set(m['source'])|set(current))
               if m['source'].get(k)!=current.get(k)}
        if set(delta)!=ALLOWED_SOURCE_CHANGES:raise ValueError('reviewed control-only source bridge changed')
        t=np.load(lineage/'search-transform.npy',allow_pickle=False)
        phase=s['phases'][1];theta=np.load(lineage/phase['phase']/phase['theta_file'],allow_pickle=False)
        if not np.array_equal(theta@t,np.load(lineage/'policy_endpoint.npy',allow_pickle=False)):
            raise ValueError('endpoint mapping differs from last full update')
        if not np.array_equal(np.load(lineage/'theta_initial.npy')@t,np.load(lineage/'policy_initial.npy')):
            raise ValueError('own initial mapping differs')
        for target,source in ((policy+'.npy','policy_endpoint.npy'),(policy+'.json','policy_endpoint.json'),
                              (policy+'_initial.npy','policy_initial.npy'),(policy+'_initial.json','policy_initial.json'),
                              (policy+'-projection.json','projection.json'),('radial-cap.json','radial-cap.json'),
                              ('search-transform.npy','search-transform.npy'),('transform-fit.json','transform-fit.json')):
            target_path=directory/target;source_path=lineage/source
            if target_path.exists():
                if sha(target_path)!=sha(source_path):raise ValueError('staged original bytes differ')
            else:shutil.copyfile(source_path,target_path)
            fixed[target]=sha(target_path)
        if policy==ARMS[0]:
            source_path=lineage/'projection.json';target_path=directory/'projection.json'
            if target_path.exists() and sha(target_path)!=sha(source_path):raise ValueError('completion descriptor changed')
            if not target_path.exists():shutil.copyfile(source_path,target_path)
            fixed['projection.json']=sha(target_path)
        paths=[lineage/n for n in ('manifest.json','summary.json','completion-verification.json','source.tar.gz','launch.json')]
        paths+=[AUDITS[policy],lineage/phase['phase']/phase['theta_file'],lineage/'theta_initial.npy']
        proofs[policy]=dict(source_delta=delta,source=m['source'],theta_file=phase['theta_file'],
            ancestor_hashes={str(p):sha(p) for p in paths})
    proof=dict(current_source=current,runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,policies=proofs,fixed_artifacts=fixed,
        original_generic_visual_prior=True,no_training_or_model_selection=True)
    immutable_json(directory/'lineage-proof.json',proof);snapshot_exclusions(directory)
    schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    immutable_json(directory/'schedules.json',schedules)


def pinned_sources():
    modules={'coordinator':__file__,'prior':prior.__file__,'engine':prior.engine.__file__,
        'completion':prior.completion.__file__,'physical':prior.physical.__file__,
        'worker':prior.engine.WORKER,'seed':Path(__file__).with_name('run_frozen_hit_confirmation.py'),
        'metrics':Path(__file__).with_name('return_metrics.py'),'protocol':PROTOCOL}
    return {name:sha(path) for name,path in modules.items()}


def validate_stage(directory):
    proof=json.loads((directory/'lineage-proof.json').read_text())
    if proof['current_source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature():
        raise ValueError('staged source/runtime changed')
    if any(sha(directory/n)!=h for n,h in proof['fixed_artifacts'].items()):raise ValueError('frozen policy/transform changed')
    for p in proof['policies'].values():
        if any(sha(path)!=h for path,h in p['ancestor_hashes'].items()):raise ValueError('ancestor evidence changed')
    schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    if schedules!=json.loads((directory/'schedules.json').read_text()):raise ValueError('seed schedule changed')
    return proof,schedules


def prepare(directory):
    proof,schedules=validate_stage(directory)
    checks=json.loads((directory/'prelaunch-checks.json').read_text())
    if not checks['passed'] or checks['source']!=prior.source_signature():raise ValueError('passing current prelaunch checks required')
    if any(sha(path)!=h for path,h in checks.get('evidence_files',{}).items()):raise ValueError('prelaunch evidence changed')
    required=['lineage-proof.json','seed-exclusions.json','schedules.json','prelaunch-checks.json',*proof['fixed_artifacts']]
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,driver_hash=sha(prior.engine.WORKER),pinned_sources=pinned_sources(),
        frozen_file_hashes={n:sha(directory/n) for n in required},development_seed=DEV_SEED,random_action_seed=ACTION_SEED,
        development_tag=TAG,bootstrap_seed=BOOTSTRAP,bootstrap_draws=10000,schedules=schedules,jobs=declared_jobs(schedules),
        arms=list(ARMS),primary_controls=list(CONTROLS),opponent_speed=.35,episode_seconds=10,
        evaluation_wall_seconds=ROLLOUT_SECONDS,total_wall_seconds=TOTAL_SECONDS,camera_cap=CAMERA_CAP,
        workers=4,gc_worker_mode=prior.GC_WORKER_MODE,new_training_observations=0,final_streams_expanded=False,
        reserved_final_seeds=list(prior.engine.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('registered frozen transfer changed')
    else:m=dict(expected);atomic_json(path,m)
    return m


def score_counts(rallies):
    wins=sum(r['reward']>0 for r in rallies)
    return dict(wins=wins,own_hit_followed_wins=sum(r['reward']>0 and r['own_returns']>0 for r in rallies),
        natural_serve_wins=sum(r['reward']>0 and not r['own_returns'] for r in rallies),
        losses=sum(r['reward']<0 for r in rallies),timeouts=sum(r['truncated'] for r in rallies))


def evidence(groups,draws=10000):
    if any(len(groups.get(n,[]))!=SERVES for n in POLICIES):return dict(complete=False)
    for i in range(SERVES):
        geometry=[None if (f:=groups[n][i]['first_incoming']) is None else {k:v for k,v in f.items() if k!='success'} for n in POLICIES]
        if any(v!=geometry[0] for v in geometry[1:]):raise ValueError('first incoming geometry differs before contact')
    index=np.random.default_rng(BOOTSTRAP).integers(SERVES,size=(draws,SERVES))
    points={};samples={}
    for n in POLICIES:
        rows=groups[n];wins=np.array([r['reward']>0 for r in rows]);own=np.array([r['reward']>0 and r['own_returns']>0 for r in rows])
        points[n]=dict(total=float(wins.mean()),own_hit=float(own.mean()))
        samples[n]=dict(total=wins[index].mean(1),own_hit=own[index].mean(1))
    out=dict(complete=True,serve_clusters=SERVES,first_incoming_geometry_identical=True,bootstrap_seed=BOOTSTRAP,
        bootstrap_draws=draws,controls=list(CONTROLS),arms={},replicated_score_transfer=False)
    for n in ARMS:
        arm={};passes=True
        for metric in ('total','own_hit'):
            delta=samples[n][metric]-np.max([samples[c][metric] for c in CONTROLS],axis=0)
            point=points[n][metric]-max(points[c][metric] for c in CONTROLS)
            ci=np.quantile(delta,[.025,.975]).tolist()
            arm[metric]=dict(lift=point,ci95=ci,strongest_control_rate=max(points[c][metric] for c in CONTROLS))
            passes=passes and point>=.10 and ci[0]>0
        arm['passes']=bool(passes);out['arms'][n]=arm
    out['replicated_score_transfer']=all(a['passes'] for a in out['arms'].values())
    out['fixed_two_policy_average']={metric:dict(lift=float(np.mean([out['arms'][n][metric]['lift'] for n in ARMS])),
        ci95=np.quantile(np.mean([samples[n][metric] for n in ARMS],axis=0)-
            np.max([samples[c][metric] for c in CONTROLS],axis=0),[.025,.975]).tolist()) for metric in ('total','own_hit')}
    return out


def outcome_evidence(directory,m):
    return dict(files={n:sha(directory/n) for n in ('manifest.json','return-annotations.json','rollout-costs.jsonl','rollout-boundary.json')
                      if (directory/n).exists()},jobs={j['id']:dict(
        journal_hash=prior.engine.journal_hash(directory/(j['id']+'.db')) if (directory/(j['id']+'.db')).exists() else None,
        files={suffix:sha(directory/(j['id']+suffix)) for suffix in ('.json','-completion.json')
               if (directory/(j['id']+suffix)).exists()}) for j in m['jobs']})


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('public summary lacks authoritative completion')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained outcome evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained summary changed')
    return saved['summary']


def supervise(directory):
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=prepare(directory)
        old=retained_outcome(directory,m)
        if old is not None:return old
        if 'started_wall_ns' not in m:
            m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        start=m['started_wall_ns']/1e9
        prior_failed=any(not json.loads(line)['ok'] for line in (directory/'rollout-costs.jsonl').read_text().splitlines()) if (directory/'rollout-costs.jsonl').exists() else False
        boundary=directory/'rollout-boundary.json'
        if not boundary.exists():
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
                evaluated=prior.engine.execute_jobs(directory,m['jobs'] if not prior_failed else [],pool,
                    deadline=start+ROLLOUT_SECONDS,camera_cap=CAMERA_CAP,
                    initial_consumed=prior.engine.counts(directory,m)['fresh_evaluation'],
                    completion_report=prior.completion.complete_report,completion_pin=prior.completion.pin_completion)
            evaluated['failed']=evaluated['failed'] or prior_failed
            evaluated['complete']=all(prior.completion.complete_report(directory,j) is not None for j in m['jobs'])
            atomic_json(boundary,dict(frozen_wall_ns=time.time_ns(),evaluation=evaluated))
        saved_boundary=json.loads(boundary.read_text())
        evaluated=saved_boundary['evaluation']
        cost_failed=any(not json.loads(line)['ok'] for line in (directory/'rollout-costs.jsonl').read_text().splitlines()) if (directory/'rollout-costs.jsonl').exists() else False
        if (evaluated['complete']!=all(prior.completion.complete_report(directory,j) is not None for j in m['jobs'])
                or evaluated['committed_cameras']!=prior.engine.counts(directory,m)['fresh_evaluation']
                or evaluated['failed']!=cost_failed):
            raise ValueError('frozen rollout boundary changed')
        annotations=prior.physical.annotate(directory,m['jobs']);atomic_json(directory/'return-annotations.json',annotations)
        by_policy={};groups={}
        for policy in POLICIES:
            jobs=sorted((j for j in m['jobs'] if j['policy']==policy),key=lambda j:j['serve_index'])
            assert [j['serve_index'] for j in jobs]==list(range(SERVES))
            rallies=[r for j in jobs if j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
            if groups[policy]['natural_serve_wins']!=score_counts(rallies)['natural_serve_wins']:
                raise ValueError('natural win annotations disagree')
            groups[policy].update(score_counts(rallies))
        paired=evidence(by_policy);counts=prior.engine.counts(directory,m)
        reports=[(j,prior.engine.read_report(directory,j['id'])) for j in m['jobs']]
        logical=dict(initial_images=sum(r.get('initial_cameras',0) for _,r in reports if r),
            transition_images=counts['fresh_evaluation'],
            encoded_images=sum(r.get('initial_cameras',0)+r['transitions'] for j,r in reports
                               if r and not j['config']['skip_baseline_perception']),
            annotation_physics_reexecutions=sum(a['verified_camera_rows'] for a in annotations.values()),
            new_visual_learning_images=0,new_actor_updates=0,
            partial_receipt_caveat='Initial/encoded image counts use retained receipts; journal transition counts are authoritative')
        retained=outcome_evidence(directory,m);elapsed=time.time()-start
        if counts['training_fitness'] or counts['fresh_evaluation']>CAMERA_CAP:raise ValueError('registered camera allocation exceeded')
        rollout_elapsed=(saved_boundary['frozen_wall_ns']-m['started_wall_ns'])/1e9
        valid=bool(evaluated['complete'] and not evaluated['failed'] and rollout_elapsed<=ROLLOUT_SECONDS and elapsed<=TOTAL_SECONDS
            and all(g['complete'] for g in groups.values()))
        summary=dict(status='complete' if valid else 'partial',evidence_eligible=valid,overall_budget_met=elapsed<=TOTAL_SECONDS,
            rollout_wall_seconds=rollout_elapsed,rollout_budget_met=rollout_elapsed<=ROLLOUT_SECONDS,
            wall_seconds=elapsed,counts=counts,logical_compute=logical,evaluation=evaluated,groups=groups,paired_score_evidence=paired,
            replicated_score_transfer=valid and paired.get('replicated_score_transfer',False),
            scope='One frozen .35 return-policy score transfer; not score-reward acquisition or complete M1B',final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=retained))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:supervise(args.directory)
