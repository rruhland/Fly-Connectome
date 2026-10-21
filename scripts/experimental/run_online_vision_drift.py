"""Registered fixed-policy visual-plasticity study; explicit recipe required.

Reviewed runtime caps/roots come from the immutable recipe. Staging copies and
pins inputs without observing a camera or starting the allocation clock.
"""
import argparse,fcntl,json,multiprocessing as mp,os,shutil,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import numpy as np

from fly_connectome.control.online_vision import PersistentVisionSession
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from run_frozen_hit_confirmation import snapshot_exclusions
from run_direct_hit_policy_search import initialize_worker
from run_pong_development import atomic_json,FINAL_SEEDS
import persistent_vision_worker as worker
import online_vision_metrics as metrics

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-online-vision-drift-protocol.md'
LINEAGES=(Path('/workspace/m1b-experiments/pong-first-return-curriculum-20261021-r1'),
          Path('/workspace/m1b-experiments/pong-first-return-replication-20261021-r1'))
AUDITS=(Path('/workspace/m1b-experiments/first-return-curriculum-parent-audit-20261021'),
        Path('/workspace/m1b-experiments/first-return-replication-parent-audit-20261021'))
PHASES=('exposure','probes','prequential')
DRAFT_ONLY=False
sha=worker.sha


def validate_recipe(recipe):
    fields={'version','seed_roots','seed_tag','bootstrap_seed','wall_seconds'}
    if set(recipe)!=fields or recipe['version']!='persistent-fixed-policy-drift-v1':
        raise ValueError('reviewed finite drift recipe required')
    roots=recipe['seed_roots']
    if set(roots)!=set(PHASES) or len(set(roots.values()))!=3:
        raise ValueError('three distinct exposure/probe/prequential roots required')
    if any(type(s) is not int or not 0<s<2147483647 or s in FINAL_SEEDS
           for s in [*roots.values(),recipe['seed_tag'],recipe['bootstrap_seed']]):
        raise ValueError('positive fresh development seeds required')
    limits=recipe['wall_seconds']
    if (set(limits)!={'total',*PHASES,'scoring_reserve'}
            or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in limits.values())
            or sum(limits[p] for p in PHASES)+limits['scoring_reserve']>limits['total']):
        raise ValueError('positive prospective phase/overall/scoring limits required')


def pinned_sources():
    names=['scripts/experimental/run_online_vision_drift.py',
        'scripts/experimental/persistent_vision_worker.py',
        'scripts/experimental/online_vision_metrics.py',
        'scripts/experimental/run_pong_development.py',
        'scripts/experimental/run_pong_policy_search_continuation.py',
        'scripts/experimental/run_direct_hit_policy_search.py',
        'scripts/experimental/run_frozen_hit_confirmation.py',str(PROTOCOL.relative_to(ROOT))]
    return {name:sha(ROOT/name) for name in names}


def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('fixed actor/transform artifact changed')
    else:shutil.copyfile(source,target)


def expand(recipe,excluded):
    result={};used=set(excluded)|set(FINAL_SEEDS)|set(recipe['seed_roots'].values())
    for phase in PHASES:
        rng=np.random.default_rng(np.random.SeedSequence([recipe['seed_roots'][phase],recipe['seed_tag']]))
        values=[]
        while len(values)<64:
            seed=int(rng.integers(1,2147483647))
            if seed not in used:values.append(seed);used.add(seed)
        result[phase]=values
    return result


def stage(directory,recipe,preflight):
    validate_recipe(recipe)
    if (not preflight['passed'] or preflight['source']!=source_signature()
            or preflight['runtime']!=runtime_signature() or not preflight['actual_fixed_policy_default_bridge']):
        raise ValueError('current-source runtime and actual-policy bridge preflight required')
    for path,digest in preflight['evidence_file_hashes'].items():
        if sha(Path(path))!=digest:raise ValueError('validated preflight evidence changed')
    directory.mkdir(parents=True,exist_ok=True);snapshot_exclusions(directory)
    exclusion=json.loads((directory/'seed-exclusions.json').read_text())
    if not set(preflight['diagnostic_seed_ids']).issubset(exclusion['ids']):
        raise ValueError('preflight actual seed IDs must be excluded from study cohorts')
    if set(recipe['seed_roots'].values())&set(exclusion['ids']):raise ValueError('fresh root collision')
    schedules=expand(recipe,exclusion['ids']);policies={};fixed={};lineage_pins={}
    for i,(lineage,audit) in enumerate(zip(LINEAGES,AUDITS),1):
        label=f'seed{i}';summary=json.loads((lineage/'summary.json').read_text())
        own=json.loads((lineage/'completion-verification.json').read_text());parent=json.loads((audit/'summary.json').read_text())
        if not (summary['status']=='complete' and summary['paired_evidence']['arms']['endpoint']['promising_development_signal']
                and own['passed'] and parent['passed']):raise ValueError('both predetermined qualified FULL policies required')
        lineage_pins[label]={str(p):sha(p) for p in (lineage/'manifest.json',lineage/'summary.json',
            lineage/'completion-verification.json',audit/'summary.json')}
        for filename in ('policy_endpoint.npy','radial-cap.json','search-transform.npy','transform-fit.json'):
            target=directory/(label+'-'+filename);immutable_copy(lineage/filename,target);fixed[target.name]=sha(target)
        cap=json.loads((directory/(label+'-radial-cap.json')).read_text())
        config=PongEvaluationConfig(schedules['exposure'][0],episodes=64,max_episode_ticks=1200,
            opponent_speed=.65,policy_scaling=cap)
        session=PersistentVisionSession(np.load(directory/(label+'-policy_endpoint.npy'),allow_pickle=False),
            config,learn_enabled=True,episode_seeds=schedules['exposure'])
        snapshot=worker.save_snapshot(directory/(label+'-before.pkl'),session.retained_vision())
        fixed[snapshot['file']]=snapshot['file_sha256']
        policies[label]=dict(weights_from=label+'-policy_endpoint.npy',cap=cap,before=snapshot['file'])
    expected=dict(version=recipe['version'],recipe=recipe,source=source_signature(),runtime=runtime_signature(),
        model_hash=MODEL_HASH,vision_configuration=VISION_CONFIGURATION,pinned_sources=pinned_sources(),
        worker_sha256=sha(worker.__file__),policies=policies,fixed_artifacts=fixed,
        lineage_pins=lineage_pins,schedules=schedules,seed_exclusions_sha256=sha(directory/'seed-exclusions.json'),
        preflight=preflight,transition_dose_per_stream=2000,probe_serves=64,
        phase_camera_caps=dict(exposure=4000,probes=128000,prequential=4000),
        reserved_final_seeds=list(FINAL_SEEDS),final_streams_expanded=False)
    path=directory/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('registered drift study changed')
    else:atomic_json(path,dict(**expected,jobs=[]))


def validate_manifest(directory):
    manifest=json.loads((directory/'manifest.json').read_text());validate_recipe(manifest['recipe'])
    if (manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature()
            or manifest['pinned_sources']!=pinned_sources() or manifest['worker_sha256']!=sha(worker.__file__)
            or sha(directory/'seed-exclusions.json')!=manifest['seed_exclusions_sha256']):
        raise ValueError('drift source/runtime/registration changed')
    for name,digest in manifest['fixed_artifacts'].items():
        if sha(directory/name)!=digest:raise ValueError('drift fixed artifact changed')
    if expand(manifest['recipe'],json.loads((directory/'seed-exclusions.json').read_text())['ids'])!=manifest['schedules']:
        raise ValueError('registered drift seed schedule changed')
    for pins in manifest['lineage_pins'].values():
        if any(sha(Path(path))!=digest for path,digest in pins.items()):raise ValueError('qualified lineage changed')
    if any(sha(Path(path))!=digest for path,digest in manifest['preflight']['evidence_file_hashes'].items()):
        raise ValueError('registered preflight evidence changed')
    return manifest


def jobs_for(directory,manifest,phase):
    result=[]
    for policy,spec in manifest['policies'].items():
        if phase=='exposure':cases=[(policy+'_exposure',manifest['schedules'][phase],True,spec['before'],'exposure')]
        else:
            definition=next(j for j in manifest['jobs'] if j['id']==policy+'_exposure')
            exposure=worker.complete_report(directory,definition)
            if exposure is None:raise ValueError('verified exact exposure dose required before downstream jobs')
            after=exposure['retained_snapshot']['file']
            if phase=='prequential':cases=[(policy+'_prequential',manifest['schedules'][phase],True,after,'prequential')]
            else:cases=[(f'{policy}_{mode}_s{i:02d}',[seed],False,snapshot,'probe')
                for mode,snapshot in (('before',spec['before']),('after',after))
                for i,seed in enumerate(manifest['schedules']['probes'])]
        for name,seeds,learn,snapshot,purpose in cases:
            config=PongEvaluationConfig(seeds[0],episodes=len(seeds),max_episode_ticks=1200,
                opponent_speed=.65,policy_scaling=spec['cap'])
            job=dict(id=name,policy=policy,purpose=purpose,phase=phase,config=asdict(config),
                probe_condition=('before' if snapshot==spec['before'] else 'after') if purpose=='probe' else None,
                episode_seeds=seeds,learn_enabled=learn,weights_from=spec['weights_from'],snapshot_from=snapshot,
                camera_dose=None if purpose=='probe' else 2000,
                input_hashes={name:sha(directory/name) for name in (spec['weights_from'],snapshot)})
            path=directory/(name+'-spec.json')
            if path.exists() and json.loads(path.read_text())!=job:raise ValueError('persistent phase job changed')
            atomic_json(path,job);result.append(job)
    existing={j['id']:j for j in manifest['jobs']}
    for job in result:
        if job['id'] in existing and existing[job['id']]!=job:raise ValueError('registered job changed')
        existing[job['id']]=job
    manifest['jobs']=list(existing.values());atomic_json(directory/'manifest.json',manifest)
    return result


def execute_phase(directory,manifest,phase,pool):
    path=directory/(phase+'-boundary.json');limits=manifest['recipe']['wall_seconds']
    index=PHASES.index(phase)
    started=manifest['started_wall_ns'] if index==0 else json.loads(
        (directory/(PHASES[index-1]+'-boundary.json')).read_text())['frozen_wall_ns']
    overall=manifest['started_wall_ns']+int((limits['total']-limits['scoring_reserve'])*1e9)
    expected=dict(phase=phase,started_wall_ns=started,
                  deadline_wall_ns=min(started+int(limits[phase]*1e9),overall))
    if path.exists():
        boundary=json.loads(path.read_text())
        if any(boundary.get(k)!=v for k,v in expected.items()):raise ValueError('durable phase clock changed')
    else:
        boundary=expected
        atomic_json(path,boundary)
    jobs=jobs_for(directory,manifest,phase);deadline=boundary['deadline_wall_ns']/1e9
    pending=[j for j in jobs if worker.complete_report(directory,j) is None];active={};failed=False
    costs=directory/(phase+'-costs.jsonl')
    if costs.exists():failed=any(not json.loads(line)['ok'] for line in costs.read_text().splitlines())
    if 'frozen_wall_ns' in boundary:
        if (boundary['complete']!=all(worker.complete_report(directory,j) is not None for j in jobs)
                or boundary['failed']!=failed
                or boundary['budget_met']!=(boundary['frozen_wall_ns']<=boundary['deadline_wall_ns'])):
            raise ValueError('frozen phase completeness/failure/budget changed')
        return boundary
    while active or (pending and not failed and time.time()<deadline):
        while pending and len(active)<4 and not failed and time.time()<deadline:
            job=pending.pop(0);active[pool.submit(worker.execute_job,str(directory),job,deadline)]=job
        if not active:break
        ready,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
        for future in ready:
            job=active.pop(future)
            try:cost=future.result()
            except BaseException:cost=dict(ok=False,job=job['id'],error=traceback.format_exc());failed=True
            failed=failed or not cost.get('ok',False)
            with costs.open('a') as f:f.write(json.dumps(cost)+'\n');f.flush();os.fsync(f.fileno())
    complete=all(worker.complete_report(directory,j) is not None for j in jobs)
    boundary.update(frozen_wall_ns=time.time_ns(),complete=complete,failed=failed)
    boundary['budget_met']=boundary['frozen_wall_ns']<=boundary['deadline_wall_ns']
    atomic_json(path,boundary);return boundary


def outcome_evidence(directory,manifest):
    """Bind final clocks, raw journals and reports without executing a camera."""
    files=['manifest.json','return-annotations.json','cost-summary.json']
    files.extend(p+'-boundary.json' for p in PHASES if (directory/(p+'-boundary.json')).exists())
    files.extend(p+'-costs.jsonl' for p in PHASES if (directory/(p+'-costs.jsonl')).exists())
    jobs={}
    for job in manifest['jobs']:
        name=job['id'];database=directory/(name+'.db')
        pin=directory/(name+'-completion.json')
        if pin.exists():worker.complete_report(directory,job)
        jobs[name]=dict(
            journal_hash=None if not database.exists() else worker.journal_hash(database),
            files={suffix:sha(directory/(name+suffix)) for suffix in
                   ('-spec.json','.json','-completion.json') if (directory/(name+suffix)).exists()})
    return dict(files={name:sha(directory/name) for name in files},jobs=jobs)


def retained_outcome(directory,manifest):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():
            raise ValueError('summary without an authoritative retained outcome')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,manifest):
        raise ValueError('finalized drift evidence changed')
    summary_path=directory/'summary.json'
    if summary_path.exists():
        if json.loads(summary_path.read_text())!=saved['summary']:
            raise ValueError('finalized drift summary changed')
    else:atomic_json(summary_path,saved['summary'])
    return saved['summary']


def cost_summary(directory,manifest):
    """Separate logical data, recovery work and process/task wall accounting."""
    import sqlite3
    totals={key:0 for key in ('new_camera_transitions','new_initial_images','new_visual_observations',
            'replayed_camera_transitions','replayed_initial_images','replayed_visual_observations')}
    task_wall=0.;failed=0;latencies={};process_peaks=[]
    for phase in PHASES:
        path=directory/(phase+'-costs.jsonl')
        if not path.exists():continue
        for line in path.read_text().splitlines():
            cost=json.loads(line)
            if not cost['ok']:failed+=1
            task_wall+=cost.get('wall_seconds',0.)
            if 'process_peak_rss_kib' in cost:process_peaks.append(cost['process_peak_rss_kib'])
            for key in totals:totals[key]+=cost.get(key,0)
    for job in manifest['jobs']:
        path=directory/(job['id']+'.db')
        if not path.exists():continue
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as connection:
            events=connection.execute('SELECT kind,count,wall_ns FROM replay_timing ORDER BY id').fetchall()
        label=job['phase']+('_'+job['probe_condition'] if job.get('probe_condition') else '')
        values=latencies.setdefault(label,[])
        values.extend((b[2]-a[2])/1e6 for a,b in zip(events,events[1:])
                      if a[0]==b[0]=='heartbeat' and b[1]==a[1]+1)
    return dict(**totals,summed_settled_task_wall_seconds=task_wall,failed_task_records=failed,
        maximum_worker_process_lifetime_peak_rss_kib=max(process_peaks,default=None),
        heartbeat_latency_ms={label:dict(samples=len(values),
            quantiles=np.quantile(values,[.5,.95,.99,1.]).tolist()) for label,values in latencies.items() if values},
        latency_definition='Consecutive committed-camera heartbeat wall intervals, including snapshots; excludes process starts/stops and recovery boundaries',
        caveat='Settled task costs do not measure unobserved crash tails; process RSS peaks may include earlier jobs in the same worker',
        per_stream_bank_and_resident_memory='Retained *-progress.jsonl at every 200 transitions; final counts in each report')


def supervise(directory):
    """Fixed-dose exposure, frozen probes and separate plastic prequential play."""
    if DRAFT_ONLY:raise RuntimeError('unallocated static draft; execution is disabled')
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest=validate_manifest(directory)
        retained=retained_outcome(directory,manifest)
        if retained is not None:
            print(json.dumps(retained),flush=True);return retained
        if 'started_wall_ns' not in manifest:
            manifest['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',manifest)
        phases=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            for phase in PHASES:
                phases.append(execute_phase(directory,manifest,phase,pool))
                if phase=='exposure' and not phases[-1]['complete']:break
        scoring_started=time.monotonic();annotations={};groups={};streams={}
        for job in manifest['jobs']:
            result=metrics.annotate_job(directory,job);annotations[job['id']]=result
            if job['purpose']=='probe':
                if worker.complete_report(directory,job) is not None:
                    if len(result['rallies'])!=1 or result['rallies'][0]['right_censored']:
                        raise ValueError('completed frozen probe is not one complete rally')
                    name=job['policy']+'_'+job['probe_condition'];groups.setdefault(name,[]).extend(result['rallies'])
            else:streams[job['id']]=result
        atomic_json(directory/'return-annotations.json',annotations)
        evidence=metrics.paired_snapshot_effects(groups,seed=manifest['recipe']['bootstrap_seed'])
        cohort_summary={name:metrics.aggregate(rallies) for name,rallies in groups.items()}
        costs=cost_summary(directory,manifest);atomic_json(directory/'cost-summary.json',costs)
        elapsed=(time.time_ns()-manifest['started_wall_ns'])/1e9
        overall=elapsed<=manifest['recipe']['wall_seconds']['total']
        complete=bool(len(phases)==3 and all(p['complete'] and p['budget_met'] and not p['failed'] for p in phases)
                      and overall and evidence.get('complete',False))
        summary=dict(status='complete' if complete else 'partial',phases=phases,overall_budget_met=overall,
            frozen_probe_evidence_eligible=bool(complete and evidence.get('analyzable',True)),
            effects=evidence,probe_groups=cohort_summary,streams=streams,
            costs=costs,
            wall_seconds=elapsed,scoring_wall_seconds=time.monotonic()-scoring_started,
            logical_camera_transitions=sum(a['camera_rows'] for a in annotations.values()),
            logical_visual_images=sum(a['logical_visual_images'] for a in annotations.values()),
            scoring_physics_reexecutions=sum(a['camera_rows'] for a in annotations.values()),
            new_observations_from_scoring=0,final_streams_expanded=False,
            interpretation='Fixed-policy plastic-vision integration; no new actor acquisition or formal noninferiority claim')
        # The authoritative outcome is written first. A crash before the public
        # summary export restores these exact bytes/eligibility without scoring
        # again against a later wall clock.
        atomic_json(directory/'study-completion.json',
                    dict(summary=summary,evidence=outcome_evidence(directory,manifest)))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--recipe',type=Path);parser.add_argument('--preflight',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.stage:
        if args.recipe is None or args.preflight is None:parser.error('reviewed recipe/preflight are required before staging')
        stage(args.directory,json.loads(args.recipe.read_text()),json.loads(args.preflight.read_text()))
    else:supervise(args.directory)
