"""Prospective epsilon-only survival comparison with a verified reused control.

Importing does not expand roots, create models, or start a clock. Execution needs
a sealed, reviewed prospective allocation after the native-score audits.
"""
import argparse,copy,fcntl,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.control.pendulum_evaluation import PendulumEvaluationConfig
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
import run_pendulum_acquisition as base
import run_pendulum_continuation as prior
import run_pong_score_transfer as seeds
import pendulum_acquisition_worker as worker
import pendulum_acquisition_metrics as metrics
import online_vision_metrics as pong_metrics

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS=Path('/workspace/m1b-experiments')
OLD=EXPERIMENTS/'pendulum-onoff-continuation-20261021-r1'
AUDIT=EXPERIMENTS/'pendulum-onoff-parent-audit-20261021'
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-pendulum-low-exploration-protocol.md'
DRAFT_ONLY=False
NEW_ROOTS=dict(development=1023917563,control_actions=1735402891)
TAG=29118;BOOTSTRAP=625198307
REUSED_ROOTS=dict(environment=1083294751,learner=1647809237)
DOSE=20000;CHECKPOINTS=(5000,10000,20000);EPSILON=.02
TRAIN_SECONDS=1200.;EVAL_SECONDS=1800.;TOTAL_SECONDS=3000.
sha=worker.sha;atomic_json=base.atomic_json;immutable=prior.immutable


def pinned_sources():
    names=('run_pendulum_low_exploration.py','run_pendulum_continuation.py','run_pendulum_acquisition.py',
           'pendulum_acquisition_worker.py','pendulum_acquisition_metrics.py','persistent_vision_worker.py',
           'online_vision_metrics.py','run_pong_score_transfer.py','return_metrics.py','run_pong_development.py',
           'run_pong_policy_search_continuation.py','run_direct_hit_policy_search.py')
    return {**{'scripts/experimental/'+n:sha(ROOT/'scripts/experimental'/n) for n in names},
            str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}


def schedule(excluded):
    if NEW_ROOTS is None or TAG is None or BOOTSTRAP is None:
        raise ValueError('prospective fresh development/action/bootstrap roots not registered')
    excluded=set(excluded)|set(base.FINAL_SEEDS)
    roots={*NEW_ROOTS.values(),BOOTSTRAP}
    if len(roots)!=3 or roots&excluded or roots&set(REUSED_ROOTS.values()):
        raise ValueError('new root collision')
    # Old environment/exploration roots are an explicit, narrow reuse exception.
    used=excluded|roots|set(REUSED_ROOTS.values());values=[]
    rng=np.random.default_rng(np.random.SeedSequence([NEW_ROOTS['development'],TAG]))
    while len(values)<64:
        value=int(rng.integers(1,2147483647))
        if value not in used:values.append(value);used.add(value)
    actions=[int(np.random.SeedSequence([NEW_ROOTS['control_actions'],TAG,i]).generate_state(1)[0]) for i in range(64)]
    if len(set(actions))!=64 or set(actions)&used:raise ValueError('control action seed collision')
    return dict(development=values,control_actions=actions)


def archived_a_reference(files):
    """Bind cold bytes to the previously audited semantic pins, without restore."""
    root=prior.OLD;manifest=root/'journal-archive-manifest.json';report=root/'journal-archive-report.json'
    archive=json.loads(manifest.read_text());receipt=json.loads(report.read_text())
    if (archive['root']!=str(root) or not receipt['passed']
            or not receipt['all_original_bytes_reconstruct_verified']):
        raise ValueError('verified byte-exact ancestor archive required')
    files.update({str(manifest):sha(manifest),str(report):sha(report)})
    old_proof=json.loads((OLD/'a-off-reuse-proof.json').read_text());entries={}
    for i in range(64):
        name=f'a_experienced_s{i:02d}';expected=old_proof['old_jobs'][name]
        pin_path=root/(name+'-completion.json');pin=json.loads(pin_path.read_text())
        if (sha(pin_path)!=expected['completion_sha256'] or pin['journal_hash']!=expected['journal_hash']
                or sha(root/(name+'.json'))!=expected['report_sha256']
                or pin['report_sha256']!=expected['report_sha256']):
            raise ValueError('ancestor semantic pin differs from the completed audit chain')
        entry=archive['entries'][name+'.db'];compressed=root/(name+'.db.zst')
        if (not entry['fully_stream_verified_before_unlink'] or sha(compressed)!=entry['archive_sha256']
                or compressed.stat().st_size!=entry['archive_bytes']):
            raise ValueError('ancestor compressed journal changed')
        files.update({str(compressed):entry['archive_sha256'],str(pin_path):sha(pin_path),
                      str(root/(name+'.json')):expected['report_sha256']})
        entries[name]=dict(archive_entry=entry,audited_semantic_pin=pin)
    return dict(archive_manifest_sha256=sha(manifest),archive_report_sha256=sha(report),entries=entries,
        verification='Previously reconstructed original bytes verified before unlink; current compressed bytes and audited semantic pins match',
        decompressions=0,new_camera_observations=0)


def reused_control_proof():
    """No reexecution: bind the already audited training and A control evidence."""
    m=json.loads((OLD/'manifest.json').read_text());summary=json.loads((OLD/'summary.json').read_text())
    if (summary['status']!='complete' or m['source']!=source_signature()
            or m['runtime']!=runtime_signature()):raise ValueError('valid identical-core control required')
    for path in (OLD/'completion-verification.json',AUDIT/'summary.json'):
        if not json.loads(path.read_text())['passed']:raise ValueError('control integrity proof missing')
    train=next(j for j in m['jobs'] if j['id']=='on_train')
    expected=dict(seed=REUSED_ROOTS['environment'],learner_seed=REUSED_ROOTS['learner'],
                  max_episode_ticks=600,alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95)
    if (train['config']!=expected or train['learn_enabled'] is not True or train['camera_dose']!=DOSE
            or train['checkpoints']!=list(CHECKPOINTS) or train['snapshot_from']!='initial-vision.pkl'
            or train['initial_weights_from']!='initial-theta.npy'):
        raise ValueError('reused control differs from the fixed epsilon-only comparison')
    report=worker.complete_report(OLD,train)
    if report is None:raise ValueError('reused control training incomplete')
    files={name:sha(OLD/name) for name in ('manifest.json','summary.json','study-completion.json',
        'completion-verification.json','a-after-annotations.json','a-off-reuse-proof.json','on_train.json','on_train-completion.json')}
    files[str(AUDIT/'summary.json')]=sha(AUDIT/'summary.json')
    ancestral=prior.OLD/'a-after-annotations.json'
    if m['ancestor_evidence'].get(str(ancestral))!=sha(ancestral):
        raise ValueError('reused E20k A baseline differs from the audited ancestor')
    files[str(ancestral)]=sha(ancestral)
    for export in report['exports'].values():
        for name,digest in export['files'].items():
            if sha(OLD/name)!=digest:raise ValueError('reused control export changed')
            files[name]=digest
    # These journals are plain; the older E20k A baseline is retained via the
    # completed ON/OFF proof chain and its byte-exact cold archive, not restored.
    cap=json.loads((OLD/'a-cap.json').read_text());a_pins={}
    for i,seed in enumerate(m['a_reused_development_seeds']):
        name=f'a_on_s{i:02d}';job=next(j for j in m['jobs'] if j['id']==name)
        config=asdict(PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap))
        if (job['config']!=config or job.get('learn_enabled') is not False
                or job['episode_seeds']!=[seed] or job['snapshot_from']!='on_at20000-vision.pkl'
                or job['weights_from']!='a-weights.npy'):
            raise ValueError('reused control A configuration changed')
        if worker.complete_report(OLD,job) is None:raise ValueError('reused A trial incomplete')
        a_pins[name]=json.loads((OLD/(name+'-completion.json')).read_text())
        for suffix in ('.json','-completion.json','-spec.json'):files[name+suffix]=sha(OLD/(name+suffix))
    archive=archived_a_reference(files)
    return dict(version='verified-epsilon-control-reuse-v1',files=files,ancestral_a_archive=archive,
        training_pin=json.loads((OLD/'on_train-completion.json').read_text()),a_pins=a_pins,
        inherited_control_training_transitions=DOSE,reused_control_a_trials=64,
        new_observations_from_reuse=0,only_training_config_change=dict(epsilon=[.2,EPSILON]))


def validate_reused_journals(proof):
    jobs={j['id']:j for j in json.loads((OLD/'manifest.json').read_text())['jobs']}
    expected={'on_train':proof['training_pin'],**proof['a_pins']}
    for name,pin in expected.items():
        if worker.pin_payload(OLD,jobs[name])!=pin:
            raise ValueError('reused control journal contents changed')


def stage(directory):
    if DRAFT_ONLY:raise ValueError('draft only: preceding allocation and audit lock still active')
    directory.mkdir(parents=True,exist_ok=True);seeds.snapshot_exclusions(directory)
    checks=json.loads((directory/'prelaunch-checks.json').read_text())
    if (not checks['passed'] or checks['source']!=source_signature()
            or any(sha(p)!=h for p,h in checks['evidence_files'].items())):
        raise ValueError('passing current prelaunch checks required')
    proof=reused_control_proof();immutable(directory/'control-reuse-proof.json',proof)
    old=json.loads((OLD/'manifest.json').read_text());fixed={
        'prelaunch-checks.json':sha(directory/'prelaunch-checks.json'),
        'control-reuse-proof.json':sha(directory/'control-reuse-proof.json')}
    originals=('initial-vision.pkl','generic-vision.pkl','initial-theta.npy','initial-weights.npy',
        'initial-cap.json','search-transform.npy','transform-fit.json','a-weights.npy','a-cap.json')
    for name in originals:
        if sha(OLD/name)!=old['fixed_artifacts'][name]:raise ValueError('common starting input changed')
        base.immutable_copy(OLD/name,directory/name);fixed[name]=sha(directory/name)
    for dose in CHECKPOINTS:
        for suffix in ('-theta.npy','-weights.npy','-cap.json','-vision.pkl'):
            origin=f'on_at{dose}{suffix}';target=f'control_at{dose}{suffix}'
            if sha(OLD/origin)!=proof['files'][origin]:raise ValueError('control checkpoint changed')
            base.immutable_copy(OLD/origin,directory/target);fixed[target]=sha(directory/target)
    projection=copy.deepcopy(old['projection']);projection['matrix_file']=str(directory/'search-transform.npy')
    p=FixedPolicyProjection(4389,projection)
    for label in ('initial',*(f'control_at{dose}' for dose in CHECKPOINTS)):
        if not np.array_equal(p.mapped_weights(np.load(directory/(label+'-theta.npy'),allow_pickle=False)),
                              np.load(directory/(label+'-weights.npy'),allow_pickle=False)):
            raise ValueError('theta/raw-head mapping changed')
    expected=dict(version='plastic-survival-epsilon-only-reused-control-v1',source=source_signature(),
        runtime=runtime_signature(),model_hash=MODEL_HASH,vision_configuration=VISION_CONFIGURATION,
        pinned_sources=pinned_sources(),seed_roots={**REUSED_ROOTS,**NEW_ROOTS},seed_tag=TAG,
        bootstrap_seed=BOOTSTRAP,bootstrap_draws=10000,
        seed_reuse_exception='Only environment and learner roots deliberately match the audited .2 control',
        seed_exclusions_sha256=sha(directory/'seed-exclusions.json'),
        schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids']),
        fixed_artifacts=fixed,projection=projection,control_reuse=proof,
        a_reused_development_seeds=old['a_reused_development_seeds'],
        transitions_per_treatment=DOSE,checkpoints=list(CHECKPOINTS),training_seconds=TRAIN_SECONDS,
        evaluation_seconds=EVAL_SECONDS,total_seconds=TOTAL_SECONDS,prior_b_policy_updates=20000,
        inherited_visual_learning_images=22282,treatment_epsilon=EPSILON,control_epsilon=.2,
        final_streams_expanded=False,reserved_final_ids=list(base.FINAL_SEEDS))
    path=directory/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('staged epsilon comparison changed')
    else:atomic_json(path,dict(**expected,jobs=[]))


def validate_manifest(directory):
    m=json.loads((directory/'manifest.json').read_text())
    if (m['source']!=source_signature() or m['runtime']!=runtime_signature() or m['pinned_sources']!=pinned_sources()
            or m['seed_roots']!={**REUSED_ROOTS,**NEW_ROOTS}
            or sha(directory/'seed-exclusions.json')!=m['seed_exclusions_sha256']
            or m['schedules']!=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])):
        raise ValueError('epsilon study source/runtime/seeds changed')
    for name,digest in m['fixed_artifacts'].items():
        if sha(directory/name)!=digest:raise ValueError('pinned epsilon-study input changed')
    for name,digest in m['control_reuse']['files'].items():
        path=Path(name) if Path(name).is_absolute() else OLD/name
        if sha(path)!=digest:raise ValueError('reused control provenance changed')
    validate_reused_journals(m['control_reuse'])
    return m


def training_jobs(m):
    refs=('initial-vision.pkl','initial-theta.npy','search-transform.npy')
    return [dict(id='low_train',arm='low',kind='train',phase='training',camera_dose=DOSE,checkpoints=list(CHECKPOINTS),
        snapshot_from='initial-vision.pkl',initial_weights_from='initial-theta.npy',learn_enabled=True,
        input_hashes={n:m['fixed_artifacts'][n] for n in refs},
        config=dict(seed=REUSED_ROOTS['environment'],learner_seed=REUSED_ROOTS['learner'],max_episode_ticks=600,
                    alpha=.001,lam=.995,epsilon=EPSILON,gamma_per_second=.95))]


def evaluation_jobs(directory,m):
    jobs=[]
    def add(name,kind,group,i,seed,snapshot,weights,config):
        refs=[snapshot]+([] if weights is None else [weights])
        jobs.append(dict(id=name,kind=kind,phase='evaluation',group=group,serve_index=i,config=asdict(config),
            snapshot_from=snapshot,weights_from=weights,input_hashes={n:sha(directory/n) for n in refs},
            **(dict(learn_enabled=False,episode_seeds=[seed]) if kind=='pong' else {})))
    for arm in ('low','control'):
        prefix=arm+'_at20000';cap=json.loads((directory/(prefix+'-cap.json')).read_text())
        for i,seed in enumerate(m['schedules']['development']):
            add(f'b_{arm}_endpoint_s{i:02d}','pendulum',arm+'_endpoint',i,seed,prefix+'-vision.pkl',prefix+'-weights.npy',PendulumEvaluationConfig(seed,policy_scaling=cap))
    cap=json.loads((directory/'initial-cap.json').read_text())
    for i,seed in enumerate(m['schedules']['development']):
        add(f'b_start_s{i:02d}','pendulum','start',i,seed,'initial-vision.pkl','initial-weights.npy',PendulumEvaluationConfig(seed,policy_scaling=cap))
    for kind in metrics.CONTROLS:
        for i,seed in enumerate(m['schedules']['development']):
            c=PendulumEvaluationConfig(seed,policy_kind='random' if kind.startswith('random') else kind,
                skip_baseline_perception=True,action_seed=m['schedules']['control_actions'][i] if kind.startswith('random') else None,
                action_hold_frames=10 if kind=='random10' else 1)
            add(f'b_control_{kind}_s{i:02d}','pendulum',kind,i,seed,'generic-vision.pkl',None,c)
    cap=json.loads((directory/'a-cap.json').read_text())
    for i,seed in enumerate(m['a_reused_development_seeds']):
        add(f'a_low_s{i:02d}','pong','low_a_after',i,seed,'low_at20000-vision.pkl','a-weights.npy',
            PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap))
    for dose in (5000,10000):
        for arm in ('low','control'):
            prefix=f'{arm}_at{dose}';cap=json.loads((directory/(prefix+'-cap.json')).read_text())
            for i,seed in enumerate(m['schedules']['development'][:32]):
                add(f'b_{prefix}_s{i:02d}','pendulum',prefix,i,seed,prefix+'-vision.pkl',prefix+'-weights.npy',PendulumEvaluationConfig(seed,policy_scaling=cap))
    return jobs


def survival_evidence(groups):
    names=('low_endpoint','control_endpoint',*metrics.CONTROLS)
    original={**groups,'experienced_endpoint':groups.get('low_endpoint',[]),'fresh_endpoint':groups.get('control_endpoint',[])}
    result=metrics.survival_evidence(original,BOOTSTRAP)
    if not result['complete']:return result
    result['arms']={a:result['arms'][key] for a,key in [('low','experienced_endpoint'),('control','fresh_endpoint')]}
    result['low_minus_control_mean_seconds']=result.pop('experienced_minus_fresh');result.pop('fixed_midpoint_curves_descriptive')
    reference=[r['seed'] for r in groups[names[0]]];index=np.random.default_rng(BOOTSTRAP).integers(64,size=(10000,64))
    if (len(groups.get('start',[]))!=64 or [r['seed'] for r in groups['start']]!=reference
            or [r['serve_index'] for r in groups['start']]!=list(range(64))):
        result.update(complete=False);return result
    values=lambda name,key:np.array([r[key] for r in groups[name]],dtype=float)
    result['low_minus_control_cap_fraction']=metrics.contrast(values('low_endpoint','time_limit'),values('control_endpoint','time_limit'),index)
    result['incremental_vs_start']={a:dict(mean_seconds=metrics.contrast(values(a+'_endpoint','survival_seconds'),values('start','survival_seconds'),index),
        cap_fraction=metrics.contrast(values(a+'_endpoint','time_limit'),values('start','time_limit'),index)) for a in ('low','control')}
    result['lower_exploration_development_signal']=bool(result['low_minus_control_mean_seconds']['ci95'][0]>0
        and result['low_minus_control_cap_fraction']['ci95'][0]>0)
    curves={}
    for arm in ('low','control'):
        curves[arm]={}
        for label,data in [('start20k',groups['start'][:32]),('25k',groups.get(arm+'_at5000',[])),
                           ('30k',groups.get(arm+'_at10000',[])),('40k',groups[arm+'_endpoint'][:32])]:
            if len(data)!=32:result.update(complete=False);continue
            if [r['serve_index'] for r in data]!=list(range(32)) or [r['seed'] for r in data]!=reference[:32]:
                raise ValueError('fixed curve cohort order changed')
            curves[arm][label]=dict(mean_survival_seconds=float(np.mean([r['survival_seconds'] for r in data])),
                time_limit_fraction=float(np.mean([r['time_limit'] for r in data])))
    result['fixed32_curves_descriptive']=curves
    result['scope']='One counterfactual training root, reused audited .2 training; later states/RNG draws need not remain matched'
    return result


def outcome_evidence(directory,m):
    return base.outcome_evidence(directory,m)


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('summary lacks authoritative completion')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained epsilon-study evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained summary changed')
    return saved['summary']


def supervise(directory):
    if DRAFT_ONLY:raise ValueError('draft only: preceding allocation and audit lock still active')
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=validate_manifest(directory)
        retained=retained_outcome(directory,m)
        if retained is not None:return retained
        if 'started_wall_ns' not in m:m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        train=training_jobs(m);base.register_jobs(directory,m,train)
        tb=base.execute_phase(directory,m,'training',train,1,phase_seconds=TRAIN_SECONDS,total_seconds=TOTAL_SECONDS)
        eb=None;groups={};after=[];annotations={}
        if tb['complete'] and tb['budget_met'] and not tb['failed']:
            jobs=evaluation_jobs(directory,m);base.register_jobs(directory,m,jobs)
            eb=base.execute_phase(directory,m,'evaluation',jobs,4,phase_seconds=EVAL_SECONDS,total_seconds=TOTAL_SECONDS)
            for j in sorted(jobs,key=lambda j:(j['group'],j['serve_index'])):
                report=worker.complete_report(directory,j)
                if j['kind']=='pong':
                    a=pong_metrics.annotate_job(directory,j);annotations[j['id']]=a
                    if report is not None:after.extend(a['rallies'])
                elif report is not None:groups.setdefault(j['group'],[]).append(dict(report['session'],seed=j['config']['seed'],serve_index=j['serve_index']))
        old_annotations=json.loads((OLD/'a-after-annotations.json').read_text())
        control=[old_annotations[f'a_on_s{i:02d}']['rallies'][0] for i in range(64)]
        ancestral=json.loads((prior.OLD/'a-after-annotations.json').read_text())
        before=[ancestral[f'a_experienced_s{i:02d}']['rallies'][0] for i in range(64)]
        atomic_json(directory/'a-after-annotations.json',annotations)
        survival=survival_evidence(groups)
        retention={a:metrics.retention_evidence(before,data,BOOTSTRAP+1) for a,data in [('low',after),('control',control)]}
        costs=base.cost_summary(directory,m)
        costs.update(reused_control_training_transitions=DOSE,reused_control_a_trials=64,
            reused_a_before_trials=64,reuse_adds_new_camera_observations=0,
            reused_a_before='One verified64-serve old E20k reference, shared by both comparisons; zero new work')
        atomic_json(directory/'cost-summary.json',costs)
        evidence=outcome_evidence(directory,m);elapsed=(time.time_ns()-m['started_wall_ns'])/1e9
        eval_elapsed=None if eb is None else (time.time_ns()-eb['started_wall_ns'])/1e9
        valid=bool(tb['complete'] and tb['budget_met'] and not tb['failed'] and eb is not None
            and eb['complete'] and eb['budget_met'] and not eb['failed'] and eval_elapsed<=EVAL_SECONDS
            and elapsed<=TOTAL_SECONDS and survival.get('complete',False) and all(r['complete'] for r in retention.values()))
        summary=dict(status='complete' if valid else 'partial',evidence_eligible=valid,training=tb,evaluation=eb,
            wall_seconds=elapsed,evaluation_scoring_wall_seconds=eval_elapsed,overall_budget_met=elapsed<=TOTAL_SECONDS,
            b_survival=survival,a_retention=retention,costs=costs,
            scope='Epsilon-only genuine reward learning, reused control, one counterfactual training root; no formal A noninferiority',
            final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=evidence))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();stage(a.directory) if a.stage else supervise(a.directory)
