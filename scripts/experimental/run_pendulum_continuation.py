"""Restarted-episode native survival learning with visual plasticity ON/OFF."""
import argparse,copy,fcntl,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.vision_transfer import bridge_snapshot
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.control.pendulum_evaluation import PendulumEvaluationConfig
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
import run_pendulum_acquisition as base
import run_pong_score_transfer as seeds
import pendulum_acquisition_worker as worker
import pendulum_acquisition_metrics as metrics
import online_vision_metrics as pong_metrics

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-pendulum-continuation-protocol.md'
EXPERIMENTS=Path('/workspace/m1b-experiments')
OLD=EXPERIMENTS/'pendulum-online-acquisition-20261021-r1'
AUDIT=EXPERIMENTS/'pendulum-parent-audit-20261021'
PREFLIGHT=EXPERIMENTS/'pendulum-onoff-preflight-20261021'
ARMS=('on','off');DOSE=20000;CHECKPOINTS=(5000,10000,20000)
ROOTS=dict(environment=1083294751,learner=1647809237,development=702936143,control_actions=1563079821)
TAG=29116;BOOTSTRAP=1142938759
TRAIN_SECONDS=1200.;EVAL_SECONDS=1800.;TOTAL_SECONDS=3000.
sha=worker.sha;atomic_json=base.atomic_json


def immutable(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('registered continuation changed')
    else:atomic_json(path,value)


def pinned_sources():
    names=('run_pendulum_continuation.py','run_pendulum_acquisition.py','pendulum_acquisition_worker.py',
           'pendulum_acquisition_metrics.py','persistent_vision_worker.py','online_vision_metrics.py',
           'run_pong_score_transfer.py','return_metrics.py','run_pong_development.py',
           'run_pong_policy_search_continuation.py','run_direct_hit_policy_search.py')
    return {**{'scripts/experimental/'+n:sha(ROOT/'scripts/experimental'/n) for n in names},
            str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}


def schedule(excluded):
    excluded=set(excluded)|set(base.FINAL_SEEDS)
    if (set(ROOTS.values())|{BOOTSTRAP})&excluded:raise ValueError('new continuation root collision')
    used=excluded|set(ROOTS.values())|{BOOTSTRAP};values=[]
    rng=np.random.default_rng(np.random.SeedSequence([ROOTS['development'],TAG]))
    while len(values)<64:
        v=int(rng.integers(1,2147483647))
        if v not in used:values.append(v);used.add(v)
    actions=[int(np.random.SeedSequence([ROOTS['control_actions'],TAG,i]).generate_state(1)[0]) for i in range(64)]
    if len(set(actions))!=64 or set(actions)&used:raise ValueError('control action seed collision')
    return dict(development=values,control_actions=actions)


def stage(directory):
    directory.mkdir(parents=True,exist_ok=True);seeds.snapshot_exclusions(directory)
    old=json.loads((OLD/'manifest.json').read_text());source=source_signature()
    checks=json.loads((directory/'prelaunch-checks.json').read_text())
    if (not checks['passed'] or checks['source']!=source
            or any(sha(p)!=h for p,h in checks['evidence_files'].items())):
        raise ValueError('passing current continuation prelaunch evidence required')
    if not all(json.loads(p.read_text())['passed'] for p in
               (OLD/'completion-verification.json',AUDIT/'summary.json',PREFLIGHT/'report.json')):
        raise ValueError('verified original boundary and actual-memory runtime required')
    if json.loads((PREFLIGHT/'report.json').read_text())['source']!=source:
        raise ValueError('actual-memory preflight source differs')
    if json.loads((OLD/'summary.json').read_text())['status']!='complete':raise ValueError('completed original dose required')
    verified=json.loads((OLD/'completion-verification.json').read_text())['endpoint_artifact_hashes']
    for name,digest in verified.items():
        if name.startswith('experienced_at20000') and sha(OLD/name)!=digest:raise ValueError('verified starting boundary artifact changed')
    delta={n:dict(before=old['source'].get(n),after=source.get(n)) for n in sorted(set(old['source'])|set(source))
           if old['source'].get(n)!=source.get(n)}
    if set(delta)!={'control/codec.py','control/pendulum_session.py'}:
        raise ValueError('only the reviewed codec and optional task-session source bridge is allowed')
    fixed={'prelaunch-checks.json':sha(directory/'prelaunch-checks.json')};starts={};bridges={}
    for label,filename in [('initial','experienced_at20000-vision.pkl'),('generic','fresh-before.pkl')]:
        expected_hash=verified[filename] if label=='initial' else old['fixed_artifacts'][filename]
        snapshot,receipt=bridge_snapshot(OLD/filename,expected_file_sha256=expected_hash,
            ancestor_source=old['source'],reviewed_control_changes=delta)
        saved=worker.save_snapshot(directory/(label+'-vision.pkl'),snapshot)
        starts[label]=saved;fixed[saved['file']]=saved['file_sha256'];bridges[label]=receipt
        immutable(directory/(label+'-bridge.json'),receipt)
        fixed[label+'-bridge.json']=sha(directory/(label+'-bridge.json'))
    if starts['initial']['cumulative_learning_observations']!=22282:raise ValueError('predetermined experienced20k boundary required')
    copies={'initial-theta.npy':'experienced_at20000-theta.npy','initial-weights.npy':'experienced_at20000-weights.npy',
            'initial-cap.json':'experienced_at20000-cap.json','search-transform.npy':'search-transform.npy',
            'transform-fit.json':'transform-fit.json','a-weights.npy':'seed1-policy_endpoint.npy','a-cap.json':'seed1-radial-cap.json'}
    for target,origin in copies.items():
        expected_hash=verified.get(origin,old['fixed_artifacts'].get(origin))
        if expected_hash is None or sha(OLD/origin)!=expected_hash:raise ValueError('original fixed input changed')
        base.immutable_copy(OLD/origin,directory/target);fixed[target]=sha(directory/target)
    projection=copy.deepcopy(old['projection']);projection['matrix_file']=str(directory/'search-transform.npy')
    declared=FixedPolicyProjection(4389,projection);theta=np.load(directory/'initial-theta.npy',allow_pickle=False)
    if not np.array_equal(declared.mapped_weights(theta),np.load(directory/'initial-weights.npy',allow_pickle=False)):
        raise ValueError('starting theta/raw-head mapping differs')
    ancestors=[OLD/n for n in ('manifest.json','summary.json','study-completion.json','completion-verification.json',
        'a-after-annotations.json','experienced_at20000-export.json')]+[AUDIT/'summary.json',PREFLIGHT/'manifest.json',PREFLIGHT/'report.json']
    expected=dict(version='new-episode-survival-continuation-onoff-v1',source=source,runtime=runtime_signature(),
        model_hash=MODEL_HASH,vision_configuration=VISION_CONFIGURATION,pinned_sources=pinned_sources(),
        seed_roots=ROOTS,seed_tag=TAG,bootstrap_seed=BOOTSTRAP,bootstrap_draws=10000,
        seed_exclusions_sha256=sha(directory/'seed-exclusions.json'),
        schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids']),
        fixed_artifacts=fixed,starts=starts,source_bridges=bridges,projection=projection,
        ancestor_evidence={str(p):sha(p) for p in ancestors},a_reused_development_seeds=old['a_reused_development_seeds'],
        transitions_per_arm=DOSE,checkpoints=list(CHECKPOINTS),training_seconds=TRAIN_SECONDS,evaluation_seconds=EVAL_SECONDS,
        total_seconds=TOTAL_SECONDS,prior_b_policy_updates=20000,inherited_visual_learning_images=22282,
        initial_state='New episode/exploration streams and zero traces/q_old; exact learned projected weights and owned memory',
        off_a_reuse_rule='Exact canonical snapshot content/count plus unchanged A weights/config/source proof; otherwise64 new jobs',
        final_streams_expanded=False,reserved_final_ids=list(base.FINAL_SEEDS))
    p=directory/'manifest.json'
    if p.exists():
        m=json.loads(p.read_text())
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('staged continuation differs')
    else:atomic_json(p,dict(**expected,jobs=[]))


def validate_manifest(directory):
    m=json.loads((directory/'manifest.json').read_text())
    if (m['source']!=source_signature() or m['runtime']!=runtime_signature() or m['pinned_sources']!=pinned_sources()
            or m['seed_roots']!=ROOTS or sha(directory/'seed-exclusions.json')!=m['seed_exclusions_sha256']
            or m['schedules']!=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])):
        raise ValueError('continuation source/runtime/seed contract changed')
    for names in (m['fixed_artifacts'],m['ancestor_evidence']):
        for name,digest in names.items():
            if sha(Path(name) if Path(name).is_absolute() else directory/name)!=digest:raise ValueError('pinned continuation input/evidence changed')
    return m


def training_jobs(m):
    refs=('initial-vision.pkl','initial-theta.npy','search-transform.npy')
    return [dict(id=arm+'_train',arm=arm,kind='train',phase='training',camera_dose=DOSE,checkpoints=list(CHECKPOINTS),
        snapshot_from='initial-vision.pkl',initial_weights_from='initial-theta.npy',learn_enabled=arm=='on',
        input_hashes={n:m['fixed_artifacts'][n] for n in refs},
        config=dict(seed=ROOTS['environment'],learner_seed=ROOTS['learner'],max_episode_ticks=600,
                    alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95)) for arm in ARMS]


def off_a_reuse(directory,m):
    exported=worker.load_snapshot(directory,'off_at20000-vision.pkl');initial=worker.load_snapshot(directory,'initial-vision.pkl')
    content_same=exported.content_hash==initial.content_hash and exported.learning_observations==initial.learning_observations
    same=content_same
    raw={};old_m=json.loads((OLD/'manifest.json').read_text())
    cap=json.loads((directory/'a-cap.json').read_text())
    ancestor_snapshot=m['source_bridges']['initial']['ancestor_file_sha256']
    if same:
        old_jobs={j['id']:j for j in old_m['jobs']}
        for i,seed in enumerate(m['a_reused_development_seeds']):
            name=f'a_experienced_s{i:02d}';j=old_jobs[name]
            pin=json.loads((OLD/(name+'-completion.json')).read_text())
            report=json.loads((OLD/(name+'.json')).read_text())
            expected=asdict(PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap))
            if (j['kind']!='pong' or j['config']!=expected or j.get('learn_enabled') is not False
                    or j.get('episode_seeds')!=[seed]
                    or j['input_hashes'].get(j['snapshot_from'])!=ancestor_snapshot
                    or sha(OLD/j['snapshot_from'])!=ancestor_snapshot):
                same=False;raw={};break
            if (not report['complete'] or sha(OLD/(name+'.json'))!=pin['report_sha256']
                    or worker.journal_hash(OLD/(name+'.db'))!=pin['journal_hash']
                    or sha(OLD/j['weights_from'])!=m['fixed_artifacts']['a-weights.npy']):
                raise ValueError('old OFF A reference no longer matches verified evidence')
            raw[name]=dict(journal_hash=pin['journal_hash'],report_sha256=pin['report_sha256'],
                           completion_sha256=sha(OLD/(name+'-completion.json')),config=j['config'])
    proof=dict(reuse=same,full_reset_vision_content_equal=content_same,initial_content_hash=initial.content_hash,
        off_content_hash=exported.content_hash,learning_observations=exported.learning_observations,
        source_bridge=m['source_bridges']['initial'],a_weights_sha256=m['fixed_artifacts']['a-weights.npy'],
        old_annotations_sha256=sha(OLD/'a-after-annotations.json'),old_jobs=raw,
        reused_trials=64 if same else 0,new_camera_observations_from_reuse=0)
    immutable(directory/'a-off-reuse-proof.json',proof);return proof


def evaluation_jobs(directory,m,reuse):
    jobs=[]
    def add(name,kind,group,i,seed,snapshot,weights,config):
        refs=[snapshot]+([] if weights is None else [weights])
        jobs.append(dict(id=name,kind=kind,phase='evaluation',group=group,serve_index=i,config=asdict(config),
            snapshot_from=snapshot,weights_from=weights,input_hashes={n:sha(directory/n) for n in refs},
            **(dict(learn_enabled=False,episode_seeds=[seed]) if kind=='pong' else {})))
    for arm in ARMS:
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
    for arm in ARMS:
        if arm=='off' and reuse['reuse']:continue
        for i,seed in enumerate(m['a_reused_development_seeds']):
            add(f'a_{arm}_s{i:02d}','pong',arm+'_a_after',i,seed,arm+'_at20000-vision.pkl','a-weights.npy',
                PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap))
    for dose in (5000,10000):
        for arm in ARMS:
            prefix=f'{arm}_at{dose}';cap=json.loads((directory/(prefix+'-cap.json')).read_text())
            for i,seed in enumerate(m['schedules']['development'][:32]):
                add(f'b_{prefix}_s{i:02d}','pendulum',prefix,i,seed,prefix+'-vision.pkl',prefix+'-weights.npy',PendulumEvaluationConfig(seed,policy_scaling=cap))
    return jobs


def survival_evidence(groups):
    # Reuse the unchanged original competence calculation with explicit aliases.
    alias={**groups,'experienced_endpoint':groups.get('on_endpoint',[]),'fresh_endpoint':groups.get('off_endpoint',[])}
    result=metrics.survival_evidence(alias,BOOTSTRAP)
    if not result['complete']:return result
    result['arms']={a:result['arms'][key] for a,key in [('on','experienced_endpoint'),('off','fresh_endpoint')]}
    result['on_minus_off']=result.pop('experienced_minus_fresh');result.pop('fixed_midpoint_curves_descriptive')
    index=np.random.default_rng(BOOTSTRAP).integers(64,size=(10000,64));result['incremental_vs_start']={};curves={}
    if len(groups.get('start',[]))!=64:
        result.update(complete=False,incremental_complete=False);return result
    reference=[r['seed'] for r in groups['on_endpoint']]
    if [r['seed'] for r in groups['start']]!=reference or [r['serve_index'] for r in groups['start']]!=list(range(64)):
        raise ValueError('incremental starting-head cohort order differs')
    start=np.array([r['survival_seconds'] for r in groups['start']]);start_cap=np.array([r['time_limit'] for r in groups['start']],dtype=float)
    result['on_minus_off_cap_fraction']=metrics.contrast(
        np.array([r['time_limit'] for r in groups['on_endpoint']],dtype=float),
        np.array([r['time_limit'] for r in groups['off_endpoint']],dtype=float),index)
    for arm in ARMS:
        end=groups[arm+'_endpoint'];values=np.array([r['survival_seconds'] for r in end]);cap=np.array([r['time_limit'] for r in end],dtype=float)
        result['incremental_vs_start'][arm]=dict(mean_seconds=metrics.contrast(values,start,index),cap_fraction=metrics.contrast(cap,start_cap,index))
        curves[arm]={}
        for label,data in [('start20k',groups['start'][:32]),('25k',groups.get(arm+'_at5000',[])),
                           ('30k',groups.get(arm+'_at10000',[])),('40k',end[:32])]:
            if len(data)==32:
                if [r['serve_index'] for r in data]!=list(range(32)) or [r['seed'] for r in data]!=reference[:32]:
                    raise ValueError('fixed-curve cohort order differs')
                curves[arm][label]=dict(mean_survival_seconds=float(np.mean([r['survival_seconds'] for r in data])),
                    time_limit_fraction=float(np.mean([r['time_limit'] for r in data])))
    result['fixed32_curves_descriptive']=curves
    result['scope']='One paired restarted training root; checkpoint-specific memory; frozen competence distinct from deployed plastic control'
    return result


def outcome_evidence(directory,m):
    evidence=base.outcome_evidence(directory,m)
    if (directory/'a-off-reuse-proof.json').exists():evidence['files']['a-off-reuse-proof.json']=sha(directory/'a-off-reuse-proof.json')
    return evidence


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('summary lacks authoritative outcome')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained continuation evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained continuation summary changed')
    return saved['summary']


def supervise(directory):
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=validate_manifest(directory)
        old=retained_outcome(directory,m)
        if old is not None:return old
        if 'started_wall_ns' not in m:m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        train=training_jobs(m);base.register_jobs(directory,m,train)
        tb=base.execute_phase(directory,m,'training',train,2,phase_seconds=TRAIN_SECONDS,total_seconds=TOTAL_SECONDS)
        eb=None;groups={};a_groups={};annotations={};reuse=None
        if tb['complete'] and tb['budget_met'] and not tb['failed']:
            reuse=off_a_reuse(directory,m);jobs=evaluation_jobs(directory,m,reuse);base.register_jobs(directory,m,jobs)
            eb=base.execute_phase(directory,m,'evaluation',jobs,4,phase_seconds=EVAL_SECONDS,total_seconds=TOTAL_SECONDS)
            for j in sorted(jobs,key=lambda j:(j['group'],j['serve_index'])):
                r=worker.complete_report(directory,j)
                if j['kind']=='pong':
                    a=pong_metrics.annotate_job(directory,j);annotations[j['id']]=a
                    if r is not None:a_groups.setdefault(j['group'],[]).extend(a['rallies'])
                elif r is not None:groups.setdefault(j['group'],[]).append(dict(r['session'],seed=j['config']['seed'],serve_index=j['serve_index']))
        old_annotations=json.loads((OLD/'a-after-annotations.json').read_text())
        before=[old_annotations[f'a_experienced_s{i:02d}']['rallies'][0] for i in range(64)]
        if reuse and reuse['reuse']:a_groups['off_a_after']=copy.deepcopy(before)
        atomic_json(directory/'a-after-annotations.json',annotations)
        survival=survival_evidence(groups)
        retention={arm:metrics.retention_evidence(before,a_groups.get(arm+'_a_after',[]),BOOTSTRAP+1) for arm in ARMS}
        costs=base.cost_summary(directory,m);costs['reused_a_before']='One verified64-serve old E20k reference, shared by both arm comparisons; zero new work'
        costs['reused_off_a_trials']=64 if reuse and reuse['reuse'] else 0
        costs['new_training_learning_images']={j['arm']:(worker.complete_report(directory,j) or {}).get('session',{}).get('visual_learning_observations') for j in train}
        atomic_json(directory/'cost-summary.json',costs)
        retained=outcome_evidence(directory,m);elapsed=(time.time_ns()-m['started_wall_ns'])/1e9
        eval_elapsed=None if eb is None else (time.time_ns()-eb['started_wall_ns'])/1e9
        valid=bool(tb['complete'] and tb['budget_met'] and not tb['failed'] and eb is not None and eb['complete']
                   and eb['budget_met'] and not eb['failed'] and eval_elapsed<=EVAL_SECONDS and elapsed<=TOTAL_SECONDS
                   and survival.get('complete',False) and all(x.get('complete',False) for x in retention.values()))
        summary=dict(status='complete' if valid else 'partial',evidence_eligible=valid,training=tb,evaluation=eb,
            wall_seconds=elapsed,evaluation_scoring_wall_seconds=eval_elapsed,overall_budget_met=elapsed<=TOTAL_SECONDS,
            b_survival=survival,a_retention=retention,costs=costs,off_a_reuse=reuse,
            scope='New-episode native-survival learning ON/OFF; no formal A noninferiority or final learning-enabled competence proof',
            final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=retained))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();stage(a.directory) if a.stage else supervise(a.directory)
