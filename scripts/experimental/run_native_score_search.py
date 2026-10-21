"""Warm-start win-only search at one fixed opponent; no privileged policy input."""
import argparse,copy,fcntl,json,multiprocessing as mp,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_pong_score_transfer as transfer
import run_direct_hit_continuation as continuation
import search_transactions as transactions
from policy_search_math import function_rms,generation_spec,mirrored_update
from return_metrics import aggregate

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-native-score-search-protocol.md'
DRAFT_ONLY=False
TRAINING_SEED=1374860291;OPTIMIZER_SEED=1009827637
DEV_SEED=1826405731;ACTION_SEED=593170827;BOOTSTRAP=1529704139;TAG=29117
TRAIN_CAMERAS=200000;TRAIN_SECONDS=2400.;EVAL_SECONDS=900.;TOTAL_SECONDS=3300.
EVAL_CAMERAS=288000;MAX_UPDATES=12;MIN_UPDATES=3;FIRST_GENERATION=0;INHERITED_UPDATES=9
SERVES=64;TRAIN_SERVES=12;NOISE_RATIO=.50;STEP_RATIO=.10
CONTROLS=('start','seed1_initial','seed2_initial','initial','noop','up','random1','random10')
POLICIES=('endpoint',*CONTROLS)
prior=transfer.prior;engine=prior.engine;completion=prior.completion
sha=prior.sha;atomic_json=prior.atomic_json;array_hash=prior.array_hash
weights_file=prior.weights_file;immutable_array=prior.immutable_array
counts=continuation.counts;training_count=continuation.training_count
logged_failures=continuation.logged_failures;execute=continuation.execute


def relative_scales(theta,z):
    rms=function_rms(theta,z)
    if not np.isfinite(rms) or rms<=0:raise ValueError('positive finite score-free function RMS required')
    return dict(policy_function_rms=rms,noise_scale=NOISE_RATIO*rms,step_size=STEP_RATIO*rms)


def roots():
    return (TRAINING_SEED,OPTIMIZER_SEED,DEV_SEED,ACTION_SEED,BOOTSTRAP)


def schedule(excluded):
    excluded=set(excluded)|set(engine.FINAL)
    if len(set(roots()))!=5 or set(roots())&excluded:raise ValueError('fresh independent root collision')
    used=excluded|set(roots());seeds=[]
    rng=np.random.default_rng(np.random.SeedSequence([DEV_SEED,TAG]))
    while len(seeds)<SERVES:
        value=int(rng.integers(1,2147483647))
        if value not in used:seeds.append(value);used.add(value)
    actions=[int(np.random.SeedSequence([ACTION_SEED,TAG,i]).generate_state(1)[0]) for i in range(SERVES)]
    if len(set(actions))!=SERVES or set(actions)&used:raise ValueError('control action seed collision')
    return dict(serves=seeds,control_actions=actions)


def pinned_sources():
    modules=dict(coordinator=__file__,transfer=transfer.__file__,prior=prior.__file__,engine=engine.__file__,
        completion=completion.__file__,transactions=transactions.__file__,continuation=continuation.__file__,
        physical=prior.physical.__file__,worker=engine.WORKER,math=engine.MATH,
        seeds=Path(__file__).with_name('run_frozen_hit_confirmation.py'),
        metrics=Path(__file__).with_name('return_metrics.py'),protocol=PROTOCOL)
    return {name:sha(path) for name,path in modules.items()}


def validate_calibration(directory):
    digest=sha(engine.CACHE)
    fit=json.loads((directory/'transform-fit.json').read_text())
    descriptor=json.loads((directory/'projection.json').read_text())
    hashes={fit['calibration_source_hash'],descriptor['calibration']['source_sha256']}
    hashes.update(json.loads((lineage/'manifest.json').read_text())['calibration_source_hash']
        for lineage in transfer.LINEAGES.values())
    if hashes!={digest}:raise ValueError('exact inherited score-free calibration provenance required')
    return digest


def stage(directory):
    if DRAFT_ONLY:raise RuntimeError('Draft only: no candidate stream expansion or clock')
    directory.mkdir(parents=True,exist_ok=True);transfer.stage_lineages(directory)
    calibration_hash=validate_calibration(directory)
    lineage=transfer.LINEAGES['seed1'];old=json.loads((lineage/'summary.json').read_text())
    if sum(p['completed_generations'] for p in old['phases'])!=INHERITED_UPDATES:
        raise ValueError('predetermined starting lineage update count differs')
    phase=old['phases'][1]
    immutable_array(directory/'theta_at0.npy',np.load(lineage/phase['phase']/phase['theta_file'],allow_pickle=False))
    theta=np.load(directory/'theta_at0.npy',allow_pickle=False);matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    weights_file(directory,'policy_at0.npy',theta@matrix)
    if sha(directory/'policy_at0.npy')!=sha(directory/'seed1.npy'):raise ValueError('starting acquired policy changed')
    transfer.snapshot_exclusions(directory)
    schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    transfer.immutable_json(directory/'schedules.json',schedules)
    rng=np.random.default_rng(OPTIMIZER_SEED)
    transfer.immutable_json(directory/'boundary-proof.json',dict(endpoint=dict(theta_hash=array_hash(theta),
        next_rng_state=copy.deepcopy(rng.bit_generator.state)),inherited_updates=INHERITED_UPDATES,
        initial_lineage='seed1',initial_theta_file=str(lineage/phase['phase']/phase['theta_file']),
        inherited_training_cameras=sum(p['observed_cameras'] for p in old['phases']),calibration_source_hash=calibration_hash,
        old_partial_fitness_reused=0))


def transaction_initial_boundary(directory):
    return json.loads((directory/'boundary-proof.json').read_text())['endpoint']


def prepare(directory):
    if DRAFT_ONLY:raise RuntimeError('Draft only: no candidate stream expansion or clock')
    proof=json.loads((directory/'lineage-proof.json').read_text())
    if proof['current_source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature():
        raise ValueError('frozen lineage source/runtime changed')
    if any(sha(directory/n)!=h for n,h in proof['fixed_artifacts'].items()):raise ValueError('frozen lineage files changed')
    if any(sha(path)!=h for p in proof['policies'].values() for path,h in p['ancestor_hashes'].items()):
        raise ValueError('ancestor evidence changed')
    schedules=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    if schedules!=json.loads((directory/'schedules.json').read_text()):raise ValueError('frozen development seeds changed')
    calibration_hash=validate_calibration(directory)
    if json.loads((directory/'boundary-proof.json').read_text())['calibration_source_hash']!=calibration_hash:
        raise ValueError('sealed calibration source changed')
    theta0=np.load(directory/'theta_at0.npy',allow_pickle=False);matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    initial=transaction_initial_boundary(directory);rng=np.random.default_rng(OPTIMIZER_SEED)
    if (initial!=dict(theta_hash=array_hash(theta0),next_rng_state=rng.bit_generator.state)
            or not np.array_equal(theta0@matrix,np.load(directory/'seed1.npy',allow_pickle=False))):
        raise ValueError('starting theta/RNG differs')
    checks=json.loads((directory/'prelaunch-checks.json').read_text())
    if not checks['passed'] or checks['source']!=prior.source_signature():raise ValueError('passing current prelaunch checks required')
    if any(sha(path)!=h for path,h in checks.get('evidence_files',{}).items()):raise ValueError('prelaunch proof changed')
    fixed=['lineage-proof.json','boundary-proof.json','seed-exclusions.json','schedules.json','prelaunch-checks.json',
           'theta_at0.npy','policy_at0.npy','policy_at0.json',*proof['fixed_artifacts']]
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,pinned_sources=pinned_sources(),driver_hash=sha(engine.WORKER),
        frozen_file_hashes={n:sha(directory/n) for n in fixed},training_seed=TRAINING_SEED,optimizer_seed=OPTIMIZER_SEED,
        development_seed=DEV_SEED,random_action_seed=ACTION_SEED,bootstrap_seed=BOOTSTRAP,development_tag=TAG,
        calibration_source_hash=calibration_hash,reused_calibration_rows=1024,new_calibration_observations=0,
        serve_seeds=schedules['serves'],action_seeds=schedules['control_actions'],training_camera_cap=TRAIN_CAMERAS,
        training_wall_seconds=TRAIN_SECONDS,evaluation_wall_seconds=EVAL_SECONDS,total_wall_seconds=TOTAL_SECONDS,
        evaluation_camera_cap=EVAL_CAMERAS,maximum_new_updates=MAX_UPDATES,minimum_complete_populations=MIN_UPDATES,
        training_serves=TRAIN_SERVES,noise_ratio=NOISE_RATIO,step_ratio=STEP_RATIO,fitness='terminal positive score indicator',
        directions=8,top_directions=4,opponent_speed=.35,hold=1,max_episode_ticks=1200,
        primary_controls=list(CONTROLS),workers=4,gc_worker_mode=prior.GC_WORKER_MODE,
        anchor_cadence='even local generations; pre-update center and fixed acquired start',
        reserved_final_seeds=list(engine.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered native-score search changed')
    else:manifest=dict(**expected,jobs=[]);atomic_json(path,manifest)
    state_path=directory/'search-state.json'
    if not state_path.exists():atomic_json(state_path,dict(completed_updates=0,next_global_generation=0,
        theta_file='theta_at0.npy',theta_hash=initial['theta_hash'],rng_state=initial['next_rng_state'],pending_diagnostics=None))
    state=json.loads(state_path.read_text());theta=np.load(directory/state['theta_file'],allow_pickle=False)
    if array_hash(theta)!=state['theta_hash']:raise ValueError('optimizer checkpoint differs')
    transactions.committed_fitness_rows(sys.modules[__name__],directory,state)
    z=np.asarray(engine.read_calibration(engine.CACHE)@matrix.T)
    return manifest,z,state,theta


def rollout_job(name,seed,weights,purpose,**kwargs):
    job=prior.rollout_job(name,seed,weights,purpose,**kwargs);job['config']['opponent_speed']=.35
    return job


def prepare_population(directory,z,state,theta,excluded):
    g=state['next_global_generation'];n=state['completed_updates'];label=f'g{g:02d}';past=set()
    for path in directory.glob('g*-definition.json'):
        definition=json.loads(path.read_text())
        if definition['global_generation']<g:past.update(definition['training_serve_seeds'])
    scales=relative_scales(theta,z)
    directions,_,serves,after=generation_spec(theta,state['rng_state'],g,z,training_seed=TRAINING_SEED,
        serves_per_generation=TRAIN_SERVES,excluded_seeds=(*excluded,*past,*roots()))
    candidates=theta[None,None,:,:]+scales['noise_scale']*directions[:,None,:,:]*np.array([1.,-1.])[None,:,None,None]
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False);immutable_array(directory/(label+'-directions.npy'),directions)
    jobs=[]
    for d in range(8):
        for sign in range(2):
            prefix=f'{label}_d{d}_{"plus" if sign==0 else "minus"}';filename=prefix+'.npy'
            weights_file(directory,filename,candidates[d,sign]@matrix)
            for s,seed in enumerate(serves):
                job=rollout_job(f'{prefix}_s{s}',seed,filename,'training_fitness')
                job.update(direction_index=d,sign_index=sign,serve_index=s);jobs.append(job)
    diagnostics=[];groups={};readouts={}
    if g%2==0:
        center=f'{label}_center_before_update{n}.npy';weights_file(directory,center,theta@matrix)
        for role,filename in (('start','seed1.npy'),('current_center',center)):
            if n==0 and role=='current_center':
                if sha(directory/center)!=sha(directory/'seed1.npy'):raise ValueError('initial anchor identity differs')
                groups[role]=groups['start'];continue
            names=[]
            for s,seed in enumerate(serves):
                job=rollout_job(f'{label}_anchor_{role}_s{s}',seed,filename,'training_anchor')
                job.update(anchor_role=role,serve_index=s);diagnostics.append(job);names.append(job['id'])
            groups[role]=names
        readouts=dict(source_local_update=n,source_theta_file=state['theta_file'],source_theta_hash=state['theta_hash'],
            reference_weight_hash=sha(directory/'seed1.npy'),center_weight_file=center,center_weight_hash=sha(directory/center),
            initial_identity_reused=n==0)
    definition=dict(generation=n,global_generation=g,input_theta_hash=array_hash(theta),input_rng_state=state['rng_state'],
        rng_state_after_draw=after,directions_hash=array_hash(directions),training_serve_seeds=serves,relative_scales=scales,
        jobs=jobs,diagnostic_jobs=diagnostics,anchor_groups=groups,anchor_readouts=readouts)
    transfer.immutable_json(directory/(label+'-definition.json'),definition);engine.register_jobs(directory,[*jobs,*diagnostics])
    return definition,directions


def population_fitness(directory,definition):
    values=np.zeros((8,2,TRAIN_SERVES));seen=set();reports=[]
    for job in definition['jobs']:
        key=(job['direction_index'],job['sign_index'],job['serve_index'])
        if key in seen or not (0<=key[0]<8 and 0<=key[1]<2 and 0<=key[2]<TRAIN_SERVES):raise ValueError('invalid fitness cell')
        seen.add(key);r=completion.complete_report(directory,job)
        if r is None or not r['complete'] or r['episodes']!=1:raise ValueError('partial fitness population')
        if r['total_reward'] not in (-1.,0.,1.) or r['wins'] not in (0,1) or r['wins']!=int(r['total_reward']>0):
            raise ValueError('raw score and terminal win indicator differ')
        values[key]=r['wins'];reports.append(dict(job=job['id'],wins=r['wins'],terminal_score=r['total_reward'],
            transitions=r['transitions'],physics_seconds=r['physics_seconds']))
    if len(seen)!=16*TRAIN_SERVES:raise ValueError('incomplete fitness population')
    return values.mean(axis=2),values,reports


def finish_fitness(directory,definition,directions,theta,z):
    scales=relative_scales(theta,z)
    if definition['input_theta_hash']!=array_hash(theta) or definition['relative_scales']!=scales:raise ValueError('fitness input differs')
    returns,wins,reports=population_fitness(directory,definition)
    proposed,optimizer=mirrored_update(theta,directions,returns,z,top=4,step_size=scales['step_size'])
    n=definition['generation']+1;g=definition['global_generation'];immutable_array(directory/f'theta_at{n}.npy',proposed)
    weights_file(directory,f'policy_at{n}.npy',proposed@np.load(directory/'search-transform.npy',allow_pickle=False))
    result=dict(generation=n-1,global_generation=g,returns=returns.tolist(),wins_by_direction_sign_serve=wins.astype(int).tolist(),
        optimizer=optimizer,rollouts=reports,new_theta_hash=array_hash(proposed),relative_scales=scales,
        policy_function_rms=function_rms(proposed,z),fitness='mean terminal win indicator; anchors excluded')
    transfer.immutable_json(directory/f'g{g:02d}-result.json',result)
    state=dict(completed_updates=n,next_global_generation=g+1,theta_file=f'theta_at{n}.npy',theta_hash=array_hash(proposed),
        rng_state=definition['rng_state_after_draw'],pending_diagnostics=f'g{g:02d}' if definition['diagnostic_jobs'] else None)
    atomic_json(directory/'search-state.json',state);return state,proposed


def finish_diagnostics(directory,definition):
    values={}
    for role,names in definition['anchor_groups'].items():
        values[role]=[]
        for name in names:
            job=next(j for j in definition['diagnostic_jobs'] if j['id']==name);r=completion.complete_report(directory,job)
            if r is None:raise ValueError('partial diagnostic cannot complete')
            values[role].append(r['wins'])
    difference=np.asarray(values['current_center'])-np.asarray(values['start'])
    transfer.immutable_json(directory/f"g{definition['global_generation']:02d}-anchor-result.json",
        dict(source=definition['anchor_readouts'],wins=values,paired_win_difference=difference.tolist(),
            mean_win_difference=float(difference.mean()),diagnostic_only=True,used_for_fitness_or_selection=False))


def evaluation_jobs(directory,m,state):
    files=dict(endpoint=f"policy_at{state['completed_updates']}.npy",start='seed1.npy',
        seed1_initial='seed1_initial.npy',seed2_initial='seed2_initial.npy')
    jobs=[]
    for policy in POLICIES:
        for i,seed in enumerate(m['serve_seeds']):
            kind='trained' if policy in files else 'random' if policy.startswith('random') else policy
            job=rollout_job(f'eval_{policy}_s{i:02d}',seed,files.get(policy),'fresh_evaluation',kind=kind,
                action_seed=m['action_seeds'][i] if kind=='random' else None,hold=10 if policy=='random10' else 1)
            job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def evidence(groups,draws=10000):
    if any(len(groups.get(n,[]))!=SERVES for n in POLICIES):return dict(complete=False)
    for i in range(SERVES):
        geometry=[None if (f:=groups[n][i]['first_incoming']) is None else {k:v for k,v in f.items() if k!='success'} for n in POLICIES]
        if any(v!=geometry[0] for v in geometry[1:]):raise ValueError('first incoming geometry differs')
    index=np.random.default_rng(BOOTSTRAP).integers(SERVES,size=(draws,SERVES));points={};samples={}
    for n in POLICIES:
        wins=np.array([r['reward']>0 for r in groups[n]],dtype=float)
        own=np.array([r['reward']>0 and r['own_returns']>0 for r in groups[n]],dtype=float)
        points[n]=dict(total=float(wins.mean()),own_hit=float(own.mean()))
        samples[n]=dict(total=wins[index].mean(1),own_hit=own[index].mean(1))
    out=dict(complete=True,controls=list(CONTROLS),serve_clusters=SERVES,bootstrap_seed=BOOTSTRAP,bootstrap_draws=draws,
        endpoint={},endpoint_minus_start={},qualifies=False)
    passes=True
    for metric in ('total','own_hit'):
        delta=samples['endpoint'][metric]-np.max([samples[c][metric] for c in CONTROLS],axis=0)
        point=points['endpoint'][metric]-max(points[c][metric] for c in CONTROLS);ci=np.quantile(delta,[.025,.975]).tolist()
        out['endpoint'][metric]=dict(lift=point,ci95=ci,strongest_control_rate=max(points[c][metric] for c in CONTROLS))
        passes=passes and point>=.10 and ci[0]>0
        out['endpoint_minus_start'][metric]=dict(lift=points['endpoint'][metric]-points['start'][metric],
            ci95=np.quantile(samples['endpoint'][metric]-samples['start'][metric],[.025,.975]).tolist())
    out['qualifies']=bool(passes);return out


def outcome_evidence(directory,m):
    retained=transfer.outcome_evidence(directory,m)
    retained['search_files']={p.name:sha(p) for pattern in ('search-state.json','training-endpoint.json','g*-definition.json',
        'g*-result.json','g*-anchor-result.json','theta_at*.npy','policy_at*.npy','policy_at*.json',
        'g*_*.npy','g*_*.json','evaluation-phase.json') for p in directory.glob(pattern)}
    return retained


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('summary lacks authoritative completion')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained search evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained search summary changed')
    return saved['summary']


def evaluate(directory,jobs,pool,deadline):
    """Keep failures and the first evaluation freeze durable across restarts."""
    m=json.loads((directory/'manifest.json').read_text());purposes={j['id']:j['purpose'] for j in m['jobs']}
    path=directory/'rollout-costs.jsonl'
    def logged_failure():
        return any(not c['ok'] and purposes[c['job']]=='fresh_evaluation'
            for c in map(json.loads,path.read_text().splitlines())) if path.exists() else False
    failed=logged_failure()
    marker=directory/'rollout-boundary.json'
    if not marker.exists():
        result=execute(directory,[] if failed else jobs,pool,deadline,EVAL_CAMERAS,counts(directory,m)['fresh_evaluation'])
        result['failed']=result['failed'] or failed
        result['complete']=all(completion.complete_report(directory,j) is not None for j in jobs)
        result['unfinished']=[j['id'] for j in jobs if completion.complete_report(directory,j) is None]
        atomic_json(marker,dict(evaluation=result,frozen_wall_ns=time.time_ns()))
    result=json.loads(marker.read_text())['evaluation']
    if (result['complete']!=all(completion.complete_report(directory,j) is not None for j in jobs)
            or result['failed']!=logged_failure() or result['committed_cameras']!=counts(directory,m)['fresh_evaluation']):
        raise ValueError('frozen evaluation boundary changed')
    return result


def supervise(directory):
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m,z,state,theta=prepare(directory)
        old=retained_outcome(directory,m)
        if old is not None:return old
        if 'started_wall_ns' not in m:m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        start=m['started_wall_ns']/1e9
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
            frozen,state,theta=transactions.train(sys.modules[__name__],directory,pool,m,z,state,theta)
            jobs=evaluation_jobs(directory,m,state) if frozen['valid'] else [];engine.register_jobs(directory,jobs)
            phase=directory/'evaluation-phase.json'
            transfer.immutable_json(phase,dict(started_wall_ns=frozen['frozen_wall_ns']))
            eval_start=json.loads(phase.read_text())['started_wall_ns']/1e9
            evaluated=evaluate(directory,jobs,pool,min(start+TOTAL_SECONDS,eval_start+EVAL_SECONDS)-120.)
        annotations=prior.physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        groups={};by_policy={}
        for policy in POLICIES:
            ordered=sorted((j for j in jobs if j['policy']==policy),key=lambda j:j['serve_index'])
            if ordered and [j['serve_index'] for j in ordered]!=list(range(SERVES)):raise ValueError('cohort ordering differs')
            rows=[r for j in ordered if j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rows;groups[policy]=dict(complete=len(rows)==SERVES,**aggregate(rows))
            groups[policy].update(transfer.score_counts(rows))
        paired=evidence(by_policy);actual=counts(directory,json.loads((directory/'manifest.json').read_text()))
        current_manifest=json.loads((directory/'manifest.json').read_text())
        reports=[(j,engine.read_report(directory,j['id'])) for j in current_manifest['jobs']]
        logical=dict(initial_images=sum(r.get('initial_cameras',0) for _,r in reports if r),
            transition_images=sum(actual.values()),
            encoded_images=sum(r.get('initial_cameras',0)+r['transitions'] for j,r in reports
                if r and not j['config']['skip_baseline_perception']),new_visual_learning_images=0,
            annotation_physics_reexecutions=sum(a['verified_camera_rows'] for a in annotations.values()),
            partial_receipt_caveat='Initial/encoded counts use retained receipts; journal transition counts are authoritative')
        retained=outcome_evidence(directory,current_manifest)
        elapsed=time.time()-start;eval_elapsed=time.time()-eval_start
        train_elapsed=frozen['frozen_wall_ns']/1e9-start;train_within=train_elapsed<=TRAIN_SECONDS
        within=elapsed<=TOTAL_SECONDS and eval_elapsed<=EVAL_SECONDS and train_within
        valid=bool(frozen['valid'] and not frozen['anchor_failure'] and evaluated['complete'] and not evaluated['failed']
            and within and paired.get('complete',False))
        summary=dict(status='complete' if valid else 'partial',evidence_eligible=valid,counts=actual,logical_compute=logical,training_endpoint=frozen,
            evaluation=evaluated,groups=groups,paired_score_evidence=paired,qualifies=valid and paired.get('qualifies',False),
            wall_seconds=elapsed,training_wall_seconds=train_elapsed,training_budget_met=train_within,
            evaluation_wall_seconds=eval_elapsed,evaluation_budget_met=eval_elapsed<=EVAL_SECONDS,
            overall_budget_met=elapsed<=TOTAL_SECONDS,
            selected_checkpoint=f"policy_at{state['completed_updates']}.npy",selection='last complete population; anchors never select',
            scope='One warm-start native-score adaptation; no independent replication or completed M1B',final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=retained))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();stage(a.directory) if a.stage else supervise(a.directory)
