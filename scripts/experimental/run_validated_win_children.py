"""Prospectively bounded, fresh-training-validated native-win child selection."""
import argparse,copy,fcntl,json,multiprocessing as mp,shutil,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_native_score_search as native
import validated_child_transactions as transaction
from policy_search_math import sample_directions,function_rms
from return_metrics import aggregate

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-validated-win-children-protocol.md'
ANCESTOR=Path('/workspace/m1b-experiments/pong-native-score-search-20261021-r1')
AUDIT=Path('/workspace/m1b-experiments/native-score-parent-audit-20261021')
NATIVE_SOURCE_ARCHIVE_SHA='a229c908f43bc82d530ad172d6eb6405a6209abcb088cd559d457cf48965979e'
DRAFT_ONLY=False
ROOTS=dict(training=1867214939,optimizer=735190321,development=1246709827,
           control_actions=981437591,bootstrap=1630297841)
TAG=29119
TRAIN_CAMERAS=160000;TRAIN_SECONDS=1800.;EVAL_SECONDS=1200.;TOTAL_SECONDS=3000.
EVAL_CAMERAS=320000;MAX_CYCLES=24;MIN_CYCLES=6
DISCOVERY_SERVES=12;VALIDATION_SERVES=24;DIRECTIONS=2;NOISE_RATIO=.50;MINIMUM_NET_WINS=2;SERVES=64
CONTROLS=('start','original','seed1_initial','seed2_initial','initial','noop','up','random1','random10')
POLICIES=('endpoint',*CONTROLS)
engine=native.engine;completion=native.completion;transfer=native.transfer;prior=native.prior
sha=native.sha;atomic_json=native.atomic_json;array_hash=native.array_hash
immutable_array=native.immutable_array;weights_file=native.weights_file;immutable_json=transfer.immutable_json
counts=native.counts;training_count=native.training_count;execute=native.execute


def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('frozen copied input changed')
    else:shutil.copyfile(source,target)


def root_values():
    if ROOTS is None or set(ROOTS)!={'training','optimizer','development','control_actions','bootstrap'}:
        raise ValueError('prospective independent roots are not registered')
    return set(ROOTS.values())


def schedule(excluded):
    roots=root_values();excluded=set(excluded)|set(engine.FINAL)
    if len(roots)!=5 or roots&excluded:raise ValueError('fresh child-study root collision')
    used=excluded|roots;serves=[];rng=np.random.default_rng(np.random.SeedSequence([ROOTS['development'],TAG]))
    while len(serves)<SERVES:
        seed=int(rng.integers(1,2147483647))
        if seed not in used:serves.append(seed);used.add(seed)
    actions=[int(np.random.SeedSequence([ROOTS['control_actions'],TAG,i]).generate_state(1)[0]) for i in range(SERVES)]
    if len(set(actions))!=SERVES or set(actions)&used:raise ValueError('random action seed collision')
    return dict(serves=serves,control_actions=actions)


def pinned_sources():
    paths=dict(coordinator=__file__,transactions=transaction.__file__,native=native.__file__,
        transfer=transfer.__file__,prior=prior.__file__,engine=engine.__file__,completion=completion.__file__,
        physical=prior.physical.__file__,worker=engine.WORKER,math=engine.MATH,
        continuation=native.continuation.__file__,native_transactions=native.transactions.__file__,
        seeds=Path(__file__).with_name('run_frozen_hit_confirmation.py'),
        metrics=Path(__file__).with_name('return_metrics.py'),protocol=PROTOCOL)
    return {name:sha(path) for name,path in paths.items()}


def stage(directory):
    if DRAFT_ONLY:raise ValueError('draft only: no roots or candidate allocation')
    directory.mkdir(parents=True,exist_ok=True)
    old=json.loads((ANCESTOR/'manifest.json').read_text());summary=json.loads((ANCESTOR/'summary.json').read_text())
    own=json.loads((ANCESTOR/'completion-verification.json').read_text());parent=json.loads((AUDIT/'summary.json').read_text())
    seal=json.loads((ANCESTOR/'source-seal.json').read_text())
    retained=json.loads((ANCESTOR/'study-completion.json').read_text())['evidence']
    archive=json.loads((ANCESTOR/'journal-archive-report.json').read_text())
    if (summary['status']!='complete' or not own['passed'] or not parent['passed'] or not archive['passed']
            or not archive['all_original_bytes_reconstruct_verified'] or old['source']!=prior.source_signature()
            or old['runtime']!=prior.runtime_signature() or old['pinned_sources']!=native.pinned_sources()
            or own['total_committed_rows']!=parent['raw_physics_rows']
            or retained['files']['manifest.json']!=sha(ANCESTOR/'manifest.json')):
        raise ValueError('verified unchanged native ancestor and exact cold storage required')
    if (seal['source_commit']!=own['source_commit'] or seal['source']!=old['source']
            or seal['source_archive_sha256']!=NATIVE_SOURCE_ARCHIVE_SHA
            or sha(ANCESTOR/'source.tar.gz')!=NATIVE_SOURCE_ARCHIVE_SHA):
        raise ValueError('audited native source archive bytes changed')
    frozen=json.loads((ANCESTOR/'training-endpoint.json').read_text())
    if frozen['new_updates']!=5 or frozen['theta_file']!='theta_at5.npy':raise ValueError('predetermined theta5 boundary required')
    copies={'theta_at0.npy':'theta_at5.npy','start.npy':'policy_at5.npy','original.npy':'seed1.npy',
        'seed1_initial.npy':'seed1_initial.npy','seed2_initial.npy':'seed2_initial.npy',
        'search-transform.npy':'search-transform.npy','transform-fit.json':'transform-fit.json',
        'radial-cap.json':'radial-cap.json','projection.json':'projection.json'}
    fixed={}
    for target,origin in copies.items():
        expected=retained['search_files'].get(origin,old['frozen_file_hashes'].get(origin))
        if expected is None or sha(ANCESTOR/origin)!=expected:raise ValueError('audited starting readout/transform changed')
        immutable_copy(ANCESTOR/origin,directory/target);fixed[target]=sha(directory/target)
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False);theta=np.load(directory/'theta_at0.npy',allow_pickle=False)
    if array_hash(theta)!=frozen['theta_hash'] or not np.array_equal(theta@matrix,np.load(directory/'start.npy',allow_pickle=False)):
        raise ValueError('theta5 must map exactly to the starting executed head')
    for name in ('start.npy','original.npy','seed1_initial.npy','seed2_initial.npy'):
        weights_file(directory,name,np.load(directory/name,allow_pickle=False));fixed[Path(name).with_suffix('.json').name]=sha((directory/name).with_suffix('.json'))
    calibration=sha(engine.CACHE)
    if calibration!=old['calibration_source_hash'] or calibration!=json.loads((directory/'transform-fit.json').read_text())['calibration_source_hash']:
        raise ValueError('unchanged score-free calibration source required')
    ancestors=[ANCESTOR/n for n in ('manifest.json','summary.json','study-completion.json','completion-verification.json',
        'training-endpoint.json','launch.json','source-seal.json','source.tar.gz',
        'journal-archive-manifest.json','journal-archive-report.json')]
    ancestors.append(AUDIT/'summary.json')
    transfer.snapshot_exclusions(directory);serves=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])
    immutable_json(directory/'schedules.json',serves)
    rng=np.random.default_rng(ROOTS['optimizer'])
    immutable_json(directory/'boundary-proof.json',dict(theta_hash=array_hash(theta),next_rng_state=rng.bit_generator.state,
        native_source_commit=own['source_commit'],inherited_native_training_cameras=199853,
        inherited_return_training_cameras=199676,native_partial_fitness_reused=0))
    immutable_json(directory/'lineage-proof.json',dict(source=prior.source_signature(),runtime=prior.runtime_signature(),
        fixed_artifacts=fixed,ancestor_evidence={str(p):sha(p) for p in ancestors},
        old_native_helpers_byte_identical=True,calibration_source_hash=calibration,
        raw_journals_cold_archived=True,original_bytes_already_independently_audited=True))
    prepare(directory)


def initial_boundary(directory):
    return json.loads((directory/'boundary-proof.json').read_text())


def prepare(directory):
    if DRAFT_ONLY:raise ValueError('draft only: no candidate allocation')
    proof=json.loads((directory/'lineage-proof.json').read_text());checks=json.loads((directory/'prelaunch-checks.json').read_text())
    if (proof['source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature()
            or not checks['passed'] or checks['source']!=prior.source_signature()
            or any(sha(p)!=h for p,h in checks['evidence_files'].items())):
        raise ValueError('current source/runtime/prelaunch proof required')
    for name,digest in proof['fixed_artifacts'].items():
        if sha(directory/name)!=digest:raise ValueError('frozen readout/transform changed')
    if any(sha(p)!=h for p,h in proof['ancestor_evidence'].items()):raise ValueError('ancestor proof changed')
    if sha(engine.CACHE)!=proof['calibration_source_hash']:raise ValueError('calibration source changed')
    seed_path=directory/'seed-exclusions.json';serves=schedule(json.loads(seed_path.read_text())['ids'])
    if serves!=json.loads((directory/'schedules.json').read_text()):raise ValueError('registered fresh cohort changed')
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,pinned_sources=pinned_sources(),driver_hash=sha(engine.WORKER),
        frozen_file_hashes={n:sha(directory/n) for n in ('lineage-proof.json','boundary-proof.json','schedules.json',
            'seed-exclusions.json','prelaunch-checks.json',*proof['fixed_artifacts'])},seed_roots=ROOTS,seed_tag=TAG,
        serve_seeds=serves['serves'],action_seeds=serves['control_actions'],calibration_source_hash=proof['calibration_source_hash'],
        training_camera_cap=TRAIN_CAMERAS,evaluation_camera_cap=EVAL_CAMERAS,training_wall_seconds=TRAIN_SECONDS,
        evaluation_wall_seconds=EVAL_SECONDS,total_wall_seconds=TOTAL_SECONDS,maximum_cycles=MAX_CYCLES,minimum_complete_cycles=MIN_CYCLES,
        directions=DIRECTIONS,discovery_serves=DISCOVERY_SERVES,validation_serves=VALIDATION_SERVES,
        validation_net_wins=MINIMUM_NET_WINS,noise_ratio=NOISE_RATIO,fitness='positive terminal score indicator only',
        opponent_speed=.35,max_episode_ticks=1200,hold=1,workers=4,gc_worker_mode=prior.GC_WORKER_MODE,
        primary_controls=list(CONTROLS),reserved_final_seeds=list(engine.FINAL),new_calibration_observations=0)
    path=directory/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('registered child allocation changed')
    else:m=dict(**expected,jobs=[]);atomic_json(path,m)
    initial=initial_boundary(directory);theta=np.load(directory/'theta_at0.npy',allow_pickle=False)
    if initial['theta_hash']!=array_hash(theta) or initial['next_rng_state']!=np.random.default_rng(ROOTS['optimizer']).bit_generator.state:
        raise ValueError('initial theta/RNG differs')
    state_path=directory/'search-state.json'
    if not state_path.exists():atomic_json(state_path,dict(completed_cycles=0,accepted_updates=0,next_cycle=0,
        theta_file='theta_at0.npy',theta_hash=initial['theta_hash'],rng_state=initial['next_rng_state']))
    state=json.loads(state_path.read_text());transaction.committed_rows(sys.modules[__name__],directory,state)
    theta=np.load(directory/state['theta_file'],allow_pickle=False)
    if array_hash(theta)!=state['theta_hash']:raise ValueError('saved incumbent differs')
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    return m,np.asarray(engine.read_calibration(engine.CACHE)@matrix.T),state,theta


def draw_cycle(theta,rng_state,cycle,z,excluded,training_seed):
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    directions=sample_directions(rng,z,DIRECTIONS);scale=NOISE_RATIO*function_rms(theta,z)
    if not np.isfinite(scale) or scale<=0:raise ValueError('positive finite incumbent function RMS required')
    candidates=np.array([theta+scale*directions[d]*sign for d in range(DIRECTIONS) for sign in (1.,-1.)])
    serving=np.random.default_rng(np.random.SeedSequence([training_seed,872631,cycle]));used=set(excluded)|set(engine.FINAL);seeds=[]
    while len(seeds)<DISCOVERY_SERVES+VALIDATION_SERVES:
        value=int(serving.integers(1,2147483647))
        if value not in used:seeds.append(value);used.add(value)
    return directions,candidates,seeds[:DISCOVERY_SERVES],seeds[DISCOVERY_SERVES:],copy.deepcopy(rng.bit_generator.state),scale


def prepare_cycle(directory,z,state,theta,excluded):
    cycle=state['next_cycle'];prefix=f'c{cycle:02d}';past=set()
    for path in directory.glob('c*-definition.json'):
        old=json.loads(path.read_text())
        if old['cycle']<cycle:past.update((*old['discovery_seeds'],*old['validation_seeds']))
    directions,children,discovery,validation,after,scale=draw_cycle(theta,state['rng_state'],cycle,z,
        (*excluded,*past,*root_values()),ROOTS['training'])
    immutable_array(directory/(prefix+'-directions.npy'),directions)
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False);readouts={};child_files=[]
    for index,value in enumerate([theta,*children]):
        role='incumbent' if index==0 else 'child'+str(index-1)
        parameter=state['theta_file'] if index==0 else prefix+'_'+role+'-theta.npy'
        immutable_array(directory/parameter,value);weight=prefix+'_'+role+'.npy';weights_file(directory,weight,value@matrix)
        readouts[role]=dict(theta_file=parameter,theta_hash=array_hash(value),weight_file=weight,weight_sha256=sha(directory/weight))
        if index:child_files.append(parameter)
    jobs=[]
    for index,role in enumerate(readouts):
        for s,seed in enumerate(discovery):
            job=native.rollout_job(f'{prefix}_discovery_{role}_s{s:02d}',seed,readouts[role]['weight_file'],'training_fitness')
            job.update(training_stage='discovery',cycle=cycle,policy_index=index,serve_index=s);jobs.append(job)
    definition=dict(cycle=cycle,input_theta_hash=array_hash(theta),input_theta_file=state['theta_file'],
        input_rng_state=state['rng_state'],rng_state_after_draw=after,directions_hash=array_hash(directions),
        noise_scale=scale,discovery_seeds=discovery,validation_seeds=validation,readouts=readouts,
        child_theta_files=child_files,discovery_jobs=jobs)
    immutable_json(directory/(prefix+'-definition.json'),definition);engine.register_jobs(directory,jobs)
    return definition


def validate_readouts(directory,definition):
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    m=json.loads((directory/'manifest.json').read_text())
    if sha(directory/'search-transform.npy')!=m['frozen_file_hashes']['search-transform.npy']:
        raise ValueError('fixed transform changed')
    directions=np.load(directory/f"c{definition['cycle']:02d}-directions.npy",allow_pickle=False)
    if array_hash(directions)!=definition['directions_hash'] or directions.shape!=(2,3,matrix.shape[0]):
        raise ValueError('pinned child directions changed')
    theta=np.load(directory/definition['input_theta_file'],allow_pickle=False)
    if array_hash(theta)!=definition['input_theta_hash']:raise ValueError('pinned incumbent changed')
    if (list(definition['readouts'])!=['incumbent',*(f'child{i}' for i in range(4))]
            or definition['child_theta_files']!=[definition['readouts'][f'child{i}']['theta_file'] for i in range(4)]):
        raise ValueError('fixed child order/parameter ownership changed')
    for index,(role,readout) in enumerate(definition['readouts'].items()):
        value=np.load(directory/readout['theta_file'],allow_pickle=False)
        expected=theta if index==0 else theta+definition['noise_scale']*directions[(index-1)//2]*(1. if index%2 else -1.)
        if (not np.array_equal(value,expected) or array_hash(value)!=readout['theta_hash']
                or sha(directory/readout['weight_file'])!=readout['weight_sha256']
                or not np.array_equal(value@matrix,np.load(directory/readout['weight_file'],allow_pickle=False))):
            raise ValueError('child parameters differ from their actual executed mapped readout')


def collect_wins(directory,jobs,seeds,weights):
    expected={(p,s) for p in range(len(weights)) for s in range(len(seeds))};seen=set();reports=[]
    for job in jobs:
        key=(job['policy_index'],job['serve_index'])
        if (any(type(i) is not int for i in key) or key not in expected or key in seen or job['config']['seed']!=seeds[key[1]]
                or job['weights_from']!=weights[key[0]]):raise ValueError('duplicate/mismatched training fitness cell')
        seen.add(key)
    if seen!=expected:raise ValueError('full unique discovery/validation roster required')
    wins=np.zeros((len(weights),len(seeds)),dtype=int)
    for job in jobs:
        report=completion.complete_report(directory,job)
        if report is None:return None
        if (not report['complete'] or report['episodes']!=1 or report['wins'] not in (0,1)
                or report['total_reward'] not in (-1.,0.,1.) or report['wins']!=int(report['total_reward']>0)):
            raise ValueError('completed raw terminal win fitness required')
        wins[job['policy_index'],job['serve_index']]=report['wins']
        reports.append(dict(job=job['id'],wins=report['wins'],terminal_score=report['total_reward'],transitions=report['transitions']))
    return wins.sum(axis=1).astype(int).tolist(),reports


def discovery_result(directory,definition):
    validate_readouts(directory,definition)
    result=collect_wins(directory,definition['discovery_jobs'],definition['discovery_seeds'],
                        [v['weight_file'] for v in definition['readouts'].values()])
    if result is None:return None
    wins,reports=result
    value=dict(wins_by_policy=wins,selected_child=transaction.choose_child(wins),reports=reports)
    path=directory/f"c{definition['cycle']:02d}-discovery.json"
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('pinned discovery decision changed')
    return value


def validation_jobs(definition,selected):
    if type(selected) is not int or not 0<=selected<4:raise ValueError('one discovered child required')
    jobs=[];prefix=f"c{definition['cycle']:02d}"
    for p,role in enumerate(('incumbent','child'+str(selected))):
        for s,seed in enumerate(definition['validation_seeds']):
            job=native.rollout_job(f'{prefix}_validation_{role}_s{s:02d}',seed,definition['readouts'][role]['weight_file'],'training_fitness')
            job.update(training_stage='validation',cycle=definition['cycle'],selected_child=selected,policy_index=p,serve_index=s);jobs.append(job)
    return jobs


def validation_result(directory,definition,selected):
    validate_readouts(directory,definition)
    result=collect_wins(directory,validation_jobs(definition,selected),definition['validation_seeds'],
        [definition['readouts'][role]['weight_file'] for role in ('incumbent','child'+str(selected))])
    if result is None:return None
    wins,reports=result;return dict(incumbent=wins[0],child=wins[1],reports=reports)


def training_failed(directory):
    return native.logged_failures(directory)[0]


def train(directory,pool,m,z,state,theta):
    api=sys.modules[__name__];marker=directory/'training-endpoint.json'
    if marker.exists():
        frozen=json.loads(marker.read_text())
        if any(frozen[k]!=v for k,v in transaction.freeze_fields(api,directory,state).items()):
            raise ValueError('frozen incumbent/cycle/cost boundary changed')
        return frozen,state,theta
    deadline=m['started_wall_ns']/1e9+TRAIN_SECONDS
    excluded=(*json.loads((directory/'seed-exclusions.json').read_text())['ids'],*m['serve_seeds'],*m['action_seeds'])
    reason='maximum complete cycles'
    while state['completed_cycles']<MAX_CYCLES:
        failed=training_failed(directory)
        if failed and not (directory/f"c{state['next_cycle']:02d}-result.json").exists():
            reason='persisted training failure';break
        path=directory/f"c{state['next_cycle']:02d}-definition.json"
        # A fully pinned discovery rejection or complete validation can commit
        # with zero new observations before checking the exhausted budget.
        if path.exists():
            definition=prepare_cycle(directory,z,state,theta,excluded)
            selection=discovery_result(directory,definition)
            if selection is not None:
                immutable_json(directory/f"c{definition['cycle']:02d}-discovery.json",selection)
                selected=selection['selected_child']
                if selected is not None:engine.register_jobs(directory,validation_jobs(definition,selected))
                if selected is None or validation_result(directory,definition,selected) is not None:
                    state,theta=transaction.finish_cycle(api,directory,definition,state,theta)
                    continue
        if failed:reason='persisted training failure';break
        if TRAIN_CAMERAS-training_count(directory)<200 or deadline-time.time()<=25:
            reason='training camera/wall budget';break
        definition=prepare_cycle(directory,z,state,theta,excluded)
        result=execute(directory,definition['discovery_jobs'],pool,deadline,TRAIN_CAMERAS,training_count(directory))
        if not result['complete']:
            reason='discovery execution failure' if result['failed'] else 'training camera/wall budget during discovery';break
        selection=discovery_result(directory,definition)
        immutable_json(directory/f"c{definition['cycle']:02d}-discovery.json",selection)
        selected=selection['selected_child']
        if selected is not None:
            jobs=validation_jobs(definition,selected);engine.register_jobs(directory,jobs)
            result=execute(directory,jobs,pool,deadline,TRAIN_CAMERAS,training_count(directory))
            if not result['complete']:
                reason='validation execution failure' if result['failed'] else 'training camera/wall budget during validation';break
        state,theta=transaction.finish_cycle(api,directory,definition,state,theta)
        print(json.dumps(dict(completed_cycles=state['completed_cycles'],accepted_updates=state['accepted_updates'],
                              training_cameras=training_count(directory))),flush=True)
    frozen=dict(**transaction.freeze_fields(api,directory,state),training_stop=reason,frozen_wall_ns=time.time_ns())
    atomic_json(marker,frozen);return frozen,state,theta


def evaluation_jobs(directory,m,state):
    endpoint=f"incumbent_after_cycle{state['completed_cycles']}.npy"
    if state['completed_cycles']==0:endpoint='start.npy'
    files=dict(endpoint=endpoint,start='start.npy',original='original.npy',
               seed1_initial='seed1_initial.npy',seed2_initial='seed2_initial.npy')
    jobs=[]
    for policy in POLICIES:
        for i,seed in enumerate(m['serve_seeds']):
            kind='trained' if policy in files else 'random' if policy.startswith('random') else policy
            job=native.rollout_job(f'eval_{policy}_s{i:02d}',seed,files.get(policy),'fresh_evaluation',kind=kind,
                action_seed=m['action_seeds'][i] if kind=='random' else None,hold=10 if policy=='random10' else 1)
            job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def evidence(groups,draws=10000):
    if any(len(groups.get(n,[]))!=SERVES for n in POLICIES):return dict(complete=False)
    for i in range(SERVES):
        shape=[None if (f:=groups[n][i]['first_incoming']) is None else {k:v for k,v in f.items() if k!='success'} for n in POLICIES]
        if any(v!=shape[0] for v in shape[1:]):raise ValueError('shared first-incoming geometry differs')
    index=np.random.default_rng(ROOTS['bootstrap']).integers(SERVES,size=(draws,SERVES));points={};samples={}
    for name in POLICIES:
        values=dict(total=np.array([r['reward']>0 for r in groups[name]],dtype=float),
            own_hit=np.array([r['reward']>0 and r['own_returns']>0 for r in groups[name]],dtype=float))
        points[name]={k:float(v.mean()) for k,v in values.items()}
        samples[name]={k:v[index].mean(1) for k,v in values.items()}
    result=dict(complete=True,controls=list(CONTROLS),serve_clusters=SERVES,bootstrap_seed=ROOTS['bootstrap'],
        bootstrap_draws=draws,endpoint={},endpoint_minus_start={},endpoint_minus_original={},qualifies=False)
    passed=True
    for metric in ('total','own_hit'):
        max_point=max(points[n][metric] for n in CONTROLS)
        delta=samples['endpoint'][metric]-np.max([samples[n][metric] for n in CONTROLS],axis=0)
        point=points['endpoint'][metric]-max_point;ci=np.quantile(delta,[.025,.975]).tolist()
        result['endpoint'][metric]=dict(lift=point,ci95=ci,strongest_control_rate=max_point)
        passed=passed and point>=.10 and ci[0]>0
        for control in ('start','original'):
            result['endpoint_minus_'+control][metric]=dict(lift=points['endpoint'][metric]-points[control][metric],
                ci95=np.quantile(samples['endpoint'][metric]-samples[control][metric],[.025,.975]).tolist())
    result['qualifies']=bool(passed);return result


def evaluate(directory,jobs,pool,deadline):
    m=json.loads((directory/'manifest.json').read_text());purpose={j['id']:j['purpose'] for j in m['jobs']}
    def failed():
        path=directory/'rollout-costs.jsonl'
        return path.exists() and any(not c['ok'] and purpose[c['job']]=='fresh_evaluation'
            for c in map(json.loads,path.read_text().splitlines()))
    marker=directory/'rollout-boundary.json'
    if not marker.exists():
        result=execute(directory,[] if failed() else jobs,pool,deadline,EVAL_CAMERAS,counts(directory,m)['fresh_evaluation'])
        result['failed']=result['failed'] or bool(failed())
        result['complete']=all(completion.complete_report(directory,j) is not None for j in jobs)
        result['unfinished']=[j['id'] for j in jobs if completion.complete_report(directory,j) is None]
        atomic_json(marker,dict(evaluation=result,frozen_wall_ns=time.time_ns()))
    result=json.loads(marker.read_text())['evaluation'];m=json.loads((directory/'manifest.json').read_text())
    if (result['failed']!=bool(failed()) or result['complete']!=all(completion.complete_report(directory,j) is not None for j in jobs)
            or result['committed_cameras']!=counts(directory,m)['fresh_evaluation']):
        raise ValueError('frozen evaluation failure/count boundary changed')
    return result


def outcome_evidence(directory,m):
    value=transfer.outcome_evidence(directory,m)
    value['search_files']={p.name:sha(p) for pattern in ('search-state.json','training-endpoint.json','c*-definition.json',
        'c*-discovery.json','c*-result.json','c*-directions.npy','c*_*.npy','c*_*.json','theta_after_cycle*.npy',
        'incumbent_after_cycle*.npy','incumbent_after_cycle*.json','evaluation-phase.json') for p in directory.glob(pattern)}
    return value


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('summary lacks authoritative completion')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained child evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained child summary changed')
    return saved['summary']


def supervise(directory):
    if DRAFT_ONLY:raise ValueError('draft only: no candidate execution')
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m,z,state,theta=prepare(directory)
        old=retained_outcome(directory,m)
        if old is not None:return old
        if 'started_wall_ns' not in m:m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        start=m['started_wall_ns']/1e9
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
            frozen,state,theta=train(directory,pool,m,z,state,theta)
            jobs=evaluation_jobs(directory,m,state) if frozen['valid'] else [];engine.register_jobs(directory,jobs)
            immutable_json(directory/'evaluation-phase.json',dict(started_wall_ns=frozen['frozen_wall_ns']))
            eval_start=json.loads((directory/'evaluation-phase.json').read_text())['started_wall_ns']/1e9
            evaluated=evaluate(directory,jobs,pool,min(start+TOTAL_SECONDS,eval_start+EVAL_SECONDS)-120.)
        annotations=prior.physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        groups={};rows={}
        for name in POLICIES:
            ordered=sorted((j for j in jobs if j['policy']==name),key=lambda j:j['serve_index'])
            if ordered and [j['serve_index'] for j in ordered]!=list(range(SERVES)):raise ValueError('cohort ordering changed')
            data=[r for j in ordered if j['id'] in annotations for r in annotations[j['id']]['rallies']]
            rows[name]=data;groups[name]=dict(complete=len(data)==SERVES,**aggregate(data))
            groups[name].update(transfer.score_counts(data))
        paired=evidence(rows);m=json.loads((directory/'manifest.json').read_text());actual=counts(directory,m)
        reports=[(j,engine.read_report(directory,j['id'])) for j in m['jobs']]
        logical=dict(initial_images=sum(r.get('initial_cameras',0) for _,r in reports if r),
            transition_images=sum(actual.values()),encoded_images=sum(r.get('initial_cameras',0)+r['transitions'] for j,r in reports
                if r and not j['config']['skip_baseline_perception']),new_visual_learning_images=0,
            annotation_physics_reexecutions=sum(v['verified_camera_rows'] for v in annotations.values()),
            partial_receipt_caveat='Initial/encoded counts use retained receipts; committed journal transitions are authoritative')
        stage_counts={name:sum(engine.sample_count(directory,j['id']) for j in m['jobs'] if j.get('training_stage')==name)
                      for name in ('discovery','validation')}
        retained=outcome_evidence(directory,m);elapsed=time.time()-start;train_elapsed=frozen['frozen_wall_ns']/1e9-start
        eval_elapsed=time.time()-eval_start;valid=bool(frozen['valid'] and evaluated['complete'] and not evaluated['failed']
            and train_elapsed<=TRAIN_SECONDS and eval_elapsed<=EVAL_SECONDS and elapsed<=TOTAL_SECONDS and paired.get('complete',False))
        summary=dict(status='complete' if valid else 'partial',evidence_eligible=valid,qualifies=valid and paired.get('qualifies',False),
            counts=actual,training_stage_counts=stage_counts,logical_compute=logical,training_endpoint=frozen,evaluation=evaluated,
            groups=groups,paired_score_evidence=paired,wall_seconds=elapsed,training_wall_seconds=train_elapsed,
            evaluation_wall_seconds=eval_elapsed,training_budget_met=train_elapsed<=TRAIN_SECONDS,
            evaluation_budget_met=eval_elapsed<=EVAL_SECONDS,overall_budget_met=elapsed<=TOTAL_SECONDS,
            selected_checkpoint=state['theta_file'],selection='last committed incumbent; no historical best selection',
            scope='One native-win executed-child training condition; validation is optimizer training, not held-out evidence',
            final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=retained))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();stage(a.directory) if a.stage else supervise(a.directory)
