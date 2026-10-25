"""One reward-selected self-repair regression proposal in the existing linear head."""
import argparse,copy,json,multiprocessing as mp,os,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from fast_pong_feature_rollout import run_episode
from pong_rewarded_residual import fit_residual,cap_residual,action_diagnostics
import run_practical_pong_v2 as base

CONFIG=dict(wall_seconds=600,workers=4,rows_per_episode=16,preservation_episodes_per_phase=8,ridge=1e-3,
    kappa_q_rms_fraction=.05,maximum_residual_function_rms=.10,target_positive_disagreement=.10,fresh_probe_per_phase=32)
ROOTS=dict(left=2237961451,right=3975108263);TAG=43721


def choose_witness(options):return min(options,key=lambda v:(v['distance'],v['weights']))


def fresh_seeds():
    used=set(base.old_engine.FINAL)
    def collect(v):
        if type(v) is int and 0<v<2**32:used.add(v)
        elif isinstance(v,list):
            for x in v:collect(x)
        elif isinstance(v,dict):
            for x in v.values():collect(x)
    for p in base.EXPERIMENTS.rglob('schedules.json'):collect(base.read(p))
    for p in base.EXPERIMENTS.glob('*/seed-exclusions.json'):collect(base.read(p).get('ids',[]))
    used.update(ROOTS.values());canonical={v%base.MODULUS for v in used};out={};rejects=[]
    for phase in ('left','right'):
        rng=np.random.default_rng(np.random.SeedSequence([ROOTS[phase],TAG]));out[phase]=[]
        while len(out[phase])<32:
            s=int(rng.integers(1,2**32,dtype=np.uint64));reserved=s in used or s%base.MODULUS in canonical
            used.add(s);canonical.add(s%base.MODULUS)
            if reserved or base.leftward(s)!=(phase=='left'):rejects.append(dict(seed=s,phase=phase))
            else:out[phase].append(s)
    return dict(probe=out,rejected=rejects,roots=ROOTS,tag=TAG)


def source_outcomes(sources):
    cached={};children={}
    for source in sources:
        for p in (source/'episodes').glob('*.json'):
            r=base.read(p)
            if not r.get('complete') or not r.get('observations_exact') or not r.get('first_hit_cutoff'):continue
            key=(r['weights_sha256'],base.config_key(r['config']));cached[key]=dict(hit=int(r['report']['hits']>0),receipt=str(p))
            if source==sources[0] and r['weights_file'].startswith('g') and '_d' in r['weights_file']:
                children.setdefault(r['weights_file'],{})[str(r['seed'])]=int(r['report']['hits']>0)
    return cached,children


def stage(directory,source,carry_theta):
    directory=Path(directory);source=Path(source);assert (source/'block-summary.json').exists(),'finish current practical block before staging'
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    (directory/'episodes').mkdir();(directory/'features').mkdir()
    theta=np.load(carry_theta);matrix=np.load(source/'search-transform.npy');weights=theta@matrix
    base.save_array(directory/'original_start.npy',np.load(base.PARENT/'endpoint.npy'));base.save_array(directory/'carry_theta.npy',theta);base.save_array(directory/'carry.npy',weights);base.save_array(directory/'search-transform.npy',matrix)
    base.atomic_json(directory/'radial-cap.json',base.read(source/'radial-cap.json'));bank=base.read(source/'state.json')['bank'];cached,children=source_outcomes([source,base.PREVIOUS])
    jobs=base.bank_jobs(directory,'carry',bank,'carry.npy');carry_hits={};provenance=[];weight_hash=base.sha(directory/'carry.npy')
    for j in jobs:
        old=cached.get((weight_hash,base.config_key(j['config'])));assert old is not None,'approved carry must have complete known bank outcomes'
        carry_hits[str(j['seed'])]=old['hit'];provenance.append(old['receipt'])
    raw=base.old_engine.read_calibration(base.old_engine.CACHE);candidates=[]
    all_seeds=set(map(str,sum(bank.values(),[])))
    for name,hits in sorted(children.items()):
        if not all_seeds.issubset(hits):continue
        W=np.load(source/name);distance=float(np.sqrt(np.mean((raw@(W-weights).T)**2)))
        candidates.append(dict(weights=name,hits=hits,distance=distance))
    repairs=[]
    for phase,seeds in bank.items():
        for seed in seeds:
            if carry_hits[str(seed)]:continue
            options=[c for c in candidates if c['hits'][str(seed)]]
            if not options:continue
            c=choose_witness(options);name=f'repair_policy_{len(repairs):02d}.npy';base.save_array(directory/name,np.load(source/c['weights']))
            repairs.append(dict(phase=phase,seed=seed,weights=name,source_weights=str(source/c['weights']),distance=c['distance']))
    preservation={}
    for p,ss in bank.items():
        successes=[s for s in ss if carry_hits[str(s)]];n=min(8,len(successes));preservation[p]=[successes[i] for i in np.linspace(0,len(successes)-1,n,dtype=int)] if n else []
    schedule=fresh_seeds();schedule.update(bank=bank,repairs=repairs,preservation=preservation);base.atomic_json(directory/'schedules.json',schedule)
    # Existing legal state buffer is calibration only, never candidate simulation.
    with np.load(source/'feature-buffer.npz') as b:np.savez(directory/'feature-buffer.npz',X=b['X'],weights=b['weights'])
    manifest=dict(config=CONFIG,source=base.source_signature(),model_hash=base.MODEL_HASH,driver_hash=base.sha(__file__),
        math_hash=base.sha(Path(__file__).with_name('pong_rewarded_residual.py')),capture_hash=base.sha(Path(__file__).with_name('fast_pong_feature_rollout.py')),
        practical_helpers_hash=base.sha(Path(base.__file__)),matrix_hash=base.ah(matrix),carry_source=str(carry_theta),carry_hash=base.ah(theta),carry_hits=carry_hits,carry_receipts=provenance,
        bank=bank,repair_witnesses=repairs,preservation=preservation,source_block=str(source),
        assistance='own rewarded successful action trajectories, phase-balanced preservation, regularized preference regression; no oracle, BP or new architecture',
        fresh_probe_semantics='original RIGHT START/carry/selected on common fresh starts only if head changes; exact duplicate heads skip execution; not final128-per-phase validation')
    base.atomic_json(directory/'manifest.json',manifest)
    print(json.dumps(dict(staged=True,new_observations=0,repairs=len(repairs),repair_phases={p:sum(r['phase']==p for r in repairs) for p in bank},preservation={p:len(v) for p,v in preservation.items()},driver_hash=manifest['driver_hash'])),flush=True)


def task(directory,j,deadline):
    directory=Path(directory);capture=j['purpose']=='train_capture';extras={}
    if capture:extras=dict(feature_matrix=np.load(directory/'search-transform.npy'),feature_rows_per_episode=500)
    r=run_episode(np.load(directory/j['weights']),j['config'],first_hit_cutoff=True,max_transitions=500,deadline_monotonic=deadline,**extras)
    data=r.pop('feature_capture',None)
    if capture and data is not None:
        total=len(data['X']);theta=np.load(directory/'carry_theta.npy');a0=np.argmax(data['X']@theta.T,axis=1)
        keep=np.flatnonzero(data['actions']!=a0) if j['capture_kind']=='repair' else np.arange(total)
        raw_rows=len(keep);keep=keep[np.linspace(0,raw_rows-1,min(16,raw_rows),dtype=int)] if raw_rows else keep
        path=directory/'features'/(j['id']+'.npz');np.savez(path,X=data['X'][keep],actions=data['actions'][keep])
        r.update(feature_file=str(path.relative_to(directory)),captured_rows=total,contrast_rows=raw_rows,retained_rows=len(keep))
    r.update(id=j['id'],seed=j['seed'],weights_file=j['weights'],weights_sha256=base.sha(directory/j['weights']),phase=j['phase'],purpose='train' if capture else j['purpose'],capture_kind=j.get('capture_kind'))
    base.atomic_json(directory/'episodes'/(j['id']+'.json'),r);return r


def dispatch(directory,jobs,pool,deadline):
    pending=list(jobs);active={};reports={}
    while pending or active:
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            j=pending.pop(0)
            with (directory/'dispatch.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='dispatch',job=j,wall_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
            active[pool.submit(task,str(directory),j,deadline-5)]=j
        if not active:break
        done,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
        for future in done:
            j=active.pop(future);reports[j['id']]=future.result()
    return reports,len(reports)==len(jobs) and all(r['complete'] and not r['failed'] for r in reports.values())


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json')
    assert m['config']==CONFIG and m['driver_hash']==base.sha(__file__) and m['math_hash']==base.sha(Path(__file__).with_name('pong_rewarded_residual.py')) and m['capture_hash']==base.sha(Path(__file__).with_name('fast_pong_feature_rollout.py')) and m['source']==base.source_signature() and m['practical_helpers_hash']==base.sha(Path(base.__file__))
    assert not (directory/'launch.json').exists(),'do not reset/restart this one-proposal clock'
    base.atomic_json(directory/'launch.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()));start=base.read(directory/'launch.json')['started_monotonic'];deadline=start+CONFIG['wall_seconds']
    theta=np.load(directory/'carry_theta.npy');matrix=np.load(directory/'search-transform.npy');bank=m['bank'];s=base.read(directory/'schedules.json');capture=[]
    for i,r in enumerate(s['repairs']):
        j=base.job(directory,f'repair_{i:02d}',r['seed'],r['weights'],r['phase'],'train_capture');j['capture_kind']='repair';capture.append(j)
    for p,seeds in s['preservation'].items():
        for i,seed in enumerate(seeds):
            j=base.job(directory,f'preserve_{p}_{i:02d}',seed,'carry.npy',p,'train_capture');j['capture_kind']='preservation';capture.append(j)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        reports,ok=dispatch(directory,capture,pool,deadline)
        if not ok:raise RuntimeError('partial capture: preserve receipts/grants; no fit/commit')
        positives=[];preservation=[]
        for j in capture:
            r=reports[j['id']];assert r['report']['hits']>0,'self-generated success witness must reproduce'
            with np.load(directory/r['feature_file']) as v:case=dict(phase=j['phase'],X=v['X'],actions=v['actions'])
            if len(case['X']):(positives if j['capture_kind']=='repair' else preservation).append(case)
        with np.load(directory/'feature-buffer.npz') as b:X=b['X'];w=b['weights']
        raw_delta,fit=fit_residual(theta,positives,preservation,X,w,ridge=CONFIG['ridge'],kappa_fraction=CONFIG['kappa_q_rms_fraction'])
        z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@matrix.T)
        delta,dose=cap_residual(theta,raw_delta,X,w,z,target=CONFIG['target_positive_disagreement'],maximum_rms=CONFIG['maximum_residual_function_rms']);fit.update(action_diagnostics(theta,delta,positives,preservation))
        proposed=theta+delta;base.save_array(directory/'raw_residual.npy',raw_delta);base.save_array(directory/'applied_residual.npy',delta)
        base.save_array(directory/'proposal_theta.npy',proposed);base.save_array(directory/'proposal.npy',proposed@matrix)
        before=base.phases(bank,m['carry_hits']);same=base.ah(proposed)==base.ah(theta)
        if same:proposal_hits=m['carry_hits'];ok=True
        else:
            jobs=base.bank_jobs(directory,'proposal',bank,'proposal.npy');rr,ok=dispatch(directory,jobs,pool,deadline)
            if not ok:raise RuntimeError('partial closed-loop proposal: preserve data, no commit')
            proposal_hits=base.hits_for(jobs,rr)
        after=base.phases(bank,proposal_hits);selected=tuple(after['key'])>tuple(before['key']);chosen=proposed if selected else theta
        base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',chosen@matrix)
        probe={}
        if selected and not same:
            for p,seeds in s['probe'].items():
                jobs=[];aliases={};seen={}
                for label in ('original_start','carry','selected'):
                    digest=base.sha(directory/(label+'.npy'));aliases[label]=seen.setdefault(digest,label)
                for i,seed in enumerate(seeds):
                    for label in dict.fromkeys(aliases.values()):jobs.append(base.job(directory,f'probe_{p}_{label}_{i:02d}',seed,label+'.npy',p,'phase_probe'))
                rr,ok=dispatch(directory,jobs,pool,deadline)
                if not ok:break
                probe[p]={label:[int(rr[f'probe_{p}_{aliases[label]}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in ('original_start','carry','selected')}
        summary=dict(status='one_proposal_complete' if ok else 'partial_probe',fit=fit,dose=dose,bank_before=before,bank_proposal=after,selected_proposal=selected,
            selected_theta_hash=base.ah(chosen),residual_rescaling_can_amplify=True,probe=probe,no_changed_head_probe_skipped=not selected,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,
            interpretation='One directed self-repair preference proposal, no Q/teacher/BP; reused bank and new probe progress, no final reliability claim')
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--carry-theta',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source,a.carry_theta)
    else:run(a.root)
