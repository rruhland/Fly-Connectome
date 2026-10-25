"""Acquire repairs from reward-trained skill donors while retaining current gains."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import osqp
import run_preserved_rewarded_residual_pong as worker
import run_rewarded_residual_pong as cache_tools
from pong_constrained_residual import solve_constrained
from pong_rewarded_residual import cap_residual,action_diagnostics,positive_disagreement
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction
base=worker.base


def read_case(path,phase,seed):
    with np.load(path) as v:return dict(phase=phase,seed=seed,**{k:v[k] for k in v.files})


def refilter(case,theta,hits):
    if hits[str(case['seed'])]:return None
    keep=case['actions']!=np.argmax(case['X']@theta.T,axis=1)
    if not np.any(keep):return None
    return dict(phase=case['phase'],seed=case['seed'],X=case['X'][keep],actions=case['actions'][keep])


def stage(directory,source):
    directory=Path(directory);source=Path(source);result=base.read(source/'summary.json');assert result['selected_proposal']
    source_m=base.read(source/'manifest.json');old_protected=Path(source_m['previous']);old_m=base.read(old_protected/'manifest.json')
    old_residual=Path(old_m['previous']);r2=Path(base.read(old_residual/'manifest.json')['source_block'])
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for name in ('episodes','features','reused_features'):(directory/name).mkdir()
    theta=np.load(source/'selected_theta.npy');matrix=np.load(source/'search-transform.npy')
    files={'carry_theta.npy':theta,'carry.npy':theta@matrix,'fallback_theta.npy':np.load(source/'carry_theta.npy'),'fallback.npy':np.load(source/'carry.npy'),
        'search-transform.npy':matrix,'original_start.npy':np.load(source/'original_start.npy'),'direct.npy':np.load(base.EXPERIMENTS/'pong-leftward-population-20261022-r1/endpoint.npy')}
    for name,x in files.items():base.save_array(directory/name,x)
    for name in ('radial-cap.json','feature-buffer.npz'):(directory/name).write_bytes((source/name).read_bytes())
    seeds=base.read(source/'schedules.json');bank={p:list(source_m['bank'][p])+list(seeds['probe'][p]) for p in ('left','right')}
    assert sum(map(len,bank.values()))==206
    cached,_=cache_tools.source_outcomes([source,old_protected,old_residual,r2,base.PREVIOUS]);hitsets={}
    for label in ('carry','fallback'):
        hits={};weight_hash=base.sha(directory/(label+'.npy'))
        for j in base.bank_jobs(directory,label,bank,label+'.npy'):
            record=cached.get((weight_hash,base.config_key(j['config'])));assert record is not None
            hits[str(j['seed'])]=record['hit']
        hitsets[label]=hits
    reused_positive=[];reused_preserve=[];reused_guards=[]
    def store(case,label,target):
        name=f'reused_features/{label}_{len(target):02d}.npz';np.savez(directory/name,**{k:v for k,v in case.items() if k not in ('phase','seed')})
        target.append(dict(phase=case['phase'],seed=case['seed'],feature_file=name))
    for kind,items in old_m['reused'].items():
        for item in items:
            receipt=base.read(item['receipt']);case=read_case(old_protected/item['feature_file'],item['phase'],receipt['seed'])
            if kind=='repair':
                case=refilter(case,theta,hitsets['carry'])
                if case is not None:store(case,'positive',reused_positive)
            elif hitsets['carry'][str(case['seed'])]:store(case,'preserve',reused_preserve)
    for path in sorted((old_protected/'episodes').glob('*.json')):
        r=base.read(path);kind=r.get('capture_kind')
        if kind not in ('repair','guard'):continue
        case=read_case(old_protected/r['feature_file'],r['phase'],r['seed'])
        if kind=='repair':
            case=refilter(case,theta,hitsets['carry'])
            if case is not None:store(case,'positive',reused_positive)
        else:
            assert hitsets['carry'][str(case['seed'])]
            case['q']=case['X']@theta.T;assert np.array_equal(np.argmax(case['q'],axis=1),case['actions']),'old RIGHT path choices must remain unchanged'
            store(case,'guard',reused_guards);store(case,'preserve',reused_preserve)
    assert len({c['seed'] for c in reused_positive})==len(reused_positive)
    queries=[dict(phase='left',seed=s,weights='direct.npy',donor='DIRECT',expected_success=False) for s in bank['left'] if not hitsets['carry'][str(s)]]
    queries +=[dict(phase='right',seed=s,weights='fallback.npy',donor='prior_RIGHT',expected_success=True) for s in bank['right'] if hitsets['fallback'][str(s)] and not hitsets['carry'][str(s)]]
    new_guards=[]
    for p in ('left','right'):
        probe=result['probe'][p]
        for seed,old,current in zip(seeds['probe'][p],probe['carry'],probe['selected']):
            if current and not old:new_guards.append(dict(phase=p,seed=seed))
    fresh=worker.fresh_seeds();fresh.update(bank=bank,queries=queries,new_guards=new_guards,guards=[c['seed'] for c in reused_guards]+[c['seed'] for c in new_guards]);base.atomic_json(directory/'schedules.json',fresh)
    m=dict(driver_hash=base.sha(__file__),solver_hash=base.sha(Path(__file__).with_name('pong_constrained_residual.py')),worker_hash=base.sha(Path(worker.__file__)),source=base.source_signature(),
        carry_hash=base.ah(theta),carry_hits=hitsets['carry'],fallback_hits=hitsets['fallback'],bank=bank,source_block=str(source),osqp_version=osqp.__version__,
        reused_positive=reused_positive,reused_preservation=reused_preserve,reused_guards=reused_guards,work_seconds=600,
        assistance='reward-trained DIRECT/prior-RIGHT donor repairs, current-success constraints, no oracle/BP/new class',
        donor_precedence='one episode per missed context: successful named skill donor before old child; old child fallback; recheck current action contrast')
    base.atomic_json(directory/'manifest.json',m)
    print(json.dumps(dict(staged=True,new_observations=0,bank_sizes={p:len(v) for p,v in bank.items()},carry=base.phases(bank,hitsets['carry']),fallback=base.phases(bank,hitsets['fallback']),
        donor_queries={d:sum(q['donor']==d for q in queries) for d in ('DIRECT','prior_RIGHT')},new_guard_captures={p:sum(g['phase']==p for g in new_guards) for p in bank},reused_positive=len(reused_positive),reused_guards=len(reused_guards))),flush=True)


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');assert m['driver_hash']==base.sha(__file__) and m['solver_hash']==base.sha(Path(__file__).with_name('pong_constrained_residual.py')) and m['worker_hash']==base.sha(Path(worker.__file__)) and m['source']==base.source_signature()
    assert not (directory/'work-clock.json').exists(),'do not restart/reset donor-mesh block'
    base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()));start=base.read(directory/'work-clock.json')['started_monotonic'];deadline=start+600
    theta=np.load(directory/'carry_theta.npy');matrix=np.load(directory/'search-transform.npy');s=base.read(directory/'schedules.json');bank=m['bank'];jobs=[]
    for i,q in enumerate(s['queries']):
        j=base.job(directory,f'donor_{i:02d}',q['seed'],q['weights'],q['phase'],'train_capture');j.update(capture_kind='repair',donor=q['donor'],expected_success=q['expected_success']);jobs.append(j)
    for i,g in enumerate(s['new_guards']):
        j=base.job(directory,f'guard_{i:02d}',g['seed'],'carry.npy',g['phase'],'train_capture');j['capture_kind']='guard';jobs.append(j)
    with worker.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        reports,ok=worker.dispatch(directory,jobs,pool,deadline)
        if not ok:raise RuntimeError('partial donor/guard collection; preserve receipts and budget')
        positive={};preserve=[];guards=[]
        for item in m['reused_positive']:
            case=read_case(directory/item['feature_file'],item['phase'],item['seed']);positive[case['seed']]=case
        for item in m['reused_preservation']:preserve.append(read_case(directory/item['feature_file'],item['phase'],item['seed']))
        for item in m['reused_guards']:guards.append(read_case(directory/item['feature_file'],item['phase'],item['seed']))
        donor_stats={d:dict(queries=0,successes=0,contrasting_successes=0) for d in ('DIRECT','prior_RIGHT')}
        for j in jobs:
            r=reports[j['id']];success=r['report']['hits']>0
            if j['capture_kind']=='guard':assert success,'new carry gain must reproduce'
            else:
                donor_stats[j['donor']]['queries']+=1;donor_stats[j['donor']]['successes']+=int(success)
                if j['expected_success']:assert success,'known rewarded donor must reproduce'
                if not success:continue
            case=read_case(directory/r['feature_file'],j['phase'],j['seed'])
            if j['capture_kind']=='guard':guards.append(case);preserve.append(case)
            elif len(case['X']):positive[case['seed']]=case;donor_stats[j['donor']]['contrasting_successes']+=1
        positive=list(positive.values())
        with np.load(directory/'feature-buffer.npz') as b:X=b['X'];w=b['weights']
        delta,fit=solve_constrained(theta,positive,preserve,guards,X,w)
        z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@matrix.T);_,dose=cap_residual(theta,delta,X,w,z)
        limit,protection=half_margin_bound(theta,delta,guards);alpha=min(dose['alpha'],limit);applied=alpha*delta
        protection.update(alpha_before=dose['alpha'],alpha_after=alpha,applied_function_rms=base.function_rms(applied,z),buffer_disagreement=positive_disagreement(theta,delta,X,w,alpha),guard_choices=preserved_action_fraction(delta,alpha,guards))
        assert protection['guard_choices']==1.
        fit.update(action_diagnostics(theta,applied,positive,preserve));base.atomic_json(directory/'fit-report.json',dict(fit=fit,dose=dose,protection=protection,donor_outcomes=donor_stats))
        proposed=theta+applied;base.save_array(directory/'proposal_theta.npy',proposed);base.save_array(directory/'proposal.npy',proposed@matrix)
        if base.ah(proposed)==base.ah(theta):hits=m['carry_hits']
        else:
            js=base.bank_jobs(directory,'proposal',bank,'proposal.npy');rr,ok=worker.dispatch(directory,js,pool,deadline)
            if not ok:raise RuntimeError('partial mesh bank check')
            hits=base.hits_for(js,rr)
        guard_returns=sum(hits[str(seed)] for seed in s['guards']);guard_ok=guard_returns==len(s['guards'])
        contenders=[dict(label='carry',score=base.phases(bank,m['carry_hits'])),dict(label='fallback',score=base.phases(bank,m['fallback_hits']))]
        if guard_ok:contenders.append(dict(label='proposal',score=base.phases(bank,hits)))
        winner=base.choose(contenders);chosen=proposed if winner['label']=='proposal' else theta if winner['label']=='carry' else np.load(directory/'fallback_theta.npy')
        base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',chosen@matrix);probe={}
        if base.ah(chosen)!=base.ah(theta):
            for p,seeds in s['probe'].items():
                aliases={};seen={};js=[]
                for label in ('original_start','carry','selected'):aliases[label]=seen.setdefault(base.sha(directory/(label+'.npy')),label)
                for i,seed in enumerate(seeds):
                    for label in dict.fromkeys(aliases.values()):js.append(base.job(directory,f'probe_{p}_{label}_{i:02d}',seed,label+'.npy',p,'phase_probe'))
                rr,ok=worker.dispatch(directory,js,pool,deadline)
                if not ok:break
                probe[p]={label:[int(rr[f'probe_{p}_{aliases[label]}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in aliases}
        summary=dict(status='one_mesh_proposal_complete' if ok else 'partial_probe',selected=winner,bank_carry=contenders[0]['score'],bank_proposal=base.phases(bank,hits),
            guard_returns=guard_returns,guard_episodes=len(s['guards']),actual_guards_preserved=guard_ok,fit=fit,dose=dose,protection=protection,donor_outcomes=donor_stats,probe=probe,
            accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,interpretation='own reward-trained skill meshing in same linear class; current gains preserved on known trajectories, no final90%claim')
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source)
    else:run(a.root)
