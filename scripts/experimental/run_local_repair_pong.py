"""Discover own rewarded local action alternatives, then integrate one guarded fit."""
import argparse,json,multiprocessing as mp,os,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import run_spatial_residual_pong as previous
from pong_local_discovery import state_indices,local_mutation
from fast_pong_feature_rollout import run_episode
from pong_constrained_residual import solve_constrained
from pong_spatial_extension import map_split
from pong_rewarded_residual import cap_residual,action_diagnostics,positive_disagreement
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction
base=previous.base


def pins():
    return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_local_repair_pong.py','pong_local_discovery.py','pong_constrained_residual.py',
        'pong_spatial_extension.py','pong_rewarded_residual.py','pong_behavior_preservation.py','fast_pong_feature_rollout.py','run_practical_pong_v2.py',
        'run_spatial_residual_pong.py','run_preserved_rewarded_residual_pong.py')}


def stage(directory,source):
    directory=Path(directory);source=Path(source);sm=base.read(source/'manifest.json');ss=base.read(source/'schedules.json');sr=base.read(source/'summary.json')
    assert sr['status']=='one_spatial_proposal_complete' and sr['selected']['label']=='proposal'
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for name in ('episodes','features','reused_features','mutants'):(directory/name).mkdir()
    old=np.load(source/'old-transform.npy');extra=np.load(source/'extra-transform.npy');theta=np.load(source/'selected_theta.npy')
    for label,name in [('carry','selected'),('fallback','carry')]:
        t=np.load(source/(name+'_theta.npy'));w=np.load(source/(name+'.npy'));assert np.array_equal(map_split(t,old,extra),w)
        base.save_array(directory/(label+'_theta.npy'),t);base.save_array(directory/(label+'.npy'),w)
    for name in ('old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json','feature-buffer.npz','original_start.npy'):(directory/name).write_bytes((source/name).read_bytes())
    bank={p:list(sm['bank'][p])+list(ss['probe'][p]) for p in ('left','right')};assert sum(map(len,bank.values()))==334
    hits={};fallback=dict(sm['carry_hits'])
    for path in (source/'episodes').glob('proposal_*.json'):
        r=base.read(path);assert r['complete'];hits[str(r['seed'])]=int(r['report']['hits']>0)
    for p in bank:
        for seed,h,f in zip(ss['probe'][p],sr['probe'][p]['selected'],sr['probe'][p]['carry']):hits[str(seed)]=h;fallback[str(seed)]=f
    assert len(hits)==334 and len(fallback)==334
    misses={int(seed) for seed,h in hits.items() if not h};known=set()
    for root in base.EXPERIMENTS.glob('pong-*-20261025-r*'):
        for path in (root/'episodes').glob('*.json'):
            r=base.read(path)
            if r.get('seed') in misses and r.get('complete') and r.get('observations_exact') and r.get('first_hit_cutoff') and r['report']['hits']>0:known.add(r['seed'])
    never={p:sorted(v for v in bank[p] if v in misses-known) for p in bank};known_miss={p:sorted(v for v in bank[p] if v in misses&known) for p in bank}
    assert len(never['left'])>=16 and len(never['right'])==15 and known_miss['right']
    targets=dict(left=never['left'][:16],right=never['right']+[known_miss['right'][0]])
    guards=list(ss['guards']);new_guards=[]
    for p in bank:
        new_guards.extend(dict(seed=v,phase=p) for v,a,b in zip(ss['probe'][p],sr['probe'][p]['carry'],sr['probe'][p]['selected']) if b and not a)
    guards+= [g['seed'] for g in new_guards];assert len(guards)==16
    reuse=[]
    for path in sorted((source/'episodes').glob('preserve_*.json')):
        r=base.read(path)
        with np.load(source/r['feature_file']) as v:case={k:v[k] for k in v.files}
        choices=np.argmax(case['X']@theta.T,axis=1)
        if not np.array_equal(choices,case['actions']):continue
        assert hits[str(r['seed'])];case['q']=case['X']@theta.T
        name='reused_features/'+path.stem+'.npz';np.savez(directory/name,**case)
        reuse.append(dict(seed=r['seed'],phase=r['phase'],kind=r['capture_kind'],feature_file=name))
    assert sum(c['kind']=='guard' for c in reuse)==13
    fresh=previous.worker.fresh_seeds();fresh.update(bank=bank,targets=targets,guards=guards,new_guards=new_guards)
    base.atomic_json(directory/'schedules.json',fresh)
    m=dict(source=base.source_signature(),files=pins(),source_block=str(source),bank=bank,carry_hits=hits,fallback_hits=fallback,reused=reuse,
        work_seconds=600,maximum_mutation_rms=.25,states_per_failure=4,alternatives_per_state=2,kappa_q_rms_fraction=.05,matching_soft_weight=.25,
        fresh_comparison='carry versus selected only,32ordinary starts perphase; original_start retained for lineage, not reexecuted',
        assistance='reward-selected local action alternatives on own legal failed states; no teacher, BP, new inputs or architecture',
        target_selection='16 never-observed-hit LEFT;15 never-observed-hit RIGHT plus1known-other-policy-hit RIGHT; numeric seed order')
    base.atomic_json(directory/'manifest.json',m)
    print(json.dumps(dict(staged=True,new_observations=0,bank=base.phases(bank,hits),targets={p:len(v) for p,v in targets.items()},guards=len(guards),reused_current_success_episodes=len(reuse),maximum_discovery_episodes=256)),flush=True)


def task(directory,j,deadline):
    directory=Path(directory);capture=j.get('capture_kind') is not None
    extras=dict(feature_matrix=np.load(directory/'search-transform.npy'),feature_rows_per_episode=500) if capture else {}
    r=run_episode(np.load(directory/j['weights']),j['config'],first_hit_cutoff=True,max_transitions=500,deadline_monotonic=deadline,**extras)
    data=r.pop('feature_capture',None)
    if data is not None:
        path=directory/'features'/(j['id']+'.npz');np.savez(path,**data);r['feature_file']=str(path.relative_to(directory))
    r.update(id=j['id'],seed=j['seed'],phase=j['phase'],weights_file=j['weights'],weights_sha256=base.sha(directory/j['weights']),purpose='train' if capture or j['purpose']=='discovery' else j['purpose'],capture_kind=j.get('capture_kind'))
    base.atomic_json(directory/'episodes'/(j['id']+'.json'),r);return r


def dispatch(directory,jobs,pool,deadline):
    pending=list(jobs);active={};reports={}
    while pending or active:
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            j=pending.pop(0)
            with (directory/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(event='dispatch',job=j,wall_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
            active[pool.submit(task,str(directory),j,deadline-5)]=j
        if not active:break
        done,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
        for future in done:
            j=active.pop(future);reports[j['id']]=future.result()
    return reports,len(reports)==len(jobs) and all(r['complete'] and not r['failed'] for r in reports.values())


def case(directory,r):
    with np.load(directory/r['feature_file']) as v:return dict(seed=r['seed'],phase=r['phase'],**{k:v[k] for k in v.files})


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json')
    assert m['source']==base.source_signature() and m['files']==pins();assert not (directory/'work-clock.json').exists()
    start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start));deadline=start+600
    theta=np.load(directory/'carry_theta.npy');old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy');matrix=np.vstack([old,extra])
    assert np.array_equal(map_split(theta,old,extra),np.load(directory/'carry.npy'))
    z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@matrix.T)
    with np.load(directory/'feature-buffer.npz') as b:X=b['X'];w=b['weights']
    kappa=.05*float(np.sqrt(np.sum(w[:,None]*(X@theta.T)**2)/3));captures=[]
    for p,seeds in s['targets'].items():
        for i,seed in enumerate(seeds):
            j=base.job(directory,f'failure_{p}_{i:02d}',seed,'carry.npy',p,'train');j['capture_kind']='failure';captures.append(j)
    for i,g in enumerate(s['new_guards']):
        j=base.job(directory,f'guard_{i:02d}',g['seed'],'carry.npy',g['phase'],'train');j['capture_kind']='guard';captures.append(j)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        rr,ok=dispatch(directory,captures,pool,deadline);assert ok,'partial failure/guard capture'
        mutations=[];jobs=[];seen={};skipped=0
        for j in captures:
            if j['capture_kind']!='failure':assert rr[j['id']]['report']['hits']>0;continue
            r=rr[j['id']];assert r['report']['hits']==0,'selected current miss must reproduce';c=case(directory,r)
            assert np.max(abs(c['X']@theta.T-c['q']))<1e-10
            for slot,t in enumerate(state_indices(c['q'])):
                a=int(np.argmax(c['q'][t]))
                for b in (v for v in range(3) if v!=a):
                    delta,info=local_mutation(theta,c['X'][t],b,z,kappa=kappa,maximum_rms=.25)
                    weights=map_split(theta+delta,old,extra);jid=f'mutate_{j["phase"]}_{j["seed"]}_{slot}_{b}'
                    if np.array_equal(weights,np.load(directory/'carry.npy')):skipped+=1;continue
                    key=(j['seed'],base.ah(weights));alias=seen.get(key)
                    if alias is None:
                        name=f'mutants/{jid}.npy';base.save_array(directory/name,weights);trial=base.job(directory,jid,j['seed'],name,j['phase'],'discovery');jobs.append(trial);seen[key]=jid
                    else:name=next(v['weights'] for v in jobs if v['id']==alias)
                    mutations.append(dict(id=jid,executed_id=seen[key],seed=j['seed'],phase=j['phase'],weights=name,state_index=t,**info))
        base.atomic_json(directory/'mutation-report.json',dict(kappa=kappa,requested=256,candidates=len(mutations),unique_episodes=len(jobs),skipped_byte_identical=skipped,
            intended_flips=sum(v['intended_flip_achieved'] for v in mutations),cap_limited=sum(v['cap_limited'] for v in mutations),mutations=mutations))
        discovery,ok=dispatch(directory,jobs,pool,deadline);assert ok,'partial local discovery'
        successful={}
        for v in mutations:
            if discovery[v['executed_id']]['report']['hits']>0:successful.setdefault(v['seed'],[]).append(v)
        selected=[min(v,key=lambda c:(c['applied_function_rms'],c['id'])) for _,v in sorted(successful.items())]
        discovery_report=dict(candidates=len(mutations),unique_episodes=len(jobs),successful_episodes=sum(r['report']['hits']>0 for r in discovery.values()),
            repaired_contexts={p:sum(v['phase']==p for v in selected) for p in ('left','right')},intended_flips=sum(v['intended_flip_achieved'] for v in mutations),cap_limited=sum(v['cap_limited'] for v in mutations),selected=selected)
        base.atomic_json(directory/'discovery-report.json',discovery_report);print(json.dumps(dict(discovery=discovery_report)),flush=True)
        if not selected:
            base.atomic_json(directory/'summary.json',dict(status='complete_no_new_rewarded_repairs',discovery=discovery_report,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start));return
        donor_jobs=[]
        for i,v in enumerate(selected):
            j=base.job(directory,f'donor_{i:02d}',v['seed'],v['weights'],v['phase'],'train');j['capture_kind']='donor';donor_jobs.append(j)
        donor_reports,ok=dispatch(directory,donor_jobs,pool,deadline);assert ok,'partial rewarded donor capture'
        positive=[];matching=[];preserve=[];guards=[]
        for item in m['reused']:
            c=case(directory,item);preserve.append(c)
            if item['kind']=='guard':guards.append(c)
        for j in captures:
            if j['capture_kind']=='guard':c=case(directory,rr[j['id']]);guards.append(c);preserve.append(c)
        for j in donor_jobs:
            r=donor_reports[j['id']];assert r['report']['hits']>0;c=case(directory,r);different=c['actions']!=np.argmax(c['X']@theta.T,axis=1)
            assert different.any(),'rewarded repair of deterministic miss must contain differing actions'
            positive.append(dict(seed=c['seed'],phase=c['phase'],X=c['X'][different],actions=c['actions'][different]))
            if (~different).any():matching.append(dict(seed=c['seed'],phase=c['phase'],X=c['X'][~different]))
        assert len(guards)==16
        delta,fit=solve_constrained(theta,positive,preserve,guards,X,w,matching_preservation=matching,matching_weight=.25)
        _,dose=cap_residual(theta,delta,X,w,z);limit,protection=half_margin_bound(theta,delta,guards);alpha=min(dose['alpha'],limit);applied=alpha*delta
        protection.update(alpha_before=dose['alpha'],alpha_after=alpha,applied_function_rms=base.function_rms(applied,z),buffer_disagreement=positive_disagreement(theta,delta,X,w,alpha),guard_choices=preserved_action_fraction(delta,alpha,guards))
        assert protection['guard_choices']==1.;fit.update(action_diagnostics(theta,applied,positive,preserve))
        base.atomic_json(directory/'fit-report.json',dict(fit=fit,dose=dose,protection=protection));base.save_array(directory/'applied_residual.npy',applied)
        proposed=theta+applied;base.save_array(directory/'proposal_theta.npy',proposed);base.save_array(directory/'proposal.npy',map_split(proposed,old,extra))
        js=base.bank_jobs(directory,'proposal',m['bank'],'proposal.npy');bank_reports,ok=dispatch(directory,js,pool,deadline);assert ok,'partial proposal bank check'
        hits=base.hits_for(js,bank_reports);guard_returns=sum(hits[str(v)] for v in s['guards']);contenders=[dict(label='carry',score=base.phases(m['bank'],m['carry_hits'])),dict(label='fallback',score=base.phases(m['bank'],m['fallback_hits']))]
        if guard_returns==len(s['guards']):contenders.append(dict(label='proposal',score=base.phases(m['bank'],hits)))
        winner=base.choose(contenders);chosen=proposed if winner['label']=='proposal' else theta if winner['label']=='carry' else np.load(directory/'fallback_theta.npy')
        base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',map_split(chosen,old,extra));probe={}
        if base.ah(chosen)!=base.ah(theta):
            for phase,seeds in s['probe'].items():
                js=[base.job(directory,f'probe_{phase}_{label}_{i:02d}',seed,label+'.npy',phase,'phase_probe') for i,seed in enumerate(seeds) for label in ('carry','selected')]
                pr,ok=dispatch(directory,js,pool,deadline)
                if not ok:break
                probe[phase]={label:[int(pr[f'probe_{phase}_{label}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in ('carry','selected')}
        summary=dict(status='one_local_discovery_fit_complete' if ok else 'partial_probe',discovery=discovery_report,selected=winner,bank_carry=contenders[0]['score'],bank_proposal=base.phases(m['bank'],hits),
            guard_returns=guard_returns,guard_episodes=len(s['guards']),fit=fit,dose=dose,protection=protection,probe=probe,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start)
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source)
    else:run(a.root)
