"""One finer-spatial constrained own-reward repair proposal; inference stays linear."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import osqp
from fly_connectome.vision import VisualStateEncoder
import run_preserved_rewarded_residual_pong as worker
import run_rewarded_residual_pong as caches
from pong_spatial_extension import extra_transform,map_split
from pong_constrained_residual import solve_constrained
from pong_rewarded_residual import cap_residual,action_diagnostics,positive_disagreement
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction
from pong_safe_mutations import balanced_buffer
base=worker.base


def inputs(directory):
    return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_spatial_residual_pong.py','pong_spatial_extension.py',
        'pong_constrained_residual.py','run_preserved_rewarded_residual_pong.py','fast_pong_feature_rollout.py','run_practical_pong_v2.py')}


def stage(directory,source):
    directory=Path(directory);source=Path(source);m=base.read(source/'manifest.json');r=base.read(source/'summary.json');s=base.read(source/'schedules.json')
    assert r['status']=='one_mesh_proposal_complete' and r['selected']['label']=='proposal'
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for n in ('episodes','features','donors'):(directory/n).mkdir()
    old=np.load(source/'search-transform.npy');raw=base.old_engine.read_calibration(base.old_engine.CACHE)
    e=VisualStateEncoder(64,64,.02);extra,info=extra_transform(raw,e.layout,e.dimension)
    assert old.shape==(214,4389) and extra.shape==(352,4389)
    matrix=np.vstack([old,extra]);theta=np.pad(np.load(source/'selected_theta.npy'),((0,0),(0,len(extra))))
    fallback=np.pad(np.load(source/'carry_theta.npy'),((0,0),(0,len(extra))))
    for label,t,reference in [('carry',theta,source/'selected.npy'),('fallback',fallback,source/'carry.npy')]:
        w=map_split(t,old,extra);assert np.array_equal(w,np.load(reference)),'zero-added mapping must preserve exact existing W bytes'
        base.save_array(directory/(label+'_theta.npy'),t);base.save_array(directory/(label+'.npy'),w)
    for name,v in [('old-transform.npy',old),('extra-transform.npy',extra),('search-transform.npy',matrix),('original_start.npy',np.load(source/'original_start.npy'))]:base.save_array(directory/name,v)
    (directory/'radial-cap.json').write_bytes((source/'radial-cap.json').read_bytes())
    bank={p:list(m['bank'][p])+list(s['probe'][p]) for p in ('left','right')};assert sum(map(len,bank.values()))==270
    roots=[base.EXPERIMENTS/n for n in ('pong-practical-interception-20261025-r1','pong-practical-interception-20261025-r2',
        'pong-rewarded-residual-20261025-r1','pong-preserved-residual-20261025-r1','pong-constrained-residual-20261025-r1','pong-meshed-donor-20261025-r1')]
    cached,_=caches.source_outcomes(roots);hits={}
    for label in ('carry','fallback'):
        hits[label]={}
        for j in base.bank_jobs(directory,label,bank,label+'.npy'):
            known=cached.get((base.sha(directory/(label+'.npy')),base.config_key(j['config'])));assert known is not None
            hits[label][str(j['seed'])]=known['hit']
    options={};distances={};desired={str(v) for p in bank for v in bank[p] if not hits['carry'][str(v)]}
    for root in roots:
        for path in sorted((root/'episodes').glob('*.json')):
            rr=base.read(path);seed=str(rr.get('seed'));weight=root/rr.get('weights_file','')
            if seed not in desired or not rr.get('complete') or not rr.get('observations_exact') or not rr.get('first_hit_cutoff') or rr['report']['hits']<1 or not weight.is_file():continue
            phase='left' if int(seed) in bank['left'] else 'right';standard=base.job(directory,'check',int(seed),'carry.npy',phase,'train')
            if base.config_key(rr['config'])!=base.config_key(standard['config']):continue
            h=base.sha(weight)
            if h not in distances:distances[h]=float(np.sqrt(np.mean(np.asarray(raw@(np.load(weight)-np.load(directory/'carry.npy')).T)**2)))
            options.setdefault(seed,[]).append(dict(seed=int(seed),phase=phase,source=str(weight),receipt=str(path),distance=distances[h]))
    repairs=[]
    for seed in sorted(options,key=int):
        chosen=min(options[seed],key=lambda v:(v['distance'],v['source'],v['receipt']));name=f'donors/repair_{len(repairs):02d}.npy'
        (directory/name).write_bytes(Path(chosen['source']).read_bytes());repairs.append(dict(chosen,weights=name))
    guards=list(s['guards'])
    for p in ('left','right'):
        guards.extend(seed for seed,a,b in zip(s['probe'][p],r['probe'][p]['carry'],r['probe'][p]['selected']) if b and not a)
    guards=list(dict.fromkeys(guards));assert len(guards)==13 and all(hits['carry'][str(seed)] for seed in guards)
    preservation={}
    for p in ('left','right'):
        phase_guards=[v for v in guards if v in bank[p]]
        preservation[p]=(phase_guards+[v for v in bank[p] if hits['carry'][str(v)] and v not in phase_guards])[:16]
        assert len(preservation[p])==16
    # Reuse the already checked prospective seed generator; its exclusions scan all published schedules.
    fresh=worker.fresh_seeds();fresh.update(bank=bank,repairs=repairs,guards=guards,preservation=preservation)
    base.atomic_json(directory/'schedules.json',fresh)
    manifest=dict(source=base.source_signature(),files=inputs(directory),source_block=str(source),bank=bank,carry_hits=hits['carry'],fallback_hits=hits['fallback'],
        transform=info,mapping='theta[:,:214]@old_transform + theta[:,214:]@extra_transform; exact zero-added old W',
        calibration_hash=base.sha(base.old_engine.CACHE),old_transform_hash=base.sha(directory/'old-transform.npy'),extra_transform_hash=base.sha(directory/'extra-transform.npy'),
        work_seconds=600,osqp_version=osqp.__version__,old_probe_reclassified_as_TRAIN=True,
        assistance='known own-reward successes; finer existing spatial features; action-margin guards; no new inputs, architecture, BP or oracle')
    base.atomic_json(directory/'manifest.json',manifest)
    print(json.dumps(dict(staged=True,new_observations=0,coordinates=len(matrix),extra=info,bank={p:len(bank[p]) for p in bank},
        carry=base.phases(bank,hits['carry']),fallback=base.phases(bank,hits['fallback']),repairs={p:sum(v['phase']==p for v in repairs) for p in bank},guards=len(guards),preservation_per_phase=16)),flush=True)


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json')
    assert m['source']==base.source_signature() and m['files']==inputs(directory)
    assert m['calibration_hash']==base.sha(base.old_engine.CACHE) and m['old_transform_hash']==base.sha(directory/'old-transform.npy') and m['extra_transform_hash']==base.sha(directory/'extra-transform.npy')
    assert not (directory/'work-clock.json').exists(),'do not restart/reset this counted work clock'
    start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start));deadline=start+600
    theta=np.load(directory/'carry_theta.npy');old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy');matrix=np.vstack([old,extra])
    assert np.array_equal(map_split(theta,old,extra),np.load(directory/'carry.npy'))
    jobs=[]
    for i,repair in enumerate(s['repairs']):
        j=base.job(directory,f'repair_{i:02d}',repair['seed'],repair['weights'],repair['phase'],'train_capture');j['capture_kind']='repair';jobs.append(j)
    for phase in ('left','right'):
        for i,seed in enumerate(s['preservation'][phase]):
            j=base.job(directory,f'preserve_{phase}_{i:02d}',seed,'carry.npy',phase,'train_capture');j['capture_kind']='guard' if seed in s['guards'] else 'preservation';jobs.append(j)
    with worker.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        reports,ok=worker.dispatch(directory,jobs,pool,deadline);assert ok,'partial expanded feature capture; preserve grants/receipts'
        positive=[];preserve=[];guards=[];buffer=[]
        for j in jobs:
            rr=reports[j['id']];assert rr['report']['hits']>0,'rewarded donor/carry success must reproduce'
            with np.load(directory/rr['feature_file']) as v:case=dict(phase=j['phase'],seed=j['seed'],**{k:v[k] for k in v.files})
            if j['capture_kind']=='repair':assert len(case['X']),'successful repair of a deterministic miss needs contrasting actions';positive.append(case)
            else:
                assert np.array_equal(np.argmax(case['X']@theta.T,axis=1),case['actions'])
                preserve.append(case);ix=np.linspace(0,len(case['X'])-1,min(16,len(case['X'])),dtype=int);buffer.append(dict(phase=j['phase'],X=case['X'][ix]))
                if j['capture_kind']=='guard':guards.append(case)
        assert len(guards)==len(s['guards']);X,w=balanced_buffer(buffer);np.savez(directory/'feature-buffer.npz',X=X,weights=w)
        delta,fit=solve_constrained(theta,positive,preserve,guards,X,w)
        z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@matrix.T);_,dose=cap_residual(theta,delta,X,w,z)
        limit,protection=half_margin_bound(theta,delta,guards);alpha=min(dose['alpha'],limit);applied=alpha*delta
        protection.update(alpha_before=dose['alpha'],alpha_after=alpha,applied_function_rms=base.function_rms(applied,z),buffer_disagreement=positive_disagreement(theta,delta,X,w,alpha),guard_choices=preserved_action_fraction(delta,alpha,guards))
        assert protection['guard_choices']==1.;fit.update(action_diagnostics(theta,applied,positive,preserve))
        base.atomic_json(directory/'fit-report.json',dict(fit=fit,dose=dose,protection=protection));base.save_array(directory/'applied_residual.npy',applied)
        proposed=theta+applied;base.save_array(directory/'proposal_theta.npy',proposed);base.save_array(directory/'proposal.npy',map_split(proposed,old,extra))
        js=base.bank_jobs(directory,'proposal',m['bank'],'proposal.npy');rr,ok=worker.dispatch(directory,js,pool,deadline);assert ok,'partial proposal bank check'
        hits=base.hits_for(js,rr);guard_returns=sum(hits[str(seed)] for seed in s['guards']);guard_ok=guard_returns==len(s['guards'])
        contenders=[dict(label='carry',score=base.phases(m['bank'],m['carry_hits'])),dict(label='fallback',score=base.phases(m['bank'],m['fallback_hits']))]
        if guard_ok:contenders.append(dict(label='proposal',score=base.phases(m['bank'],hits)))
        winner=base.choose(contenders);chosen=proposed if winner['label']=='proposal' else theta if winner['label']=='carry' else np.load(directory/'fallback_theta.npy')
        base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',map_split(chosen,old,extra));probe={}
        if base.ah(chosen)!=base.ah(theta):
            for phase,seeds in s['probe'].items():
                js=[base.job(directory,f'probe_{phase}_{label}_{i:02d}',seed,label+'.npy',phase,'phase_probe') for i,seed in enumerate(seeds) for label in ('original_start','carry','selected')]
                rr,ok=worker.dispatch(directory,js,pool,deadline)
                if not ok:break
                probe[phase]={label:[int(rr[f'probe_{phase}_{label}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in ('original_start','carry','selected')}
        result=dict(status='one_spatial_proposal_complete' if ok else 'partial_probe',selected=winner,bank_carry=contenders[0]['score'],bank_proposal=base.phases(m['bank'],hits),
            guard_returns=guard_returns,guard_episodes=len(s['guards']),fit=fit,dose=dose,protection=protection,probe=probe,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start)
        base.atomic_json(directory/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source)
    else:run(a.root)
