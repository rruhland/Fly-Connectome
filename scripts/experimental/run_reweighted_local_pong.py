"""One4x repair-loss refit on unchanged saved own-reward trajectories."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import run_local_repair_pong as prior
from pong_constrained_residual import solve_constrained
from pong_spatial_extension import map_split
from pong_rewarded_residual import cap_residual,action_diagnostics,positive_disagreement
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction,guard_margin_distribution
base=prior.base


def pins():return dict(prior.pins(),**{'run_reweighted_local_pong.py':base.sha(__file__)})


def stage(directory,source,behavior_only=False):
    directory=Path(directory);source=Path(source);sm=base.read(source/'manifest.json');sr=base.read(source/'summary.json')
    assert sr['status']=='one_local_discovery_fit_complete' and sr['selected']['label']=='proposal'
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for name in ('episodes','features','data'):(directory/name).mkdir()
    for target,old in [('fit_start_theta.npy','carry_theta.npy'),('fit_start.npy','carry.npy'),('carry_theta.npy','selected_theta.npy'),('carry.npy','selected.npy')]:
        (directory/target).write_bytes((source/old).read_bytes())
    for name in ('old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json','feature-buffer.npz'):(directory/name).write_bytes((source/name).read_bytes())
    theta=np.load(directory/'fit_start_theta.npy');cases={k:[] for k in ('positive','matching','preservation','guard')}
    def save(c,kind):
        name=f'data/{kind}_{len(cases[kind]):02d}.npz';np.savez(directory/name,**{k:v for k,v in c.items() if k not in ('phase','seed')});cases[kind].append(dict(seed=c['seed'],phase=c['phase'],feature_file=name))
    for item in sm['reused']:
        c=prior.case(source,item);save(c,'preservation')
        if item['kind']=='guard':save(c,'guard')
    witnesses=[];guard_cases=[]
    if (source/'witness-selection.json').exists():witnesses=base.read(source/'witness-selection.json')['entries']
    for path in sorted((source/'episodes').glob('*.json')):
        r=base.read(path);kind=r.get('capture_kind')
        if kind=='guard':c=prior.case(source,r);save(c,'preservation');save(c,'guard')
        elif kind=='donor' and not (source/'witness-selection.json').exists():
            assert r['report']['hits']>0;witnesses.append(r)
    positives=[]
    for item in witnesses:
        c=prior.case(source,item);different=c['actions']!=np.argmax(c['X']@theta.T,axis=1);assert different.any()
        positives.append(dict(seed=c['seed'],phase=c['phase'],X=c['X'][different],actions=c['actions'][different]))
        if (~different).any():save(dict(seed=c['seed'],phase=c['phase'],X=c['X'][~different]),'matching')
    if 'target_consistency' in sr['fit']:
        positives,conflicts=prior.drop_guard_conflicts(positives,[prior.case(directory,v) for v in cases['guard']])
    for c in positives:save(c,'positive')
    expected=base.read(source/'fit-report.json')['fit']
    for kind,field in [('positive','positive_rows'),('matching','matching_preservation_rows'),('preservation','preservation_rows'),('guard','guard_rows')]:
        assert sum(len(prior.case(directory,v)['X']) for v in cases[kind])==expected[field],kind
    carry_hits={}
    for path in (source/'episodes').glob('proposal_*.json'):
        r=base.read(path);carry_hits[str(r['seed'])]=int(r['report']['hits']>0)
    assert len(carry_hits)==sum(map(len,sm['bank'].values()))
    fresh=prior.previous.worker.fresh_seeds();fresh.update(bank=sm['bank'],guards=base.read(source/'schedules.json')['guards'],repair_contexts=[dict(seed=v['seed'],phase=v['phase']) for v in witnesses])
    base.atomic_json(directory/'schedules.json',fresh)
    m=dict(files=pins(),source=base.source_signature(),source_block=str(source),bank=sm['bank'],carry_hits=carry_hits,fit_start_hits=sm['carry_hits'],cases=cases,
        positive_weight=4.,matching_weight=.25,work_seconds=600,behavior_only=behavior_only,guard_margin_q_rms_fraction=1e-4 if behavior_only else None,
        comparison='same saved data and fit start; carried head remains eligible; fresh only carry versus selected; no qualification2 TRAIN import',
        data_hashes={v['feature_file']:base.sha(directory/v['feature_file']) for items in cases.values() for v in items})
    base.atomic_json(directory/'manifest.json',m);print(json.dumps(dict(staged=True,new_observations=0,positive_episodes=len(cases['positive']),guards=len(cases['guard']),bank=base.phases(sm['bank'],carry_hits),cases={k:len(v) for k,v in cases.items()},behavior_only=behavior_only)),flush=True)


def run(directory,technical=False):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json');assert m['source']==base.source_signature()
    assert all(base.sha(directory/name)==h for name,h in m['data_hashes'].items())
    if technical:
        assert (directory/'fit-failure.json').exists() and not (directory/'fit-report.json').exists() and not list((directory/'episodes').glob('*.json'))
        original_failure=directory/'fit-failure-initial.json'
        if not original_failure.exists():original_failure.write_bytes((directory/'fit-failure.json').read_bytes())
        assert all(v==pins()[k] for k,v in m['files'].items() if k!='run_reweighted_local_pong.py')
        start=base.read(directory/'work-clock.json')['started_monotonic'];remaining=start+600-time.monotonic();extension=300 if remaining<200 else 0
        base.atomic_json(directory/'technical-continuation.json',dict(previous_files=m['files'],new_files=pins(),working_set_hash=base.sha(Path(__file__).with_name('pong_working_set_qp.py')),original_failure_hash=base.sha(original_failure),
            original_clock_preserved=True,elapsed_before_continuation=time.monotonic()-start,explicit_extension_seconds=extension,
            reason='authorized technical recovery of the unchanged4xfit after dense solver limit; no new capture/discovery/data'))
    else:
        assert m['files']==pins() and not (directory/'work-clock.json').exists();start=time.monotonic();extension=0
        base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start))
    deadline=start+600+extension
    theta=np.load(directory/'fit_start_theta.npy');old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy')
    assert np.array_equal(map_split(theta,old,extra),np.load(directory/'fit_start.npy'))
    cases={k:[prior.case(directory,v) for v in values] for k,values in m['cases'].items()}
    with np.load(directory/'feature-buffer.npz') as f:X=f['X'];w=f['weights']
    solver=solve_constrained;options={}
    if technical or m.get('behavior_only'):
        from pong_working_set_qp import solve_constrained as solver
        source=Path(m['source_block']);old_alpha=base.read(source/'fit-report.json')['protection']['alpha_after']
        options['initial_delta']=np.load(source/'applied_residual.npy')/old_alpha
    margin_cap=None
    if m.get('behavior_only'):
        margin_cap=1e-4*float(np.sqrt(np.sum(w[:,None]*(X@theta.T)**2)/3));options['guard_margin_cap']=margin_cap
    try:delta,fit=solver(theta,cases['positive'],cases['preservation'],cases['guard'],X,w,matching_preservation=cases['matching'],matching_weight=.25,positive_weight=4.,**options)
    except Exception as error:
        base.atomic_json(directory/'fit-failure.json',dict(error=repr(error),wall_seconds=time.monotonic()-start,new_observations=0,data_preserved=True));raise
    z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@np.vstack([old,extra]).T);_,dose=cap_residual(theta,delta,X,w,z)
    limit,protection=half_margin_bound(theta,delta,cases['guard'],margin_cap=margin_cap);alpha=min(dose['alpha'],limit);applied=alpha*delta
    protection.update(alpha_before=dose['alpha'],alpha_after=alpha,applied_function_rms=base.function_rms(applied,z),buffer_disagreement=positive_disagreement(theta,delta,X,w,alpha),guard_choices=preserved_action_fraction(delta,alpha,cases['guard']))
    assert protection['guard_choices']==1.;fit.update(action_diagnostics(theta,applied,cases['positive'],cases['preservation']))
    fit['guard_margin_distribution']=guard_margin_distribution(delta,alpha,cases['guard'],margin_cap=margin_cap)
    base.atomic_json(directory/'fit-report.json',dict(fit=fit,dose=dose,protection=protection));base.save_array(directory/'applied_residual.npy',applied)
    proposed=theta+applied;base.save_array(directory/'proposal_theta.npy',proposed);base.save_array(directory/'proposal.npy',map_split(proposed,old,extra))
    with prior.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        if np.array_equal(np.load(directory/'proposal.npy'),np.load(directory/'fit_start.npy')):hits=m['fit_start_hits'];ok=True
        else:
            jobs=base.bank_jobs(directory,'proposal',m['bank'],'proposal.npy');rr,ok=prior.dispatch(directory,jobs,pool,deadline);assert ok,'partial4xbankcheck'
            hits=base.hits_for(jobs,rr)
        guard_returns=sum(hits[str(seed)] for seed in s['guards']);repaired={p:sum(hits[str(v['seed'])] for v in s['repair_contexts'] if v['phase']==p) for p in ('left','right')}
        carry_label='carry' if m.get('behavior_only') else 'carry_1x';proposal_label='proposal' if m.get('behavior_only') else 'proposal_4x'
        contenders=[dict(label=carry_label,score=base.phases(m['bank'],m['carry_hits'])),dict(label='fit_start',score=base.phases(m['bank'],m['fit_start_hits']))]
        if guard_returns==len(s['guards']):contenders.append(dict(label=proposal_label,score=base.phases(m['bank'],hits)))
        winner=base.choose(contenders);chosen=proposed if winner['label']==proposal_label else np.load(directory/'carry_theta.npy') if winner['label']==carry_label else theta
        base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',map_split(chosen,old,extra));probe={}
        if winner['label']!=carry_label:
            for phase,seeds in s['probe'].items():
                jobs=[base.job(directory,f'probe_{phase}_{label}_{i:02d}',seed,label+'.npy',phase,'phase_probe') for i,seed in enumerate(seeds) for label in ('carry','selected')]
                rr,ok=prior.dispatch(directory,jobs,pool,deadline)
                if not ok:break
                probe[phase]={label:[int(rr[f'probe_{phase}_{label}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in ('carry','selected')}
        result=dict(status='one_reweighted_fit_complete' if ok else 'partial_probe',selected=winner,bank_carry=contenders[0]['score'],bank_proposal=base.phases(m['bank'],hits),
            integrated_repairs=repaired,guard_returns=guard_returns,guard_episodes=len(s['guards']),fit=fit,dose=dose,protection=protection,probe=probe,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start)
        if m.get('behavior_only') and ok:
            from pong_local_learning_state import load_state
            chosen_hits=hits if winner['label']==proposal_label else m['carry_hits'] if winner['label']==carry_label else m['fit_start_hits']
            state=load_state(Path(m['source_block']));state.update(theta=str(directory/'selected_theta.npy'),weights=str(directory/'selected.npy'),prior_theta=str(directory/'carry_theta.npy'),prior_weights=str(directory/'carry.npy'),
                hits=chosen_hits,prior_hits=m['carry_hits'],probe=probe,probe_seeds=s['probe'])
            base.atomic_json(directory/'learning-state.json',state)
        base.atomic_json(directory/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run','continue'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--behavior-only',action='store_true');a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source,a.behavior_only)
    else:run(a.root,technical=a.mode=='continue')
