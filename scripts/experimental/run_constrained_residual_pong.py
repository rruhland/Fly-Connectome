"""One constraint-aware preference fit, reusing saved legal learner experience."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import osqp
import run_preserved_rewarded_residual_pong as prior
from pong_constrained_residual import load_saved_cases,solve_constrained
from pong_rewarded_residual import cap_residual,action_diagnostics,positive_disagreement
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction
base=prior.base


def solve(directory,previous):
    directory=Path(directory);previous=Path(previous);directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'work-clock.json').exists(),'do not reset constrained-fit work clock'
    (directory/'episodes').mkdir();(directory/'features').mkdir()
    base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()))
    previous_m=base.read(previous/'manifest.json');old_summary=base.read(previous/'summary.json');assert not old_summary['selected_proposal']
    for name in ('carry_theta.npy','carry.npy','original_start.npy','search-transform.npy','radial-cap.json','feature-buffer.npz'):(directory/name).write_bytes((previous/name).read_bytes())
    theta=np.load(directory/'carry_theta.npy');matrix=np.load(directory/'search-transform.npy')
    positive,preservation,guards=load_saved_cases(previous)
    with np.load(directory/'feature-buffer.npz') as b:X=b['X'];weights=b['weights']
    raw_delta,info=solve_constrained(theta,positive,preservation,guards,X,weights)
    assert info['guard_action_fraction']==1.,'constrained proposal must retain recorded guard choices'
    z=np.asarray(base.old_engine.read_calibration(base.old_engine.CACHE)@matrix.T)
    desired,dose=cap_residual(theta,raw_delta,X,weights,z)
    limit,protection=half_margin_bound(theta,raw_delta,guards);alpha=min(dose['alpha'],limit);delta=alpha*raw_delta
    protection.update(alpha_before=dose['alpha'],alpha_after=alpha,protected_function_rms=base.function_rms(delta,z),protected_buffer_disagreement=positive_disagreement(theta,raw_delta,X,weights,alpha),
        preserved_guard_action_fraction=preserved_action_fraction(raw_delta,alpha,guards))
    assert protection['preserved_guard_action_fraction']==1.
    info.update(action_diagnostics(theta,delta,positive,preservation))
    for name,value in [('raw_residual.npy',raw_delta),('applied_residual.npy',delta),('proposal_theta.npy',theta+delta),('proposal.npy',(theta+delta)@matrix)]:base.save_array(directory/name,value)
    schedule=prior.fresh_seeds();schedule.update(bank=previous_m['bank'],guards=base.read(previous/'schedules.json')['guards']);base.atomic_json(directory/'schedules.json',schedule)
    m=dict(driver_hash=base.sha(__file__),solver_hash=base.sha(Path(__file__).with_name('pong_constrained_residual.py')),prior_driver_hash=base.sha(Path(prior.__file__)),
        source=base.source_signature(),osqp_version=osqp.__version__,previous=str(previous),previous_manifest_hash=base.sha(previous/'manifest.json'),
        carry_hits=previous_m['carry_hits'],bank=previous_m['bank'],work_seconds=600,
        semantics='same 163positive/1152preservation/896guard rows; convex weightedridge with half-margin inequalities, no new capture or BP/class change')
    base.atomic_json(directory/'manifest.json',m)
    report=dict(proposal_ready=True,solver=info,dose=dose,protection=protection,new_camera_observations=0,
        solve_and_preparation_wall_seconds=time.monotonic()-base.read(directory/'work-clock.json')['started_monotonic'])
    base.atomic_json(directory/'solve-report.json',report);print(json.dumps(report),flush=True)


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');assert m['driver_hash']==base.sha(__file__) and m['solver_hash']==base.sha(Path(__file__).with_name('pong_constrained_residual.py')) and m['prior_driver_hash']==base.sha(Path(prior.__file__)) and m['source']==base.source_signature()
    assert not (directory/'evaluation-start.json').exists(),'do not repeat this one-proposal bank test'
    base.atomic_json(directory/'evaluation-start.json',dict(wall_ns=time.time_ns()))
    start=base.read(directory/'work-clock.json')['started_monotonic'];deadline=start+600;s=base.read(directory/'schedules.json');bank=m['bank'];theta=np.load(directory/'carry_theta.npy');proposed=np.load(directory/'proposal_theta.npy');fit=base.read(directory/'solve-report.json')
    with prior.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        if base.ah(theta)==base.ah(proposed):hits=m['carry_hits'];ok=True
        else:
            jobs=base.bank_jobs(directory,'proposal',bank,'proposal.npy');reports,ok=prior.dispatch(directory,jobs,pool,deadline)
            if not ok:raise RuntimeError('partial constrained-proposal bank test; preserve grants/receipts')
            hits=base.hits_for(jobs,reports)
        actual_guards=sum(hits[str(seed)] for seed in s['guards']);guard_ok=actual_guards==len(s['guards'])
        before=base.phases(bank,m['carry_hits']);after=base.phases(bank,hits);selected=bool(guard_ok and tuple(after['key'])>tuple(before['key']))
        chosen=proposed if selected else theta;base.save_array(directory/'selected_theta.npy',chosen);base.save_array(directory/'selected.npy',chosen@np.load(directory/'search-transform.npy'))
        probe={}
        if selected and base.ah(chosen)!=base.ah(theta):
            for p,seeds in s['probe'].items():
                aliases={};seen={};jobs=[]
                for label in ('original_start','carry','selected'):aliases[label]=seen.setdefault(base.sha(directory/(label+'.npy')),label)
                for i,seed in enumerate(seeds):
                    for label in dict.fromkeys(aliases.values()):jobs.append(base.job(directory,f'probe_{p}_{label}_{i:02d}',seed,label+'.npy',p,'phase_probe'))
                reports,ok=prior.dispatch(directory,jobs,pool,deadline)
                if not ok:break
                probe[p]={label:[int(reports[f'probe_{p}_{aliases[label]}_{i:02d}']['report']['hits']>0) for i in range(32)] for label in aliases}
        summary=dict(status='one_proposal_complete' if ok else 'partial_probe',selected_proposal=selected,bank_before=before,bank_proposal=after,
            actual_guard_returns=actual_guards,guard_episodes=len(s['guards']),actual_guard_returns_preserved=guard_ok,fit=fit,probe=probe,
            accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,reused_features_new_observations=0,
            interpretation='one constraint-aware fit in existing linear class; source feature rows reused, actual bank and conditional fresh progress only')
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('solve','run'));p.add_argument('root',type=Path);p.add_argument('--previous',type=Path);a=p.parse_args()
    if a.mode=='solve':solve(a.root,a.previous)
    else:run(a.root)
