"""Four reward-verified reduced interventions, one incremental raw policy fit.

Existing own-return witnesses supply interventions. Removing a forced chunk is
retained only after native next-return success. No new search actions, expert,
hidden-state targets, hard behavior guards or new policy architecture.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_action_credit import clone_session,step_action
from pong_rally_credit import outcome_from_record
from rally_raw_capture import SparseRows,start
from run_rally_raw_optimizer import build,native_task,more
from pong_rally_raw_margin import fit
from return_metrics import aggregate
import run_practical_pong_v2 as base

RAW=base.EXPERIMENTS/'pong-rally-raw-optimizer-20261027-r1'
DATA=base.EXPERIMENTS/'pong-rally-aggregate-20261027-r1'


def trial(snapshot,mask,command,deadline,max_cameras):
    session=clone_session(snapshot);before=session.episode_hits;initial=session.initial_cameras;start_t=session.transitions;raw=SparseRows();actions=[];contrasting=[];reason='camera_cap_censored';outcome=None
    for i in range(max_cameras):
        if time.monotonic()>=deadline:reason='wall_deadline_censored';break
        raw.add(session);a,q=session._choose_policy();forced=i<len(mask) and mask[i]
        r=step_action(session,int(command)) if forced else session.step(check_frozen=False)
        actions.append(r['action']);contrasting.append(bool(forced and a!=command))
        outcome,ending=outcome_from_record(r)
        if ending is not None:reason=ending;break
    session.verify_frozen();assert session.initial_cameras==initial
    assert outcome!=1 or session.episode_hits-before==1
    return dict(outcome=outcome,reason=reason,cameras=session.transitions-start_t,forced_requested=int(mask.sum()),contrasting_overrides=int(sum(contrasting)),actions=np.array(actions,dtype=int),X=raw.matrix())


def prune_task(directory,case,index,deadline):
    directory=Path(directory);began=time.monotonic();W=np.load(directory/'carry.npy');receipt=base.read(case['receipt']);query=case['transition'];session=start(W,receipt['config'])
    for expected in receipt['records'][:query]:
        if time.monotonic()>=deadline:raise TimeoutError('prefix deadline')
        assert session.step(check_frozen=False)==expected
    snapshot=clone_session(session);session.verify_frozen();prefix=query;max_cameras=receipt['transitions']-query
    with np.load(case['memory']['feature_path']) as f:original_actions=f['actions']; original_forced=f['forced']
    hold=int(np.sum(original_forced[query:]));command=int(original_actions[query]);mask=np.ones(hold,dtype=bool);trials=[];spent=0
    best=trial(snapshot,mask,command,deadline,max_cameras);spent+=best['cameras'];assert best['outcome']==1 and np.array_equal(best['actions'],original_actions[query:])
    trials.append({k:best[k] for k in ('outcome','reason','cameras','forced_requested','contrasting_overrides')});trials[-1]['kind']='verify_original';original_contrasts=best['contrasting_overrides'];best_mask=mask.copy();visited={mask.tobytes()};accepted=0
    # Greedy chunk deletion, late chunks first. An empty mask is the known native
    # miss already present in the immutable current receipt, so never re-pay it.
    for divisions in (2,4,8):
        boundaries=np.linspace(0,hold,divisions+1,dtype=int)
        for chunk in reversed(range(divisions)):
            if len(trials)>=8 or time.monotonic()>=deadline:break
            candidate=best_mask.copy();candidate[boundaries[chunk]:boundaries[chunk+1]]=False
            if not candidate.any() or candidate.tobytes() in visited:continue
            visited.add(candidate.tobytes());r=trial(snapshot,candidate,command,deadline,max_cameras);spent+=r['cameras'];success=r['outcome']==1
            trials.append(dict(kind='delete_chunk',start=int(boundaries[chunk]),stop=int(boundaries[chunk+1]),retained=success,**{k:r[k] for k in ('outcome','reason','cameras','forced_requested','contrasting_overrides')}))
            if success:best=r;best_mask=candidate;accepted+=1
        if len(trials)>=8 or time.monotonic()>=deadline:break
    prefix_X=sp.load_npz(case['prefix_raw'])[:query];full=sp.vstack([prefix_X,best['X']],format='csr');actions=np.r_[[r['action'] for r in receipt['records'][:query]],best['actions']]
    Xpath=f'witnesses/case_{index:02d}.npz';meta=f'witnesses/case_{index:02d}_actions.npy';sp.save_npz(directory/Xpath,full);base.save_array(directory/meta,actions)
    hits=[i for i,r in enumerate(receipt['records'][:query]) if r['hits']]+[full.shape[0]-1]
    result=dict(seed=case['seed'],phase=case['phase'],role=case['role'],transition=query,source='native-reward-verified chunk deletion of current burst witness',path=Xpath,actions_path=meta,hit_indices=hits,target_hits=len(hits),
        original_hold=hold,remaining_forced=int(best_mask.sum()),original_contrasting_overrides=original_contrasts,remaining_contrasting_overrides=best['contrasting_overrides'],accepted_deletions=accepted,trials=trials,
        transition_cameras=prefix+spent,prefix_cameras=prefix,suffix_cameras=spent,initial_cameras=session.initial_cameras,wall_seconds=time.monotonic()-began,complete=best['outcome']==1)
    base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result


def cases(raw_manifest):
    byseed={r['seed']:r for r in raw_manifest['rows']};witnesses={w['seed']:w for w in base.read(DATA/'discovery-result.json')['witnesses']};outputs=base.read(RAW/'capture-summary.json')['outputs'];prefix={o['seed']:str(RAW/o['path']) for o in outputs if o['kind']=='current'};selected=[]
    for role in ('opening','later'):
        for phase in ('left','right'):
            choices=[]
            for memory in raw_manifest['memories']:
                if memory['source']!='current_burst_success' or memory['phase']!=phase:continue
                r=byseed[memory['seed']]
                if (r['report']['hits']>0)!=(role=='later'):continue
                with np.load(memory['feature_path']) as f:cost=int(np.sum(f['forced']&(f['actions']!=np.argmax(f['q'],axis=1))))
                choices.append((cost,memory['seed'],memory))
            cost,seed,memory=min(choices,key=lambda v:v[:2]);r=byseed[seed]
            selected.append(dict(seed=seed,role=role,phase=phase,contrasting_overrides=cost,receipt=r['receipt'],transition=witnesses[seed]['transition'],memory=memory,prefix_raw=prefix[seed]))
    return selected


def run(directory):
    directory=Path(directory);assert not directory.exists();directory.mkdir();(directory/'cases').mkdir();(directory/'witnesses').mkdir()
    for name in ('carry.npy','radial-cap.json'):(directory/name).write_bytes((RAW/name).read_bytes())
    raw_manifest=base.read(RAW/'manifest.json');assert base.read(RAW/'completion-verification.json')['passed'];selected=cases(raw_manifest)
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_incremental_pruned.py','pong_rally_raw_margin.py','run_rally_raw_optimizer.py','rally_raw_capture.py','pong_action_credit.py')}
    m=dict(source=base.source_signature(),files=files,carry_sha256=base.sha(directory/'carry.npy'),raw_data_source=str(RAW),cases=selected,work_seconds=420,max_trials_per_case=8,
        positive_memory='only four chosen current burst recoveries; all other requested changes deferred',preservation='all197 current earned successful segments, soft',no_new_search=True,no_holdout_fitted=True,no_backprop=True)
    base.atomic_json(directory/'manifest.json',m);began=time.monotonic();deadline=began+420;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline));results=[]
    print(json.dumps(dict(launched=True,cases=[{k:c[k] for k in ('seed','role','phase','contrasting_overrides')} for c in selected],seconds=420)),flush=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(prune_task,str(directory),case,i,min(deadline-180,began+150)) for i,case in enumerate(selected)]
        for f in futures:results.append(f.result())
    base.atomic_json(directory/'pruning-summary.json',dict(cases=results,transition_cameras=sum(r['transition_cameras'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),all_complete=all(r['complete'] for r in results)))
    W,unused,preservation,unused_matching,qerror=build(RAW);positive=[];matching=[]
    for r in results:
        X=sp.load_npz(directory/r['path']);actions=np.load(directory/r['actions_path']);contrast=actions!=np.argmax(X@W.T,axis=1);start_t=0
        for ordinal,end in enumerate(r['hit_indices']):
            segment=np.arange(start_t,end+1);start_t=end+1;group=('first_' if ordinal==0 else 'later_')+r['phase']
            for mask,target in ((contrast,positive),(~contrast,matching)):
                index=segment[mask[segment]]
                if len(index):target.append(dict(seed=r['seed'],group=group,X=X[index],actions=actions[index]))
    normal0=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:normal0[k] for k in ('input_rms','kappa')}
    delta,report=fit(W,positive,preservation,matching,normal,deadline=min(deadline-150,time.monotonic()+60),ridge=1e-6)
    base.save_array(directory/'applied_delta.npy',delta);base.save_array(directory/'proposal.npy',W+delta);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    outcomes=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(r['seed'] for r in raw_manifest['rows']));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory),seed,i,deadline-1)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);outcomes.append(f.result())
            base.atomic_json(directory/'progress.json',dict(phase='native96',games=len(outcomes),cameras=sum(r['transitions'] for r in outcomes)));fill()
    byseed={r['seed']:r for r in outcomes};control={r['seed']:r for r in raw_manifest['rows']}
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=aggregate([r['annotation'] for r in raw_manifest['rows']]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(raw_manifest['rows']),candidate_more=more(outcomes),
        repaired_contexts_extra_returns=sum(byseed[r['seed']]['report']['hits']>control[r['seed']]['report']['hits'] for r in results if r['seed'] in byseed),repair_outcomes=[dict(seed=r['seed'],role=r['role'],baseline_hits=control[r['seed']]['report']['hits'],candidate_hits=byseed[r['seed']]['report']['hits'] if r['seed'] in byseed else None) for r in results],
        pruning_transition_cameras=sum(r['transition_cameras'] for r in results),pruning_initials=sum(r['initial_cameras'] for r in results),native_transition_cameras=sum(r['transitions'] for r in outcomes),native_initials=sum(r['initial_cameras'] for r in outcomes),wall_seconds=time.monotonic()-began,
        carry_immutable=base.sha(directory/'carry.npy')==m['carry_sha256'],source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files} and m['source']==base.source_signature(),proposal_sha256=base.sha(directory/'proposal.npy'),no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();run(a.directory)
