"""Reusable current-policy burst discovery, reward pruning and small raw update.

Current earned segments are soft memory. At most two failed approaches per
opening/later x initial-direction group are searched; one minimal witnessed
repair per group is fitted. Every discovery action is judged by native next hit.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_action_credit import clone_session
from rally_raw_capture import start
from pong_rally_raw_margin import fit
from run_rally_incremental_pruned import trial
from run_rally_raw_optimizer import native_task,more
from return_metrics import aggregate
import run_practical_pong_v2 as base


def current_rows(source):
    source=Path(source);summary=base.read(source/'summary.json');assert summary['status']=='complete';W=np.load(source/'proposal.npy');weight_hash=base.sha(source/'proposal.npy');rows=[]
    for path in sorted((source/'native_candidate/episodes').glob('*.json')):
        r=base.read(path);assert r['complete'] and r['weights_sha256']==weight_hash
        cfg=r['config'];assert cfg['episodes']==1 and cfg['max_episode_ticks']==2400 and cfg['opponent_speed']==.65 and cfg['action_hold_frames']==1 and cfg['policy_kind']=='trained'
        assert cfg['action_seed'] is None and not cfg['blank_camera'] and not cfg['skip_baseline_perception'] and cfg['policy_scaling']==base.read(source/'radial-cap.json')
        r.update(receipt=str(path),raw_feature=str(source/r['feature_file']));rows.append(r)
    assert len(rows)==96 and aggregate([r['annotation'] for r in rows])==summary['candidate']
    return W,rows


def choose_cases(rows):
    groups=[]
    for role in ('opening','later'):
        for phase in ('left','right'):
            eligible=[r for r in rows if r['report']['losses'] and (r['report']['hits']>0)==(role=='later') and base.leftward(r['seed'])==(phase=='left')]
            # Ordinary misses ordered by seed; no oracle landing/hidden-state ranking.
            selected=sorted(eligible,key=lambda r:r['seed'])[:2];group=[]
            for r in selected:
                last=max((i for i,v in enumerate(r['records']) if v['hits']),default=-1);floor=max(last+1,5 if role=='opening' else 0)
                positions=sorted({max(floor,r['transitions']-lead) for lead in (40,80,120) if max(floor,r['transitions']-lead)<r['transitions']})
                group.append(dict(seed=r['seed'],phase=phase,role=role,receipt=r['receipt'],raw_feature=r['raw_feature'],positions=positions))
            groups.append(group)
    return [g[i] for i in range(2) for g in groups if len(g)>i]


def discover_task(directory,case,index,deadline):
    directory=Path(directory);began=time.monotonic();receipt=base.read(case['receipt']);session=start(np.load(directory/'carry.npy'),receipt['config']);snapshots={};stop=max(case['positions']);trials=[];spent=0;witness=None
    for expected in receipt['records'][:stop]:
        if time.monotonic()>=deadline:break
        if session.transitions in case['positions']:snapshots[session.transitions]=clone_session(session)
        assert session.step(check_frozen=False)==expected
    if session.transitions==stop:snapshots[stop]=clone_session(session)
    prefix=session.transitions;session.verify_frozen();raw_prefix=sp.load_npz(case['raw_feature'])
    for hold in (20,40,80):
        if witness is not None:break
        for t in case['positions']:
            if witness is not None or time.monotonic()>=deadline:break
            if t not in snapshots:continue
            n=receipt['transitions']-t
            for command in range(3):
                if time.monotonic()>=deadline:break
                mask=np.ones(min(hold,n),dtype=bool);r=trial(snapshots[t],mask,command,deadline,n);spent+=r['cameras']
                trials.append(dict(kind='discovery',transition=t,hold=hold,command=command,**{k:r[k] for k in ('outcome','reason','cameras','forced_requested','contrasting_overrides')}))
                if r['outcome']!=1:continue
                best=r;best_mask=mask;visited={mask.tobytes()};deletions=[]
                for divisions in (2,4,8):
                    boundaries=np.linspace(0,len(mask),divisions+1,dtype=int)
                    for chunk in reversed(range(divisions)):
                        if len(deletions)>=7 or time.monotonic()>=deadline:break
                        proposal=best_mask.copy();proposal[boundaries[chunk]:boundaries[chunk+1]]=False
                        if not proposal.any() or proposal.tobytes() in visited:continue
                        visited.add(proposal.tobytes());v=trial(snapshots[t],proposal,command,deadline,n);spent+=v['cameras'];accepted=v['outcome']==1
                        deletions.append(dict(kind='delete_chunk',start=int(boundaries[chunk]),stop=int(boundaries[chunk+1]),retained=accepted,**{k:v[k] for k in ('outcome','reason','cameras','forced_requested','contrasting_overrides')}))
                        if accepted:best=v;best_mask=proposal
                    if len(deletions)>=7 or time.monotonic()>=deadline:break
                trials+=deletions;X=sp.vstack([raw_prefix[:t],best['X']],format='csr');actions=np.r_[[v['action'] for v in receipt['records'][:t]],best['actions']].astype(np.int64)
                path=f'witnesses/case_{index:02d}.npz';apath=f'witnesses/case_{index:02d}_actions.npy';sp.save_npz(directory/path,X);np.save(directory/apath,actions,allow_pickle=False)
                hits=[i for i,v in enumerate(receipt['records'][:t]) if v['hits']]+[X.shape[0]-1]
                witness=dict(seed=case['seed'],phase=case['phase'],role=case['role'],transition=t,hold=hold,command=command,path=path,actions_path=apath,hit_indices=hits,target_hits=len(hits),original_contrasting_overrides=r['contrasting_overrides'],contrasting_overrides=best['contrasting_overrides'],remaining_forced=int(best_mask.sum()),deletions_retained=sum(v['retained'] for v in deletions))
                break
    result=dict(seed=case['seed'],phase=case['phase'],role=case['role'],witness=witness,trials=trials,prefix_cameras=prefix,suffix_cameras=spent,initial_cameras=session.initial_cameras,complete=prefix==stop and (witness is not None or time.monotonic()<deadline),wall_seconds=time.monotonic()-began)
    base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result


def dataset(directory,W,rows,witnesses):
    positive=[];matching=[];preservation=[]
    for r in rows:
        X=sp.load_npz(r['raw_feature']);actions=np.array([v['action'] for v in r['records']],dtype=np.int64);start_t=0;phase='left' if base.leftward(r['seed']) else 'right'
        assert X.shape==(len(actions),4389)
        for ordinal,end in enumerate(i for i,v in enumerate(r['records']) if v['hits']):
            preservation.append(dict(seed=r['seed'],group=('first_' if ordinal==0 else 'later_')+phase,X=X[start_t:end+1],actions=actions[start_t:end+1]));start_t=end+1
    for witness in witnesses:
        X=sp.load_npz(directory/witness['path']);actions=np.load(directory/witness['actions_path']);assert np.issubdtype(actions.dtype,np.integer) and (actions>=0).all() and (actions<=2).all()
        contrast=actions!=np.argmax(X@W.T,axis=1);start_t=0
        for ordinal,end in enumerate(witness['hit_indices']):
            segment=np.arange(start_t,end+1);start_t=end+1;group=('first_' if ordinal==0 else 'later_')+witness['phase']
            for mask,target in ((contrast,positive),(~contrast,matching)):
                ix=segment[mask[segment]]
                if len(ix):target.append(dict(seed=witness['seed'],group=group,X=X[ix],actions=actions[ix]))
    return positive,preservation,matching


def run(directory,source):
    directory=Path(directory);source=Path(source);assert not directory.exists();W,rows=current_rows(source);selected=choose_cases(rows);assert selected
    directory.mkdir();(directory/'cases').mkdir();(directory/'witnesses').mkdir()
    for old,new in [('proposal.npy','carry.npy'),('radial-cap.json','radial-cap.json')]:(directory/new).write_bytes((source/old).read_bytes())
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_incremental_cycle.py','run_rally_incremental_pruned.py','pong_rally_raw_margin.py','run_rally_raw_optimizer.py','rally_raw_capture.py','pong_action_credit.py')}
    manifest=dict(source=base.source_signature(),files=files,source_root=str(source),carry_sha256=base.sha(directory/'carry.npy'),rows=[r['receipt'] for r in rows],cases=selected,work_seconds=420,discovery_seconds=180,fit_seconds=60,positive_memory='one clean reward-earned recovery per available opening/later x phase group',soft_current_success_segments=sum(r['report']['hits'] for r in rows),no_holdout_fitted=True,no_backprop=True,policy_class='unchanged capped raw4389 ->3linear greedy actions')
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();deadline=began+420;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline));results=[]
    print(json.dumps(dict(launched=True,source_root=str(source),cases=len(selected),earned_segments=manifest['soft_current_success_segments'],seconds=420)),flush=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(selected));active={}
        def fill():
            while len(active)<4 and time.monotonic()<began+178:
                try:i,case=next(pending)
                except StopIteration:break
                active[pool.submit(discover_task,str(directory),case,i,began+180)]=(i,case)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);results.append(f.result())
            base.atomic_json(directory/'progress.json',dict(phase='discovery',cases=len(results),recovered=sum(r['witness'] is not None for r in results),cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results)));fill()
    discovered=[r['witness'] for r in results if r['witness']];chosen=[]
    for role in ('opening','later'):
        for phase in ('left','right'):
            group=[w for w in discovered if w['role']==role and w['phase']==phase]
            if group:chosen.append(min(group,key=lambda w:(w['contrasting_overrides'],w['remaining_forced'],w['seed'])))
    discovery=dict(cases=results,witnesses=discovered,selected_witnesses=chosen,transition_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),censored_cases=sum(not r['complete'] for r in results))
    base.atomic_json(directory/'discovery-summary.json',discovery)
    if not chosen:
        base.atomic_json(directory/'summary.json',dict(status='no_rewarded_repairs',discovery=discovery,wall_seconds=time.monotonic()-began,no_model_change=True));return
    positive,preservation,matching=dataset(directory,W,rows,chosen);normal0=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:normal0[k] for k in ('input_rms','kappa')}
    delta,report=fit(W,positive,preservation,matching,normal,deadline=min(deadline-150,time.monotonic()+60),ridge=1e-6);base.save_array(directory/'applied_delta.npy',delta);base.save_array(directory/'proposal.npy',W+delta);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    outcomes=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(r['seed'] for r in rows));active={}
        def fill_native():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory),seed,i,deadline-1)]=(i,seed)
        fill_native()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);outcomes.append(f.result())
            base.atomic_json(directory/'progress.json',dict(phase='native96',games=len(outcomes),cameras=sum(r['transitions'] for r in outcomes)));fill_native()
    byseed={r['seed']:r for r in outcomes};control={r['seed']:r for r in rows}
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=aggregate([r['annotation'] for r in rows]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(rows),candidate_more=more(outcomes),
        repair_outcomes=[dict(seed=w['seed'],role=w['role'],baseline_hits=control[w['seed']]['report']['hits'],candidate_hits=byseed[w['seed']]['report']['hits'] if w['seed'] in byseed else None) for w in chosen],discovery_contexts=len(discovered),selected_repairs=len(chosen),
        discovery_transition_cameras=discovery['transition_cameras'],discovery_initials=discovery['initial_cameras'],native_transition_cameras=sum(r['transitions'] for r in outcomes),native_initials=sum(r['initial_cameras'] for r in outcomes),wall_seconds=time.monotonic()-began,
        carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files} and manifest['source']==base.source_signature(),proposal_sha256=base.sha(directory/'proposal.npy'),no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--source-root',required=True);a=p.parse_args();run(a.directory,a.source_root)
