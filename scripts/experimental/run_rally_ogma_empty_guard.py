"""One180s evaluation of SAME frozen weights with empty-current-input fallback."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from pong_ogma_empty_guard import EmptyCurrentGuard,CONTRACT
from run_rally_ogma_prior import start,install,metrics,DEST as ORIGINAL,MODEL,BANK,EXPECTED,read,write,sha
from run_rally_repertoire_collection import select_starts,exclusions
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import ReturnCounter,native_count_status
from rally_raw_capture import SparseRows
import run_practical_pong_v2 as base

DEST=ORIGINAL/'empty-current-correction'
CHECKPOINT=ORIGINAL/'final.bin';LIBRARY=ORIGINAL/'libM1BOgmaPrior.so'


def prepare():
    assert not DEST.exists();DEST.mkdir()
    original=read(ORIGINAL/'spec.json');assert read(ORIGINAL/'summary.json')['status']=='rejected_screen8'
    files={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('pong_ogma_empty_guard.py')):sha(Path(__file__).with_name('pong_ogma_empty_guard.py'))}
    files.update(original['source_files'])
    identity=dict(contract=CONTRACT,guard_source_sha256=files[str(Path(__file__).with_name('pong_ogma_empty_guard.py'))],
        evaluator_source_sha256=sha(__file__),checkpoint_sha256=sha(CHECKPOINT),library_sha256=sha(LIBRARY),
        baseline_policy_sha256=EXPECTED,baseline_cap_sha256=sha(MODEL/'radial-cap.json'),engine=base.source_signature())
    policy_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    fresh=DEST/'fresh';fresh.mkdir();drawn=select_starts(1943906127,63185,exclusions(DEST),32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:]);write(fresh/'schedules.json',drawn)
    spec=dict(method='same_final_weights_empty_current_inference_correction',seconds=180,workers=4,checkpoint=str(CHECKPOINT),library=str(LIBRARY),identity=identity,policy_id=policy_id,
        files=files,original_spec_sha256=sha(ORIGINAL/'spec.json'),original_source_frozen=all(sha(p)==h for p,h in original['source_files'].items()),
        screen8=original['screen8'],screen32=original['screen32'],fresh_tag=63185,no_training=True,no_new_warmup=True,no_refit=True,no_automatic_promotion=True,
        override='after every normal temporal step, if all1792selected raw currentvalues==0 choose exact40069 action/Q; feedback actual executed action; keep raw p/residual separate',
        qualification='8must improve total with no later regression;32then needs total/later gains,retainedfirst/second/laterfraction beforefresh;neutral/negative stops',
        pinned_assets={str(p):sha(p) for p in (CHECKPOINT,LIBRARY,Path(BANK),MODEL/'rbf-model.npz',MODEL/'radial-cap.json')})
    assert spec['original_source_frozen'];write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),policy_id=policy_id),flush=True)


def task(directory,seed,index,deadline,policy_id,guarded=True,baseline_receipt=None):
    if time.monotonic()>=deadline:return None
    directory=Path(directory);began=time.monotonic();session=start(seed,deadline)
    if session is None:return None
    runtime=EmptyCurrentGuard(LIBRARY,session.rbf_readout,CHECKPOINT) if guarded else None
    decisions=install(session,runtime,False) if guarded else [];before=runtime.weights() if runtime else None
    game=session.env.game;original=game.step;counter=ReturnCounter();records=[];raw=SparseRows()
    def instrument(drive,**kwargs):return counter.step(game,original,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);r=session.step(check_frozen=False);records.append(r)
            if runtime:runtime.feedback(r['action'],decisions[-1]['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')})
    feature=directory/f'game_{index:02d}-raw.npz';X=raw.matrix();sp.save_npz(feature,X)
    result=dict(seed=seed,complete=session.complete,report=report,annotation=annotation,records=records,decisions=decisions,
        transitions=session.transitions,initial_cameras=session.initial_cameras,feature_file=str(feature),policy_id=policy_id if guarded else EXPECTED,
        model_sha256=sha(CHECKPOINT) if guarded else EXPECTED,transport_kind=CONTRACT if guarded else 'visual-transport-v1',wall_seconds=time.monotonic()-began)
    if runtime:
        assert runtime.weights()==before;runtime.close();audit=EmptyCurrentGuard(LIBRARY,session.rbf_readout,CHECKPOINT)
        for i,(r,expected) in enumerate(zip(records,decisions)):
            lo,hi=X.indptr[i:i+2];got=audit.raw_query(X.indices[lo:hi],X.data[lo:hi]);assert got==expected and got['action']==r['action'] and got['q']==r['q']
            audit.feedback(r['action'],got['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
        assert audit.weights()==before;audit.close()
        prefix=0
        for d in decisions:
            if not d['empty_current_fallback']:break
            prefix+=1
        result.update(sequential_audit_rows=len(records),empty_current_fallback_rows=sum(d['empty_current_fallback'] for d in decisions),
            initial_empty_prefix=prefix,raw_unguarded_changes_suppressed=sum(d['empty_current_fallback'] and d['unguarded_action']!=d['action'] for d in decisions),
            changed_greedy_rows=sum(d['action']!=d['base_action'] for d in decisions))
        if baseline_receipt is not None:
            baseline=read(baseline_receipt)
            for r,expected in zip(records[:prefix],baseline['records'][:prefix]):
                assert all(r[k]==expected[k] for k in ('action','q','hits'))
            result['baseline_prefix_action_Q_hits_exact_rows']=prefix
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(directory/f'game_{index:02d}.json',result);return result


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180))
    bank={e['seed']:e for e in read(BANK)['bank']};candidate=[];summary=dict(status='started',policy_id=spec['policy_id'],new_transition_cameras=0,new_initial_cameras=0)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for count in (8,32):
            folder=DEST/f'screen{count}';folder.mkdir();todo=spec['screen32'][len(candidate):count];got=[]
            for offset in range(0,len(todo),4):
                if time.monotonic()>=deadline-2:break
                futures=[pool.submit(task,str(folder),seed,i,deadline,spec['policy_id'],True,bank[seed]['receipt']) for i,seed in enumerate(todo[offset:offset+4],len(candidate)+offset)]
                got.extend(r for r in [f.result() for f in futures] if r is not None)
            candidate+=got;summary['new_transition_cameras']+=sum(r['transitions'] for r in got);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in got)
            a=metrics([read(bank[s]['receipt']) for s in spec['screen32'][:count]]);b=metrics(candidate);complete=len(candidate)==count and all(r['complete'] for r in candidate)
            result=dict(baseline=a,candidate=b,complete=complete,sequential_audit_rows=sum(r['sequential_audit_rows'] for r in candidate),
                empty_current_fallback_rows=sum(r['empty_current_fallback_rows'] for r in candidate),baseline_prefix_action_Q_hits_exact_rows=sum(r.get('baseline_prefix_action_Q_hits_exact_rows',0) for r in candidate))
            write(folder/'summary.json',result);summary[f'screen{count}']=result
            if not complete or (count==8 and (b['aggregate']['own_returns']<=a['aggregate']['own_returns'] or b['more']['later_returns']<a['more']['later_returns'])):
                summary['status']=f'rejected_screen{count}' if complete else f'screen{count}_partial';break
        else:
            gate=b['aggregate']['own_returns']>a['aggregate']['own_returns'] and b['more']['later_returns']>a['more']['later_returns'] and b['aggregate']['first_incoming_returns']>=a['aggregate']['first_incoming_returns'] and b['more']['tail']['2']>=a['more']['tail']['2'] and b['more']['later_fraction']>=a['more']['later_fraction']
            summary['fresh_gate']=gate
            if gate and time.monotonic()<deadline-2:
                seeds=read(DEST/'fresh/schedules.json')['probe'];groups={'carry':[],'candidate':[]};folders={h:DEST/'fresh'/h for h in groups}
                for folder in folders.values():folder.mkdir()
                jobs=[(head,i,seed) for i,seed in enumerate(seeds) for head in ('carry','candidate')]
                for offset in range(0,len(jobs),4):
                    if time.monotonic()>=deadline-2:break
                    jobslice=jobs[offset:offset+4];futures=[(head,pool.submit(task,str(folders[head]),seed,i,deadline,spec['policy_id'],head=='candidate')) for head,i,seed in jobslice]
                    for head,future in futures:
                        r=future.result()
                        if r is not None:groups[head].append(r)
                for learned,baseline in zip(groups['candidate'],groups['carry']):
                    assert learned['seed']==baseline['seed'];n=learned['initial_empty_prefix']
                    for r,expected in zip(learned['records'][:n],baseline['records'][:n]):assert all(r[k]==expected[k] for k in ('action','q','hits'))
                complete=all(len(g)==32 and all(r['complete'] for r in g) for g in groups.values())
                summary['fresh']=dict(groups={h:metrics(g) for h,g in groups.items()},complete=complete,paired=paired_metrics(groups['carry'],groups['candidate']) if complete else None)
                summary['new_transition_cameras']+=sum(r['transitions'] for g in groups.values() for r in g);summary['new_initial_cameras']+=sum(r['initial_cameras'] for g in groups.values() for r in g);summary['status']='fresh_complete' if complete else 'fresh_partial'
            else:summary['status']='screen32_no_fresh'
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_training=True,no_automatic_promotion=True)
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');args=p.parse_args()
    if args.operation=='prepare':prepare()
    else:run(args.approved_spec_sha256)
