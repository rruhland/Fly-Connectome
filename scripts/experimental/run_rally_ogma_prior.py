"""One600s predictive temporal hierarchy / fixed40069 prior online-learning pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_ogma_prior import OgmaRuntime,TEMPERATURE
from rally_raw_capture import BOOKKEEPING,SparseRows
from return_metrics import ReturnCounter,aggregate,native_count_status
from run_rally_raw_optimizer import more
from run_rally_raw_fresh_screen import paired_metrics
from run_rally_repertoire_collection import exclusions,select_starts
from run_rally_value_collection import ROOT,MODEL,BANK,EXPECTED,read,write,sha
import run_practical_pong_v2 as base

DEST=ROOT/'pong-ogma-prior-20261028-r1'
INDEX=ROOT/'pong-value-collection-20261028-r1/eligible-training-transitions.json'
LIBRARY=Path('/workspace/ogmaneo2-feasibility-c5be9/libM1BOgmaPrior.so')


def start(seed,deadline=None):
    random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING)
    config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,action_hold_frames=1,policy_scaling=read(MODEL/'radial-cap.json'))
    session=FreshRBFEvaluator(load_readout(MODEL/'rbf-model.npz'),config)
    if deadline is not None and time.monotonic()>=deadline:return None
    session._begin_episode();return session


def install(session,runtime,training):
    rows=[]
    def choose():
        v=runtime.query(session,learn_actor=training);rows.append(v);return v['action'],np.asarray(v['q'])
    session._choose_policy=choose
    return rows


def train_game(seed,runtime,deadline,index):
    runtime.reset();session=start(seed,deadline)
    if session is None:return None
    rows=install(session,runtime,True);raw=SparseRows();records=[];rep=runtime.representation();before=runtime.weights();began=time.monotonic()
    while not session.complete and time.monotonic()<deadline:
        raw.add(session);r=session.step(check_frozen=False);records.append(r)
        runtime.feedback(r['action'],rows[-1]['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
    # Last actual observed successor receives its actual reward/terminal update, no camera.
    flush=None
    if records:
        raw.add(session);flush=runtime.query(session,learn_actor=True)
    session.verify_frozen();assert runtime.representation()==rep
    folder=DEST/'training';tag=f'episode_{index:03d}'
    sp.save_npz(folder/(tag+'-raw-next.npz'),raw.matrix())
    report=dict(seed=seed,complete=session.complete,transitions=session.transitions,initials=session.initial_cameras,
        actual_hits=sum(r['hits'] for r in records),records=records,decisions=rows,terminal_or_partial_flush=flush,
        representation_frozen=True,actor_before=before,actor_after=runtime.weights(),
        report=session.summary(),wall_seconds=time.monotonic()-began,raw_file=str(folder/(tag+'-raw-next.npz')))
    write(folder/(tag+'.json'),report);return report


def native_task(directory,seed,index,deadline,checkpoint=None):
    directory=Path(directory);began=time.monotonic()
    if began>=deadline:return None
    session=start(seed,deadline)
    if session is None:return None
    runtime=None;decisions=[]
    if checkpoint is not None:
        runtime=OgmaRuntime(DEST/'libM1BOgmaPrior.so',session.rbf_readout,checkpoint);decisions=install(session,runtime,False);before=runtime.weights();rep=runtime.representation()
    game=session.env.game;original=game.step;counter=ReturnCounter();records=[];raw=SparseRows()
    def instrument(drive,**kwargs):return counter.step(game,original,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);r=session.step(check_frozen=False);records.append(r)
            if runtime is not None:runtime.feedback(r['action'],decisions[-1]['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')})
    result=dict(seed=seed,complete=session.complete,report=report,annotation=annotation,records=records,decisions=decisions,
        transitions=session.transitions,initial_cameras=session.initial_cameras,feature_file=str(directory/f'game_{index:02d}-raw.npz'),
        model_kind='stateful-ogma-prior-v1' if runtime else 'frozen40069',transport_kind='stateful-ogma-prior-v1' if runtime else 'visual-transport-v1',
        model_sha256=sha(checkpoint) if runtime else EXPECTED,wall_seconds=time.monotonic()-began)
    sp.save_npz(result['feature_file'],raw.matrix())
    if runtime:
        assert runtime.weights()==before and runtime.representation()==rep
        # Independent chronological replay from the identical frozen checkpoint.
        runtime.close();audit=OgmaRuntime(DEST/'libM1BOgmaPrior.so',session.rbf_readout,checkpoint);X=raw.matrix()
        for i,r in enumerate(records):
            lo,hi=X.indptr[i:i+2];got=audit.raw_query(X.indices[lo:hi],X.data[lo:hi]);expected=decisions[i]
            assert got==expected and got['action']==r['action'] and got['q']==r['q']
            audit.feedback(r['action'],got['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
        assert audit.weights()==before;audit.close();result['sequential_audit_rows']=len(records)
        result['baseline_greedy_changes']=sum(r['action']!=r['base_action'] for r in decisions)
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(directory/f'game_{index:02d}.json',result);return result


def audit_first_online_game(receipt):
    runtime=OgmaRuntime(DEST/'libM1BOgmaPrior.so',load_readout(MODEL/'rbf-model.npz'),DEST/'online-initial.bin')
    X=sp.load_npz(receipt['raw_file']).tocsr();assert X.shape[0]==receipt['transitions']+1
    for i,(r,expected) in enumerate(zip(receipt['records'],receipt['decisions'])):
        lo,hi=X.indptr[i:i+2];got=runtime.raw_query(X.indices[lo:hi],X.data[lo:hi],learn_actor=True)
        assert got==expected and got['action']==r['action'] and got['q']==r['q']
        runtime.feedback(r['action'],got['p'],r['hits'],r['elapsed_seconds'],r['terminated'])
    lo,hi=X.indptr[-2:];flush=runtime.raw_query(X.indices[lo:hi],X.data[lo:hi],learn_actor=True)
    assert flush==receipt['terminal_or_partial_flush'] and runtime.weights()==receipt['actor_after']
    runtime.close();result=dict(passed=True,rows=receipt['transitions'],actual_probability_action_and_parameter_trajectory_exact=True,real_successor_flush_exact=True,new_cameras=0)
    write(DEST/'first-online-game-audit.json',result);return result


def metrics(rows):return dict(aggregate=aggregate([r['annotation'] for r in rows]),more=more(rows))


def prepare():
    assert not DEST.exists();DEST.mkdir();assert sha(MODEL/'rbf-model.npz')==EXPECTED
    (DEST/'libM1BOgmaPrior.so').write_bytes(LIBRARY.read_bytes())
    bank=read(BANK);first=[e['seed'] for e in read(ROOT/'pong-value-feasibility-20261028-r1/screen8-bank.json')['bank']]
    screen=first+[e['seed'] for e in bank['bank'] if e['seed'] not in first][:24]
    fresh=DEST/'fresh';fresh.mkdir();drawn=select_starts(1943906127,63169,exclusions(DEST),32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:]);write(fresh/'schedules.json',drawn)
    items=read(INDEX)['episodes'];rng=np.random.default_rng(2026102890);order=rng.permutation(len(items)).tolist()
    starts=[j['seed'] for j in read(ROOT/'pong-value-collection-20261028-r1/spec.json')['jobs']]
    training=rng.permutation(starts).tolist()
    sources=[Path(__file__),Path(__file__).with_name('pong_ogma_prior.py'),Path(__file__).with_name('ogmaneo_prior_adapter.cpp')]
    sources+=list(Path(__file__).with_name('vendor').joinpath('ogmaneo2_prior').rglob('*'))
    sources += [Path(__file__).with_name(n) for n in ('pong_fixed_rbf.py','pong_fixed_pairwise.py','rally_raw_capture.py','return_metrics.py','run_practical_pong_v2.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_rally_value_collection.py','ogma_prior_core_test.cpp','test_ogma_prior.py')]
    sources=[p for p in sources if p.is_file()]
    pinned={str(p):sha(p) for p in (INDEX,BANK,MODEL/'rbf-model.npz',MODEL/'proposal.npy',MODEL/'radial-cap.json')}
    for e in items:
        for key in ('raw_file','transition_file','receipt'):
            if key in e:pinned[e[key]]=sha(e[key])
    spec=dict(method='learned_Ogma_temporal_codes_zero_local_residual_over40069',seconds=600,prediction_deadline_seconds=90,online_deadline_seconds=390,evaluation_seconds=180,
        source=base.source_signature(),source_files={str(p):sha(p) for p in sources},library_sha256=sha(DEST/'libM1BOgmaPrior.so'),
        pinned_assets=pinned,
        episodes=items,prediction_execution_order=order,training_starts=training,screen8=first,screen32=screen,
        prior_temperature=TEMPERATURE,actor_alpha=.02,actor_beta=.02,actor_min_steps=8,actor_history_capacity=128,actor_history_iters=8,
        actor_credit='signed TD times saved(actual_action_onehot-actual_behavior_probability); heuristic delayed replay, not unbiased current-policy gradient',
        input='existing capped raw current7planes1792coordinates; channel,y,x -> x,y,channel; frozen vision/map/cap',
        representation='ImageEncoder8x8x16,radius2; hierarchy8/4/2 spatialx16,temporalH4,ticks2; own previous action as context; observationprediction onlywarmup',
        discount='each actual transition .95**elapsed_seconds; natural terminal sets bootstrap0 AFTER reward accumulation; timeout/partial can bootstrap saved successor',
        warmup='chronological executions until90s, final execution may be partial; no actor updates; no expertlabels or reward-value targets; report observedrewardrows separately',
        online='one carried model; representations frozen, runtime state reset each episode; actor trained only on actual own hits/actions/probabilities',
        endpoint='finalonly; intermediate checkpoint at first completed episode after150s online',
        fresh_tag=63169,no_backprop=True,no_privileged_inputs=True,no_expert_labels=True,no_sweep=True,no_automatic_promotion=True,
        license='OgmaNeo2 dependency CC BY-NC-SA4.0, isolated attribution retained')
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    began=time.monotonic();deadline=began+600;write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600))
    assert sha(DEST/'libM1BOgmaPrior.so')==spec['library_sha256']
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['source_files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    runtime=OgmaRuntime(DEST/'libM1BOgmaPrior.so',load_readout(MODEL/'rbf-model.npz'));runtime.save(DEST/'initial-random.bin');warm=[]
    for i in spec['prediction_execution_order']:
        if time.monotonic()>=began+90:break
        item=spec['episodes'][i];runtime.reset();X=sp.load_npz(item['raw_file']).tocsr()
        with np.load(item['transition_file']) as f:actions=f['actions'].copy();hits=f['hits'].copy()
        rows=0
        for j in range(X.shape[0]):
            if time.monotonic()>=began+90:break
            lo,hi=X.indptr[j:j+2];runtime.raw_query(X.indices[lo:hi],X.data[lo:hi],learn_representation=True)
            runtime.feedback(int(actions[j]) if j<len(actions) else 1,np.full(3,1/3),0,0,False);rows+=1
        warm.append(dict(episode=i,rows=rows,complete=rows==X.shape[0],actual_reward_rows_observed=int(hits[:rows].sum())))
        write(DEST/'progress.json',dict(phase='prediction_warmup',executions=len(warm),rows=sum(w['rows'] for w in warm),seconds=time.monotonic()-began))
    runtime.reset();runtime.save(DEST/'online-initial.bin');rep=runtime.representation();write(DEST/'warmup.json',dict(executions=warm,rows=sum(w['rows'] for w in warm),seconds=time.monotonic()-began,representation_fingerprint=rep,actor_updates=0,new_cameras=0))
    (DEST/'training').mkdir();training=[];mid=False
    while time.monotonic()<began+390:
        i=len(training);seed=spec['training_starts'][i%len(spec['training_starts'])]
        r=train_game(seed,runtime,began+390,i)
        if r is None:break
        training.append(r)
        if not mid and time.monotonic()>=began+240:runtime.save(DEST/'online-intermediate.bin');mid=True
        write(DEST/'progress.json',dict(phase='online',games=len(training),cameras=sum(x['transitions'] for x in training),actual_hits=sum(x['actual_hits'] for x in training),seconds=time.monotonic()-began))
    assert runtime.representation()==rep;runtime.reset();runtime.save(DEST/'final.bin');runtime.close()
    summary=dict(status='online_complete',warmup_rows=sum(w['rows'] for w in warm),training_games=len(training),training_transition_cameras=sum(r['transitions'] for r in training),training_initials=sum(r['initials'] for r in training),training_hits=sum(r['actual_hits'] for r in training),evaluation_transition_cameras=0,evaluation_initials=0,final_sha256=sha(DEST/'final.bin'))
    summary['first_online_game_audit']=audit_first_online_game(training[0])
    evaluation_deadline=min(deadline,time.monotonic()+180);bank=read(BANK);entries={e['seed']:e for e in bank['bank']};baseline={s:read(entries[s]['receipt']) for s in spec['screen32']};candidate=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for count in (8,32):
            directory=DEST/f'screen{count}';directory.mkdir();todo=spec['screen32'][len(candidate):count]
            got=[]
            for offset in range(0,len(todo),4):
                if time.monotonic()>=evaluation_deadline-2:break
                futures=[pool.submit(native_task,str(directory),seed,i,evaluation_deadline,str(DEST/'final.bin')) for i,seed in enumerate(todo[offset:offset+4],len(candidate)+offset)]
                got.extend(r for r in [f.result() for f in futures] if r is not None)
            candidate+=got
            summary['evaluation_transition_cameras']+=sum(r['transitions'] for r in got);summary['evaluation_initials']+=sum(r['initial_cameras'] for r in got)
            a=metrics([baseline[s] for s in spec['screen32'][:count]]);b=metrics(candidate);complete=len(candidate)==count and all(r['complete'] for r in candidate)
            result=dict(baseline=a,candidate=b,complete=complete,sequential_audit_rows=sum(r.get('sequential_audit_rows',0) for r in candidate));write(directory/'summary.json',result);summary[f'screen{count}']=result
            if not complete or (count==8 and (b['aggregate']['own_returns']<=a['aggregate']['own_returns'] or b['more']['later_returns']<a['more']['later_returns'])):
                summary['status']=f'rejected_screen{count}' if complete else f'screen{count}_partial';break
        else:
            gate=b['aggregate']['own_returns']>a['aggregate']['own_returns'] and b['more']['later_returns']>a['more']['later_returns'] and b['aggregate']['first_incoming_returns']>=a['aggregate']['first_incoming_returns'] and b['more']['tail']['2']>=a['more']['tail']['2'] and b['more']['later_fraction']>=a['more']['later_fraction']
            summary['fresh_gate']=gate
            if gate and time.monotonic()<evaluation_deadline-2:
                seeds=read(DEST/'fresh/schedules.json')['probe'];groups={}
                groups={'carry':[],'candidate':[]}
                folders={head:DEST/'fresh'/head for head in groups}
                for directory in folders.values():directory.mkdir()
                # At most four in flight: each chunk contains complete adjacent pairs.
                jobs=[(head,i,seed,None if head=='carry' else str(DEST/'final.bin')) for i,seed in enumerate(seeds) for head in ('carry','candidate')]
                for start_index in range(0,len(jobs),4):
                    if time.monotonic()>=evaluation_deadline-2:break
                    chunk=jobs[start_index:start_index+4]
                    futures=[(head,pool.submit(native_task,str(folders[head]),seed,i,evaluation_deadline,checkpoint)) for head,i,seed,checkpoint in chunk]
                    for head,future in futures:
                        result=future.result()
                        if result is not None:groups[head].append(result)
                complete=all(len(v)==32 and all(r['complete'] for r in v) for v in groups.values())
                summary['fresh']=dict(groups={k:metrics(v) for k,v in groups.items()},complete=complete,paired=paired_metrics(groups['carry'],groups['candidate']) if complete else None)
                summary['evaluation_transition_cameras']+=sum(r['transitions'] for v in groups.values() for r in v);summary['evaluation_initials']+=sum(r['initial_cameras'] for v in groups.values() for r in v);summary['status']='fresh_complete' if complete else 'fresh_partial'
            else:summary['status']='screen32_no_fresh'
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),all_native_transition_cameras=summary['training_transition_cameras']+summary['evaluation_transition_cameras'],all_native_initial_cameras=summary['training_initials']+summary['evaluation_initials'],library_frozen=sha(DEST/'libM1BOgmaPrior.so')==spec['library_sha256'],no_automatic_promotion=True,source_frozen=all(sha(p)==h for p,h in spec['source_files'].items()),baseline_assets_unchanged=all(sha(p)==h for p,h in spec['pinned_assets'].items()))
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');args=p.parse_args()
    if args.operation=='prepare':prepare()
    else:run(args.approved_spec_sha256)
