"""One same-reward-data raw-coordinate fit and native96 comparison.

Uses the immutable capture clock. Capture and fitting source pins are separate,
so writing this fitter while capture executes does not revise a running worker.
"""
import argparse,json,multiprocessing as mp,random,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from pong_rally_raw_margin import fit
from rally_raw_capture import SparseRows,BOOKKEEPING,task as capture_task
from return_metrics import ReturnCounter,aggregate,native_count_status
import run_practical_pong_v2 as base


def build(directory):
    m=base.read(directory/'manifest.json');capture=base.read(directory/'capture-summary.json');assert capture['status']=='complete'
    outputs=list(capture['outputs'])
    if (directory/'collision-recovery.json').exists(): outputs+=base.read(directory/'collision-recovery.json')['outputs']
    paths={(o['kind'],o['seed'],o.get('source')):directory/o['path'] for o in outputs};W=np.load(directory/'carry.npy');positive=[];matching=[];preservation=[]
    source=Path(m['data_source']);oldtheta=np.load(source/'carry_theta.npy');raw_q_error=0.
    for memory in m['memories']:
        X=sp.load_npz(paths['memory',memory['seed'],memory['source']]);
        with np.load(memory['feature_path']) as f:actions=f['actions'][:X.shape[0]];projected=f['X'][:X.shape[0]]
        assert X.shape[0]==memory.get('slice_end',len(actions))
        q=X@W.T;raw_q_error=max(raw_q_error,float(np.max(abs(q-projected@oldtheta.T))));contrast=actions!=np.argmax(q,axis=1);start=0
        for ordinal,end in enumerate(memory['hit_indices']):
            segment=np.arange(start,end+1);start=end+1;group=('first_' if ordinal==0 else 'later_')+memory['phase']
            for mask,target in ((contrast,positive),(~contrast,matching)):
                index=segment[mask[segment]]
                if len(index):target.append(dict(seed=memory['seed'],group=group,X=X[index],actions=actions[index]))
    for r in m['rows']:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if not hits:continue
        X=sp.load_npz(paths['current',r['seed'],None]);actions=np.array([v['action'] for v in r['records']]);start=0
        for ordinal,end in enumerate(hits):
            group=('first_' if ordinal==0 else 'later_')+('left' if base.leftward(r['seed']) else 'right')
            preservation.append(dict(seed=r['seed'],group=group,X=X[start:end+1],actions=actions[start:end+1]));start=end+1
    assert raw_q_error<1e-10
    return W,positive,preservation,matching,raw_q_error


def native_task(directory,seed,index,deadline):
    directory=Path(directory);began=time.monotonic();W=np.load(directory/'proposal.npy')
    config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=base.read(directory/'radial-cap.json'))
    random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING)
    model=directory/'rbf-model.npz';model_hash=base.sha(model) if model.exists() else base.sha(directory/'proposal.npy')
    if model.exists():
        from pong_fixed_rbf import load_readout,FreshRBFEvaluator
        readout=load_readout(model);assert np.array_equal(readout.base.weights,W);session=FreshRBFEvaluator(readout,config)
    else:session=FreshPongEvaluator(W,config)
    game=session.env.game;original=game.step;counter=ReturnCounter();records=[];events=[];raw=SparseRows()
    def instrument(drive,**kwargs):
        own=counter.opportunities;opponent=counter.opponent_opportunities;tick=int(game.rally_steps[0])+1
        out=counter.step(game,original,drive,**kwargs)
        if counter.opportunities>own:events.append(dict(kind='own',ordinal=counter.opportunities,success=bool(out.hits[0]),tick=tick,transition=session.transitions,completed_own_hits=counter.returns))
        if counter.opponent_opportunities>opponent:events.append(dict(kind='opponent',ordinal=counter.opponent_opportunities,success=float(game.ball_velocity[0,0])<0,tick=tick,transition=session.transitions,after_own_hits=counter.returns))
        return out
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            if session.initial_cameras==session.episodes:session._begin_episode()
            raw.add(session);records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,truncated=last['truncated'],terminated=last['terminated'],reward=last['reward'],seconds=last['seconds'])
    count_status=native_count_status(annotation,report)
    path=f'native_candidate/features/train_{index:02d}.npz';sp.save_npz(directory/path,raw.matrix())
    result=dict(seed=seed,complete=report['complete'],config=asdict(config),weights_sha256=model_hash,report=report,annotation=annotation,annotation_count_status=count_status,records=records,opportunity_events=events,feature_file=path,feature_format='actual capped raw CSR',transitions=session.transitions,initial_cameras=session.initial_cameras,learned_contents_unchanged=True,wall_seconds=time.monotonic()-began)
    if model.exists():result.update(model_kind='fixed-local-rbf-residual-v1',model_file=str(model),base_weights_sha256=base.sha(directory/'proposal.npy'))
    base.atomic_json(directory/'native_candidate'/'episodes'/f'train_{index:02d}.json',result)
    if report['complete']:assert count_status['valid_terminal_counts'],(seed,count_status,annotation,report)
    return result


def more(values):
    a=aggregate([r['annotation'] for r in values]);later=a['own_returns']-a['first_incoming_returns'];opps=a['own_opportunities']-a['first_incoming_opportunities']
    return dict(later_returns=later,later_opportunities=opps,later_fraction=later/opps if opps else None,tail={str(k):sum(r['report']['hits']>=k for r in values) for k in range(1,8)},timeouts=sum(r['report']['truncations'] for r in values),opponent_ended=sum(r['report']['wins'] for r in values))


def repair_collisions(directory, manifest, deadline):
    # Original capture remains immutable. Duplicate historical memories are
    # separately replayed with source-qualified paths, all extra cameras paid.
    counts={}
    for memory in manifest['memories']: counts[memory['seed']]=counts.get(memory['seed'],0)+1
    collided={seed for seed,n in counts.items() if n>1}
    jobs=[j for j in manifest['jobs'] if j['kind']=='old_memory' and j['seed'] in collided]
    assert len(jobs)==4 and len(collided)==2
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[]
        for j in jobs:
            sub=directory/'collision_recovery'/j['memory']['source']; (sub/'capture').mkdir(parents=True,exist_ok=True);(sub/'capture_receipts').mkdir(exist_ok=True)
            for name in ('carry.npy','radial-cap.json'):(sub/name).write_bytes((directory/name).read_bytes())
            futures.append((pool.submit(capture_task,str(sub),j,deadline-1),sub))
        for future,sub in futures:
            r=future.result()
            for o in r['outputs']:o['path']=str((sub/o['path']).relative_to(directory))
            results.append(r)
    report=dict(reason='two historical seeds each have distinct old burst and first-return memories; original seed-only filenames collided',original_capture_preserved=True,outputs=[o for r in results for o in r['outputs']],transition_cameras=sum(r['cameras'] for r in results),initial_cameras=sum(r['initials'] for r in results),all_complete=all(o['complete'] for r in results for o in r['outputs']))
    assert report['all_complete'];base.atomic_json(directory/'collision-recovery.json',report)


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');clock=base.read(directory/'work-clock.json');deadline=clock['deadline_monotonic']
    assert m['source']==base.source_signature() and base.sha(directory/'carry.npy')==m['carry_sha256'];assert not (directory/'fit-stage.json').exists()
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_raw_optimizer.py','pong_rally_raw_margin.py','rally_raw_capture.py')}
    base.atomic_json(directory/'fit-stage.json',dict(files=files,source=base.source_signature(),original_clock=clock,ridge=1e-6,no_backprop=True,no_new_search=True))
    repair_collisions(directory,m,deadline)
    W,positive,preservation,matching,qerror=build(directory)
    original=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:original[k] for k in ('input_rms','kappa')}
    delta,report=fit(W,positive,preservation,matching,normal,deadline=min(deadline-160,time.monotonic()+120),ridge=1e-6)
    report['raw_vs_projected_carried_q_maxerror']=qerror;report['zero_residual_bytes_exact']=np.array_equal(W+np.zeros_like(W),np.load(directory/'carry.npy'))
    base.save_array(directory/'applied_delta.npy',delta);base.save_array(directory/'proposal.npy',W+delta);base.atomic_json(directory/'fit-report.json',report)
    print(json.dumps(dict(fit=report)),flush=True)
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    started=time.monotonic();base.atomic_json(directory/'native-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=started,deadline_monotonic=deadline));outcomes=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(r['seed'] for r in m['rows']));active={}
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
    capture=base.read(directory/'capture-summary.json'); recovery=base.read(directory/'collision-recovery.json');byseed={r['seed']:r for r in outcomes};baseline={r['seed']:r for r in m['rows']}
    witness_new=[v for v in m['memories'] if v['source']=='current_burst_success']
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=aggregate([r['annotation'] for r in m['rows']]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(m['rows']),candidate_more=more(outcomes),
        new_witness_extra_returns=sum(byseed[w['seed']]['report']['hits']>baseline[w['seed']]['report']['hits'] for w in witness_new if w['seed'] in byseed),capture_transition_cameras=capture['transition_cameras']+recovery['transition_cameras'],capture_initials=capture['initial_cameras']+recovery['initial_cameras'],collision_recovery_cameras=recovery['transition_cameras'],native_transition_cameras=sum(r['transitions'] for r in outcomes),native_initials=sum(r['initial_cameras'] for r in outcomes),wall_seconds=time.monotonic()-clock['started_monotonic'],original_clock_preserved=True,carry_immutable=base.sha(directory/'carry.npy')==m['carry_sha256'],source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files} and m['source']==base.source_signature(),proposal_sha256=base.sha(directory/'proposal.npy'),no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();run(a.directory)
