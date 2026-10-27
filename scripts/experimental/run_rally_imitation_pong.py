"""One same-basis rewarded-witness fit, followed by ordinary64TRAIN games."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from pong_rally_imitation import fit,diagnostics
from run_rally_baseline_pong import collect_one
from return_metrics import aggregate
import run_practical_pong_v2 as base


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_imitation_pong.py','pong_rally_imitation.py','run_rally_baseline_pong.py','pong_rewarded_residual.py')}


def run(directory,discovery):
    directory=Path(directory);discovery=Path(discovery);dm=base.read(discovery/'manifest.json');ds=base.read(discovery/'summary.json');baseline=Path(dm['baseline'])
    assert ds['status']=='complete' and ds['repaired_contexts']>0 and ds['no_fit_or_candidate_deployed']
    assert not directory.exists();directory.mkdir(parents=True);began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+300
    for name in ('carry.npy','carry_theta.npy','old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json'):(directory/name).write_bytes((discovery/name).read_bytes())
    theta=np.load(directory/'carry_theta.npy');normal_path=base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json';oldfit=base.read(normal_path)
    normalization={k:oldfit[k] for k in ('input_rms','kappa')};positive=[];matching=[];chosen=[]
    for seed in sorted({w['seed'] for w in ds['witnesses']}):
        options=[w for w in ds['witnesses'] if w['seed']==seed];w=min(options,key=lambda w:(w['contrasting_forced_actions'],w['hold'],w['feature_file']));chosen.append(w)
        with np.load(discovery/w['feature_file']) as values:X=values['X'];actions=values['actions'];q=values['q'];forced=values['forced']
        assert X.shape[1]==566 and np.max(abs(X@theta.T-q))<1e-10
        contrast=actions!=np.argmax(q,axis=1);assert np.all(~contrast|forced) and contrast.any()
        group='later_'+w['phase'];positive.append(dict(seed=seed,group=group,X=X[contrast],actions=actions[contrast]))
        if (~contrast).any():matching.append(dict(seed=seed,group=group,X=X[~contrast],actions=actions[~contrast]))
    rows=[base.read(p) for p in sorted((baseline/'episodes').glob('*.json'))];preservation=[];allcases=[]
    for r in rows:
        with np.load(baseline/r['feature_file']) as values:X=values['X'];actions=values['actions']
        phase='left' if base.leftward(r['seed']) else 'right';previous=0
        allcases.append(dict(seed=r['seed'],group=phase,X=X,actions=actions))
        for contact,index in enumerate(i for i,row in enumerate(r['records']) if row['hits']>0):
            group=('first_' if contact==0 else 'later_')+phase
            preservation.append(dict(seed=r['seed'],group=group,X=X[previous:index+1],actions=actions[previous:index+1]));previous=index+1
    delta,report=fit(theta,positive,preservation,matching,normalization);report['whole_baseline_distribution']=diagnostics(theta,delta,allcases)
    report['witness_contexts']=len(chosen);report['successful_segments']=len(preservation);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    proposal=theta+delta;old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy');base.save_array(directory/'proposal_theta.npy',proposal);base.save_array(directory/'proposal.npy',proposal[:,:214]@old+proposal[:,214:]@extra);base.save_array(directory/'applied_delta.npy',delta)
    native=directory/'native_candidate';native.mkdir();(native/'episodes').mkdir();(native/'features').mkdir()
    for source,name in [('proposal.npy','carry.npy'),('proposal_theta.npy','carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json')]:(native/name).write_bytes((directory/source).read_bytes())
    schedule=base.read(baseline/'schedules.json');base.atomic_json(directory/'schedules.json',schedule)
    manifest=dict(files=pins(),source=base.source_signature(),baseline=str(baseline),discovery=str(discovery),normalization_source=str(normal_path),normalization_source_hash=base.sha(normal_path),
        carry_sha256=base.sha(directory/'carry.npy'),proposal_sha256=base.sha(directory/'proposal.npy'),witnesses=chosen,baseline_successful_segments=len(preservation),
        native_rollout_root=str(native),native_rollout_alias='native_candidate/carry.npy is the candidate, named only to reuse the frozen collector; never the preserved control',work_seconds=300,
        positive_weight=1.,broad_soft_weight=1.,matching_soft_weight=.25,ridge=.001,no_hard_actions_or_return_veto=True,no_radial_shrink=True,no_backprop=True,
        decision='No automatic global promotion. Report actual64TRAIN paired returns/firstretention/tails for parentreview; heldout screen separatelyreserved andnotrun.')
    base.atomic_json(directory/'manifest.json',manifest);results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(schedule['train']));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-3:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(collect_one,str(native),seed,i,deadline-2,False)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);results.append(f.result())
            base.atomic_json(directory/'progress.json',dict(native_games=len(results),policy_cameras=sum(r['transitions'] for r in results)));fill()
    aggregate_candidate=aggregate([r['annotation'] for r in results]);before=base.read(baseline/'summary.json');byseed={r['seed']:r for r in results}
    witnesses=[dict(seed=w['seed'],phase=w['phase'],hold=w['hold'],baseline_hits=next(r['report']['hits'] for r in rows if r['seed']==w['seed']),candidate_hits=byseed[w['seed']]['report']['hits']) for w in chosen if w['seed'] in byseed]
    depth={}
    for e in [e for r in results for e in r['opportunity_events'] if e['kind']=='own']:
        d=depth.setdefault(str(e['ordinal']),dict(opportunities=0,returns=0,misses=0));d['opportunities']+=1;d['returns']+=e['success'];d['misses']+=not e['success']
    first=depth.get('1',dict(opportunities=0,returns=0));later_op=sum(v['opportunities'] for k,v in depth.items() if k!='1');later_hit=sum(v['returns'] for k,v in depth.items() if k!='1')
    summary=dict(status='complete' if len(results)==64 and all(r['complete'] for r in results) else 'bounded_partial',candidate_sha256=manifest['proposal_sha256'],carry_immutable=True,
        baseline=before['aggregate'],candidate=aggregate_candidate,contact_depth=depth,later_opportunities=later_op,later_returns=later_hit,later_return_fraction=later_hit/later_op if later_op else None,
        tail={str(k):sum(r['report']['hits']>=k for r in results) for k in range(1,7)},witness_context_outcomes=witnesses,
        witness_extra_return_contexts=sum(w['candidate_hits']>w['baseline_hits'] for w in witnesses),
        first_return_change=first['returns']-before['aggregate']['first_incoming_returns'],train_transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),
        no_fresh_evaluation=True,no_automatic_global_promotion=True,wall_seconds=time.monotonic()-began,source_frozen=manifest['files']==pins() and manifest['source']==base.source_signature())
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--discovery',default='/workspace/m1b-experiments/pong-rally-discovery-20261027-r1');a=p.parse_args();run(a.directory,a.discovery)
