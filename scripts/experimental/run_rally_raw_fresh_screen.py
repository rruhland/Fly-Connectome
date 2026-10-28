"""Fixed paired raw-readout fresh full-game screen, no fitting or retry."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from run_rally_raw_optimizer import native_task
from run_rally_fresh_screen import metrics
import run_practical_pong_v2 as base


def run(directory,baseline,candidate,absolute_deadline=None):
    directory=Path(directory);baseline=Path(baseline);candidate=Path(candidate);schedule=base.read(directory/'schedules.json');seeds=schedule['probe'];assert len(seeds)==32 and not (directory/'work-clock.json').exists()
    assert len({s%base.MODULUS for s in seeds+schedule.get('unused_reserved',[])})==len(seeds)+len(schedule.get('unused_reserved',[]))
    for source in (baseline,candidate):
        if (source/'completion-verification.json').exists():assert base.read(source/'completion-verification.json')['passed']
        else:assert base.read(source/'own-completion.json')['status']=='PASS'
    assert base.read(baseline/'radial-cap.json')==base.read(candidate/'radial-cap.json')
    for head,source in [('carry',baseline),('candidate',candidate)]:
        folder=directory/head;folder.mkdir();(folder/'native_candidate/episodes').mkdir(parents=True);(folder/'native_candidate/features').mkdir()
        for name in ('proposal.npy','radial-cap.json'):(folder/name).write_bytes((source/name).read_bytes())
        if (source/'rbf-model.npz').exists():(folder/'rbf-model.npz').write_bytes((source/'rbf-model.npz').read_bytes())
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_raw_fresh_screen.py','run_rally_raw_optimizer.py','run_rally_fresh_screen.py')}
    model_hash=lambda head:base.sha(directory/head/('rbf-model.npz' if (directory/head/'rbf-model.npz').exists() else 'proposal.npy'))
    if (directory/'candidate/rbf-model.npz').exists():
        for n in ('pong_fixed_rbf.py','pong_fixed_pairwise.py'):files[n]=base.sha(Path(__file__).with_name(n))
    manifest=dict(source=base.source_signature(),files=files,baseline_root=str(baseline),candidate_root=str(candidate),heads={head:model_hash(head) for head in ('carry','candidate')},settings='native20s/.65opponent/K1/fullgame/unchangedlegalcamera',seeds=seeds,work_seconds=300,no_fitting=True,no_retries=True)
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();deadline=min(began+300,absolute_deadline) if absolute_deadline is not None else began+300;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline));results={head:{} for head in ('carry','candidate')}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter((head,i,seed) for i,seed in enumerate(seeds) for head in ('carry','candidate'));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:head,i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory/head),seed,i,deadline-1)]=(head,i)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:
                head,i=active.pop(f);results[head][i]=f.result()
            base.atomic_json(directory/'progress.json',dict(games=sum(map(len,results.values())),policy_cameras=sum(r['transitions'] for v in results.values() for r in v.values())));fill()
    complete=all(len(v)==32 and all(r['complete'] for r in v.values()) for v in results.values());groups={head:metrics([v[i] for i in sorted(v)]) for head,v in results.items()};paired={}
    if complete:
        carry=[results['carry'][i] for i in range(32)];learned=[results['candidate'][i] for i in range(32)]
        for a,b in zip(carry,learned):
            fa=a['annotation']['first_incoming'];fb=b['annotation']['first_incoming'];assert {k:v for k,v in fa.items() if k!='success'}=={k:v for k,v in fb.items() if k!='success'}
        ha=np.array([r['report']['hits'] for r in carry]);hb=np.array([r['report']['hits'] for r in learned]);index=np.random.default_rng(1834076291).integers(0,32,(10000,32))
        for name,a,b in [('first_return',ha>0,hb>0),('at_least2',ha>=2,hb>=2),('at_least3',ha>=3,hb>=3),('mean_hits',ha,hb)]:
            delta=b.astype(float)-a.astype(float);paired[name]=dict(difference=float(delta.mean()),paired_bootstrap95=np.quantile(delta[index].mean(1),[.025,.975]).tolist(),gained_games=int(np.sum(delta>0)),lost_games=int(np.sum(delta<0)))
        oa=np.array([r['annotation']['own_opportunities']-int(r['annotation']['first_incoming'] is not None) for r in carry]);ob=np.array([r['annotation']['own_opportunities']-int(r['annotation']['first_incoming'] is not None) for r in learned]);la=np.maximum(ha-1,0);lb=np.maximum(hb-1,0)
        with np.errstate(divide='ignore',invalid='ignore'):delta=lb[index].sum(1)/ob[index].sum(1)-la[index].sum(1)/oa[index].sum(1)
        valid=np.isfinite(delta);paired['later_return_fraction']=dict(difference=groups['candidate']['later_return_fraction']-groups['carry']['later_return_fraction'],paired_bootstrap95=np.quantile(delta[valid],[.025,.975]).tolist(),invalid_draws=int((~valid).sum()),conditional_exposure_warning=True)
    summary=dict(status='complete' if complete else 'bounded_partial',groups=groups,paired=paired,baseline_root=str(baseline),candidate_root=str(candidate),fresh_transition_cameras=sum(r['transitions'] for v in results.values() for r in v.values()),initial_cameras=sum(r['initial_cameras'] for v in results.values() for r in v.values()),wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},weights_frozen=manifest['heads']=={head:model_hash(head) for head in ('carry','candidate')},no_fitting=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--baseline-root',required=True);p.add_argument('--candidate-root',required=True);a=p.parse_args();run(a.directory,a.baseline_root,a.candidate_root)
