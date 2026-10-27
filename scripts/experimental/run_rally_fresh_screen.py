"""One fixed32-start paired ordinary-game screen; no learning or retries."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from run_rally_baseline_pong import collect_one
from return_metrics import aggregate
import run_practical_pong_v2 as base


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_fresh_screen.py','run_rally_baseline_pong.py','pong_action_credit.py','return_metrics.py')}


def metrics(rows):
    a=aggregate([r['annotation'] for r in rows]);first=a['first_incoming_returns'];later=a['own_returns']-first;opps=a['own_opportunities']-a['first_incoming_opportunities']
    depth={}
    for e in [e for r in rows for e in r['opportunity_events'] if e['kind']=='own']:
        d=depth.setdefault(str(e['ordinal']),dict(opportunities=0,returns=0,misses=0));d['opportunities']+=1;d['returns']+=e['success'];d['misses']+=not e['success']
    return dict(aggregate=a,later_returns=later,later_opportunities=opps,later_return_fraction=later/opps if opps else None,contact_depth=depth,
        tail={str(k):sum(r['report']['hits']>=k for r in rows) for k in range(1,7)},own_misses=sum(r['report']['losses'] for r in rows),
        opponent_ended=sum(r['report']['wins'] for r in rows),timeouts=sum(r['report']['truncations'] for r in rows))


def run(directory,candidate):
    directory=Path(directory);candidate=Path(candidate);schedule=base.read(directory/'schedules.json');seeds=schedule['probe'];assert len(seeds)==32
    assert base.read(candidate/'completion-verification.json')['passed'] and not (directory/'work-clock.json').exists()
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+300
    for head,weights,theta in [('carry','carry.npy','carry_theta.npy'),('candidate','proposal.npy','proposal_theta.npy')]:
        folder=directory/head;folder.mkdir();(folder/'episodes').mkdir();(folder/'features').mkdir()
        for source,name in [(weights,'carry.npy'),(theta,'carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json')]:(folder/name).write_bytes((candidate/source).read_bytes())
    manifest=dict(files=pins(),source=base.source_signature(),candidate_root=str(candidate),heads={h:base.sha(directory/h/'carry.npy') for h in ('carry','candidate')},
        work_seconds=300,no_learning=True,no_retries=True,seeds=32,max_cameras=64000,settings='native20s,.65opponent,K1,fullgame;carry/candidate aliases inside each rollout folder')
    assert manifest['heads']['carry']=='184dd67a67999224dbe4eb5ff3ba3b86ebdf821d15f1486eef2432a56e99fb15';base.atomic_json(directory/'manifest.json',manifest)
    results={h:{} for h in ('carry','candidate')};jobs=[(h,i,s) for i,s in enumerate(seeds) for h in ('carry','candidate')]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(jobs);active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:h,i,s=next(pending)
                except StopIteration:break
                active[pool.submit(collect_one,str(directory/h),s,i,deadline-1,False)]=(h,i,s)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                h,i,s=active.pop(future);results[h][i]=future.result()
            base.atomic_json(directory/'progress.json',dict(games=sum(map(len,results.values())),policy_cameras=sum(r['transitions'] for v in results.values() for r in v.values())));fill()
    complete=all(len(v)==32 and all(r['complete'] for r in v.values()) for v in results.values());groups={h:metrics([v[i] for i in sorted(v)]) for h,v in results.items()};paired={}
    if complete:
        original=[results['carry'][i] for i in range(32)];learned=[results['candidate'][i] for i in range(32)]
        for a,b in zip(original,learned):
            fa=a['annotation']['first_incoming'];fb=b['annotation']['first_incoming'];assert {k:v for k,v in fa.items() if k!='success'}=={k:v for k,v in fb.items() if k!='success'}
        index=np.random.default_rng(1834076291).integers(0,32,(10000,32))
        ha=np.array([r['report']['hits'] for r in original]);hb=np.array([r['report']['hits'] for r in learned])
        oa=np.array([r['annotation']['own_opportunities']-int(r['annotation']['first_incoming'] is not None) for r in original]);ob=np.array([r['annotation']['own_opportunities']-int(r['annotation']['first_incoming'] is not None) for r in learned])
        la=np.maximum(ha-1,0);lb=np.maximum(hb-1,0)
        for name,da,db in [('first_return',ha>0,hb>0),('at_least2',ha>=2,hb>=2),('at_least3',ha>=3,hb>=3),('mean_hits',ha,hb)]:
            delta=db.astype(float)-da.astype(float);paired[name]=dict(difference=float(delta.mean()),paired_bootstrap95=np.quantile(delta[index].mean(1),[.025,.975]).tolist())
        with np.errstate(divide='ignore',invalid='ignore'):delta=lb[index].sum(1)/ob[index].sum(1)-la[index].sum(1)/oa[index].sum(1)
        valid=np.isfinite(delta);paired['later_return_fraction']=dict(difference=groups['candidate']['later_return_fraction']-groups['carry']['later_return_fraction'],paired_bootstrap95=np.quantile(delta[valid],[.025,.975]).tolist(),invalid_draws=int((~valid).sum()),conditional_exposure_warning=True)
    summary=dict(status='complete' if complete else 'bounded_partial',groups=groups,paired=paired,
        fresh_transition_cameras=sum(r['transitions'] for v in results.values() for r in v.values()),initial_cameras=sum(r['initial_cameras'] for v in results.values() for r in v.values()),
        wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and manifest['files']==pins(),no_fitting=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--candidate',default='/workspace/m1b-experiments/pong-rally-imitation-20261027-r1');a=p.parse_args();run(a.directory,a.candidate)
