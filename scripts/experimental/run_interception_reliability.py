"""Frozen two-cohort first-return reliability check and conditional full games."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import run_practical_pong_v2 as base

ROOTS=dict(c1_left=1658097321,c1_right=2946190753,c2_left=3821094657,c2_right=719403821,full=2479018365)
TAG=54197;BOOTSTRAP=2738411927


def schedule():
    used=set(base.old_engine.FINAL)|set(ROOTS.values())|{BOOTSTRAP}
    def collect(v):
        if type(v) is int and 0<v<2**32:used.add(v)
        elif isinstance(v,list):
            for x in v:collect(x)
        elif isinstance(v,dict):
            for x in v.values():collect(x)
    for p in base.EXPERIMENTS.rglob('schedules.json'):collect(base.read(p))
    for p in base.EXPERIMENTS.glob('*/seed-exclusions.json'):collect(base.read(p).get('ids',[]))
    canonical={v%base.MODULUS for v in used};rejected=[]
    def draw(domain,n,phase=None):
        rng=np.random.default_rng(np.random.SeedSequence([ROOTS[domain],TAG]));out=[]
        while len(out)<n:
            v=int(rng.integers(1,2**32,dtype=np.uint64));reserved=v in used or v%base.MODULUS in canonical
            used.add(v);canonical.add(v%base.MODULUS)
            if reserved or phase is not None and base.leftward(v)!=(phase=='left'):rejected.append(dict(seed=v,domain=domain))
            else:out.append(v)
        return out
    return dict(cohorts={c:{p:draw(c+'_'+p,64,p) for p in ('left','right')} for c in ('c1','c2')},full=draw('full',32),roots=ROOTS,tag=TAG,rejected=rejected)


def interval(hits,n):
    z=1.959963984540054;p=hits/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def paired(original,current,key,baseline_label='original_RIGHTtheta6'):
    original=np.asarray(original);current=np.asarray(current);d=current-original;n=len(d);rng=np.random.default_rng(np.random.SeedSequence([BOOTSTRAP,key]))
    ci=np.quantile(d[rng.integers(0,n,size=(10000,n))].mean(1),[.025,.975])
    return dict(baseline_label=baseline_label,legacy_original_fields_are_baseline_aliases=True,episodes=n,original_hits=int(original.sum()),current_hits=int(current.sum()),original_rate=float(original.mean()),current_rate=float(current.mean()),
        current_wilson95=interval(int(current.sum()),n),gains=int(np.sum((1-original)*current)),losses=int(np.sum(original*(1-current))),
        paired_difference=float(d.mean()),paired_bootstrap95=ci.tolist())


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_interception_reliability.py','run_practical_pong_v2.py','fast_pong_feature_rollout.py')}


def stage(directory,source,baseline_source=None):
    directory=Path(directory);source=Path(source);directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists();(directory/'episodes').mkdir()
    sr=base.read(source/'summary.json');assert (sr.get('selected',{}).get('label')=='proposal' or sr.get('selected_proposal') is True) and base.read(source/'completion-verification.json')['passed']
    (directory/'current.npy').write_bytes((source/'selected.npy').read_bytes())
    if baseline_source is None:
        original=base.PARENT;theta=np.load(original/'theta_at6.npy');matrix=np.load(original/'search-transform.npy')
        assert np.array_equal(theta@matrix,np.load(original/'endpoint.npy'));baseline_path=original/'endpoint.npy';baseline_label='original_RIGHTtheta6'
    else:
        original=Path(baseline_source);assert base.read(original/'completion-verification.json')['passed']
        theta=np.load(original/'selected_theta.npy');old=np.load(original/'old-transform.npy');extra=np.load(original/'extra-transform.npy')
        assert np.array_equal(theta[:,:214]@old+theta[:,214:]@extra,np.load(original/'selected.npy'))
        assert base.read(source/'radial-cap.json')==base.read(original/'radial-cap.json')
        baseline_path=original/'selected.npy';baseline_label='r4'
    (directory/'original.npy').write_bytes(baseline_path.read_bytes())
    (directory/'radial-cap.json').write_bytes((source/'radial-cap.json').read_bytes());base.atomic_json(directory/'schedules.json',schedule())
    m=dict(files=pins(),source=base.source_signature(),current_source=str(source/'selected.npy'),original_source=str(baseline_path),baseline_source=str(baseline_path),baseline_label=baseline_label,legacy_original_fields_are_baseline_aliases=True,
        current_hash=base.sha(directory/'current.npy'),original_hash=base.sha(directory/'original.npy'),work_seconds=600,
        first_hit_episodes=512,conditional_full_episodes=64,first_hit_success='native own hits>0 per ordinary episode; missing first incoming remains failure',
        full_gate='both current aggregate phase rates>=.90 over128ordinary starts each; point-estimate screen, report uncertainty separately',
        bootstrap=dict(seed=BOOTSTRAP,replicates=10000,paired_episode_resampling=True),no_learning=True,no_cohort_retry=True)
    base.atomic_json(directory/'manifest.json',m);print(json.dumps(dict(staged=True,new_observations=0,first_hit_episodes=512,conditional_full_episodes=64,current_hash=m['current_hash'],original_hash=m['original_hash'],baseline_label=m['baseline_label'])),flush=True)


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json');assert m['files']==pins() and m['source']==base.source_signature()
    assert m['current_hash']==base.sha(directory/'current.npy') and m['original_hash']==base.sha(directory/'original.npy');assert not (directory/'work-clock.json').exists()
    start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start));deadline=start+600;jobs=[]
    for c,phases in s['cohorts'].items():
        for p,seeds in phases.items():
            for i,seed in enumerate(seeds):
                for head in ('original','current'):jobs.append(base.job(directory,f'{c}_{p}_{head}_{i:02d}',seed,head+'.npy',p,'phase_probe'))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        reports,complete=base.dispatch(directory,jobs,pool,deadline);cohorts={};aggregate={};full={};eligible=False;full_complete=None
        if complete:
            arrays={};key=0
            for c,phases in s['cohorts'].items():
                cohorts[c]={}
                for p,seeds in phases.items():
                    arms={head:[int(reports[f'{c}_{p}_{head}_{i:02d}']['report']['hits']>0) for i in range(64)] for head in ('original','current')}
                    arrays[(c,p)]=arms;key+=1;cohorts[c][p]=paired(arms['original'],arms['current'],key,m['baseline_label'])
            for p in ('left','right'):
                arms={head:sum([arrays[(c,p)][head] for c in ('c1','c2')],[]) for head in ('original','current')};key+=1;aggregate[p]=paired(arms['original'],arms['current'],key,m['baseline_label'])
            eligible=all(a['current_rate']>=.90 for a in aggregate.values());base.atomic_json(directory/'first-hit-results.json',dict(cohorts=cohorts,aggregate=aggregate,conditional_full_eligible=eligible))
            if eligible:
                js=[base.job(directory,f'full_{head}_{i:02d}',seed,head+'.npy','unfiltered','full_probe',first_hit=False) for i,seed in enumerate(s['full']) for head in ('original','current')]
                rr,full_complete=base.dispatch(directory,js,pool,deadline)
                if full_complete:
                    for head in ('original','current'):
                        hits=np.array([rr[f'full_{head}_{i:02d}']['report']['hits'] for i in range(32)])
                        full[head]=dict(episodes=32,first_returns=int(np.sum(hits>0)),own_hits=int(hits.sum()),mean_own_hits=float(hits.mean()),at_least2=int(np.sum(hits>=2)),at_least3=int(np.sum(hits>=3)),at_least4=int(np.sum(hits>=4)),per_game_hits=hits.tolist())
        result=dict(baseline_label=m['baseline_label'],baseline_source=m['baseline_source'],legacy_original_fields_are_baseline_aliases=True,status='complete' if complete and full_complete is not False else 'partial',first_hit_complete=complete,cohorts=cohorts,aggregate=aggregate,
            conditional_full_eligible=eligible,full_complete=full_complete,full=full,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,
            interpretation='fixed heldout check of one frozen learned head; no retry or checkpoint selection; phase reliability is not a rally/continuous-learning claim')
        base.atomic_json(directory/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--baseline-source',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source,a.baseline_source)
    else:run(a.root)
