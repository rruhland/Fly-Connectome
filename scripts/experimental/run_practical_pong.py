"""Repeated ordinary-serve practice with a small persistent phase-balanced bank."""
import argparse,copy,hashlib,json,multiprocessing as mp,os,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from fast_pong_rollout import run_episode,EXECUTION_KIND
from policy_search_math import sample_directions,mirrored_update,function_rms
import run_pong_policy_search_continuation as old_engine
from fly_connectome.control.session import source_signature,MODEL_HASH

EXPERIMENTS=Path('/workspace/m1b-experiments')
PARENT=EXPERIMENTS/'pong-rightward-interception-20261022-r1'
ROOTS=dict(bank_left=2864193751,bank_right=1250379683,probe_left=3741058267,probe_right=1985627049,full=3167840521,optimizer=817493625)
TAG=41027;MODULUS=2147483647
CONFIG=dict(generations=6,wall_seconds=1800,workers=4,initial_bank_per_phase=16,maximum_bank_total=64,
    probe_every=3,probe_per_phase=16,full_probe_games=32,noise_scale=.25,step_cap=.10,directions=8,top=4,
    train_expansion_threshold=.85,fresh_expansion_threshold=.80,first_hit_ticks=1200,full_ticks=2400,opponent_speed=.65)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
def atomic_json(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def save_array(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,np.asarray(x,dtype=np.float64),allow_pickle=False)
    os.replace(tmp,p)
def leftward(seed):return ((seed%MODULUS*48271+1)%MODULUS)%2==1


def phases(bank,hits):
    rates={p:float(np.mean([hits[str(s)] for s in bank[p]])) for p in ('left','right')}
    return dict(rates=rates,key=[min(rates.values()),.5*sum(rates.values())],counts={p:dict(hits=sum(hits[str(s)] for s in bank[p]),episodes=len(bank[p])) for p in bank})


def choose(scored):
    # Input priority is incumbent, combined, then flattened children. Strict >
    # keeps the incumbent on ties and the combined proposal before tied children.
    best=0
    for i in range(1,len(scored)):
        if tuple(scored[i]['score']['key'])>tuple(scored[best]['score']['key']):best=i
    return scored[best]


def expansion(bank,train,probe):
    if not all(train['rates'][p]>=CONFIG['train_expansion_threshold'] for p in bank):return []
    if not any(probe[p]['current_rate']<CONFIG['fresh_expansion_threshold'] for p in bank):return []
    # Interleave failed phase contexts deterministically, avoiding capacity bias.
    failed={p:[s for s,h in zip(probe[p]['seeds'],probe[p]['current_hits']) if not h and s not in bank[p]] for p in bank}
    additions=[]
    for i in range(max(map(len,failed.values()),default=0)):
        for p in ('left','right'):
            if i<len(failed[p]) and sum(map(len,bank.values()))+len(additions)<CONFIG['maximum_bank_total']:
                additions.append(dict(phase=p,seed=failed[p][i],hit=0))
    return additions


def schedule():
    used=set(old_engine.FINAL);sources={}
    def collect(v):
        if type(v) is int and 0<v<2**32:used.add(v)
        elif isinstance(v,list):
            for x in v:collect(x)
        elif isinstance(v,dict):
            for x in v.values():collect(x)
    for p in sorted(EXPERIMENTS.rglob('schedules.json')):
        collect(read(p));sources[str(p)]=sha(p)
    # Previous seed exclusions also reserve historical roots and non-schedule IDs.
    for p in sorted(EXPERIMENTS.glob('*/seed-exclusions.json')):collect(read(p).get('ids',[]))
    used.update(ROOTS.values());canonical={v%MODULUS for v in used};rejected=[]
    def draw(domain,n,phase=None):
        rng=np.random.default_rng(np.random.SeedSequence([ROOTS[domain],TAG]));out=[]
        while len(out)<n:
            s=int(rng.integers(1,2**32,dtype=np.uint64));reason=None
            if s in used or s%MODULUS in canonical:reason='reserved identity'
            used.add(s);canonical.add(s%MODULUS)
            if reason is None and phase is not None and leftward(s)!=(phase=='left'):reason='ordinary direction rejection'
            if reason is None:out.append(s)
            else:rejected.append(dict(seed=s,domain=domain,reason=reason))
        return out
    return dict(bank={p:draw('bank_'+p,16,p) for p in ('left','right')},
        probe={p:draw('probe_'+p,32,p) for p in ('left','right')},full=draw('full',32),rejected=rejected,excluded_metadata=sources)


def job(directory,jid,seed,weights,phase,purpose,first_hit=True):
    return dict(id=jid,seed=seed,weights=weights,phase=phase,purpose=purpose,first_hit=first_hit,
        config=dict(seed=seed,episodes=1,max_episode_ticks=1200 if first_hit else 2400,policy_kind='trained',action_hold_frames=1,opponent_speed=.65,
                    policy_scaling=read(directory/'radial-cap.json')),
        camera_grant=500 if first_hit else 1000)


def task(directory,j,deadline):
    directory=Path(directory);weights=np.load(directory/j['weights'],allow_pickle=False)
    result=run_episode(weights,j['config'],first_hit_cutoff=j['first_hit'],max_transitions=j['camera_grant'],deadline_monotonic=deadline)
    result.update(id=j['id'],seed=j['seed'],weights_file=j['weights'],weights_sha256=sha(directory/j['weights']),phase=j['phase'],purpose=j['purpose'])
    atomic_json(directory/'episodes'/(j['id']+'.json'),result)
    return result


def initialize():torch.set_num_threads(1)


def dispatch(directory,jobs,pool,deadline):
    pending=list(jobs);active={};results={};log=directory/'dispatch.jsonl'
    while pending or active:
        while pending and len(active)<CONFIG['workers'] and time.monotonic()<deadline-2:
            j=pending.pop(0);path=directory/'episodes'/(j['id']+'.json')
            if path.exists():results[j['id']]=read(path);continue
            with log.open('a') as f:
                f.write(json.dumps(dict(event='dispatch',job=j,wall_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
            active[pool.submit(task,str(directory),j,deadline-5.)]=j
        if not active:break
        done,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
        for future in done:
            j=active.pop(future)
            try:results[j['id']]=future.result()
            except Exception as e:
                # The reserved grant stays charged; no invented exact observation count.
                atomic_json(directory/'episodes'/(j['id']+'-lost.json'),dict(id=j['id'],error=repr(e),camera_grant=j['camera_grant'],initial_grant=1,observations_exact=False))
                raise
    complete=len(results)==len(jobs) and all(r['complete'] and not r['failed'] for r in results.values())
    return results,complete


def bank_jobs(directory,label,bank,weights,purpose='train'):
    return [job(directory,f'{label}_{p}_{i:02d}',seed,weights,p,purpose) for p in ('left','right') for i,seed in enumerate(bank[p])]


def hits_for(jobs,results):
    return {str(j['seed']):int(results[j['id']]['report']['hits']>0) for j in jobs}


def accounting(directory):
    actual=dict(train=0,phase_probe=0,full_probe=0,initials=0);complete=0;lost_upper=0;unknown=False
    received=set()
    for p in (directory/'episodes').glob('*.json'):
        r=read(p)
        if p.stem.endswith('-lost'):continue
        received.add(r['id']);actual[r['purpose']]+=r['transitions'];actual['initials']+=r['initial_cameras'];complete+=int(r['complete'])
        if not r['observations_exact']:
            unknown=True;lost_upper+=max(0,r['transition_camera_upper_bound']-r['transitions'])+max(0,r['initial_camera_upper_bound']-r['initial_cameras'])
    entries=[json.loads(v) for v in (directory/'dispatch.jsonl').read_text().splitlines()] if (directory/'dispatch.jsonl').exists() else []
    for e in entries:
        if e['job']['id'] not in received:unknown=True;lost_upper+=e['job']['camera_grant']+1
    return dict(actual_observed=actual,observations_exact=not unknown,unacknowledged_or_ambiguous_observation_upper_bound=lost_upper,completed_episodes=complete)


def stage(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    (directory/'episodes').mkdir();torch.set_num_threads(1)
    theta=np.load(PARENT/'theta_at6.npy',allow_pickle=False);matrix=np.load(PARENT/'search-transform.npy',allow_pickle=False)
    assert np.array_equal(theta@matrix,np.load(PARENT/'endpoint.npy',allow_pickle=False))
    save_array(directory/'theta_start.npy',theta);save_array(directory/'start.npy',theta@matrix);save_array(directory/'search-transform.npy',matrix)
    atomic_json(directory/'radial-cap.json',read(PARENT/'radial-cap.json'));seeds=schedule();atomic_json(directory/'schedules.json',seeds)
    m=dict(config=CONFIG,roots=ROOTS,tag=TAG,source=source_signature(),model_hash=MODEL_HASH,harness_sha256=sha(Path(__file__).with_name('fast_pong_rollout.py')),trainer_sha256=sha(__file__),
        start_theta=str(PARENT/'theta_at6.npy'),start_hash=ah(theta),matrix_hash=ah(matrix),execution_kind=EXECUTION_KIND,
        inherited_training=558832,inherited_used=532296,inherited_unused=26536,separate_interrupted_paid_unused=14879,
        assistance='persistent ordinary-seed TRAIN practice, failed-probe curriculum, lexicographic two-phase selection; no privileged policy inputs/backprop/new architecture',
        objective='maximize (minimum phase return rate,equal-weight phase mean); stable incumbent/combined/flattened-child ties',
        original_direction_scale=.25,combined_step_cap=.10,probe_failures_may_become_training=True,
        crash_semantics='completed-generation checkpoint; completed episode receipts survive; unacknowledged grants charged conservatively; no arbitrary-camera continuation')
    atomic_json(directory/'manifest.json',m)
    state=dict(generation=0,theta='theta_start.npy',weights='start.npy',theta_hash=ah(theta),bank=seeds['bank'],cached_hits=None,rng=np.random.default_rng(ROOTS['optimizer']).bit_generator.state)
    atomic_json(directory/'state.json',state)
    print(json.dumps(dict(staged=True,new_observations=0,root=str(directory),config=CONFIG,harness_sha256=m['harness_sha256'],trainer_sha256=m['trainer_sha256'])),flush=True)


def run(directory):
    directory=Path(directory);m=read(directory/'manifest.json');assert m['config']==CONFIG and m['trainer_sha256']==sha(__file__) and m['harness_sha256']==sha(Path(__file__).with_name('fast_pong_rollout.py')) and m['source']==source_signature()
    assert not (directory/'block-summary.json').exists(),'new block needs explicit new prospective budget'
    assert not (directory/'launch.json').exists(),'interrupted blocks require explicit closure/new budget; never reset/restart this clock'
    atomic_json(directory/'launch.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()))
    launch=read(directory/'launch.json');deadline=launch['started_monotonic']+CONFIG['wall_seconds'];state=read(directory/'state.json');matrix=np.load(directory/'search-transform.npy');theta=np.load(directory/state['theta'])
    z=np.asarray(old_engine.read_calibration(old_engine.CACHE)@matrix.T);seeds=read(directory/'schedules.json');reason='six completed generations';history=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers'],mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        if state['cached_hits'] is None:
            jobs=bank_jobs(directory,'initial',state['bank'],state['weights']);reports,complete=dispatch(directory,jobs,pool,deadline)
            if not complete:reason='initial bank incomplete'
            else:
                state['cached_hits']=hits_for(jobs,reports);atomic_json(directory/'state.json',state)
                print(json.dumps(dict(stage='initial',training=phases(state['bank'],state['cached_hits']),accounting=accounting(directory))),flush=True)
        while state['cached_hits'] is not None and state['generation']<CONFIG['generations'] and time.monotonic()<deadline-2:
            n=state['generation']+1;bank=state['bank'];rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);directions=sample_directions(rng,z,8)
            save_array(directory/f'g{n:02d}-directions.npy',directions);jobs=[]
            for d in range(8):
                for sign in range(2):
                    candidate=theta+.25*directions[d]*(1 if sign==0 else -1);name=f'g{n:02d}_d{d}_{sign}.npy';save_array(directory/name,candidate@matrix)
                    jobs+=bank_jobs(directory,f'g{n:02d}_d{d}_{sign}',bank,name)
            reports,complete=dispatch(directory,jobs,pool,deadline)
            if not complete:reason='candidate population incomplete';break
            fitness=np.zeros((8,2));children=[]
            for d in range(8):
                for sign in range(2):
                    js=[j for j in jobs if j['weights']==f'g{n:02d}_d{d}_{sign}.npy'];h=hits_for(js,reports);score=phases(bank,h);fitness[d,sign]=score['key'][1]
                    children.append(dict(label='child',direction=d,sign=sign,hits=h,score=score))
            proposed,diagnostics=mirrored_update(theta,directions,fitness,z);save_array(directory/f'g{n:02d}_combined.npy',proposed@matrix)
            js=bank_jobs(directory,f'g{n:02d}_combined',bank,f'g{n:02d}_combined.npy');rr,complete=dispatch(directory,js,pool,deadline)
            if not complete:reason='combined proposal incomplete';break
            h=hits_for(js,rr);incumbent=dict(label='incumbent',hits=state['cached_hits'],score=phases(bank,state['cached_hits']))
            combined=dict(label='combined',hits=h,score=phases(bank,h));winner=choose([incumbent,combined]+children)
            before=theta.copy()
            if winner['label']=='combined':theta=proposed
            elif winner['label']=='child':theta=theta+.25*directions[winner['direction']]*(1 if winner['sign']==0 else -1)
            save_array(directory/f'theta_g{n:02d}.npy',theta);save_array(directory/f'policy_g{n:02d}.npy',theta@matrix)
            state=dict(generation=n,theta=f'theta_g{n:02d}.npy',weights=f'policy_g{n:02d}.npy',theta_hash=ah(theta),bank=bank,cached_hits=winner['hits'],rng=copy.deepcopy(rng.bit_generator.state))
            result=dict(generation=n,before=incumbent['score'],after=winner['score'],selected={k:v for k,v in winner.items() if k not in ('hits',)},fitness=fitness.tolist(),combined_diagnostics=diagnostics,
                applied_function_rms=function_rms(theta-before,z),theta_hash=ah(theta),state_after=state,accounting=accounting(directory))
            atomic_json(directory/f'g{n:02d}-result.json',result);atomic_json(directory/'state.json',state);history.append(result)
            print(json.dumps(dict(stage='generation',generation=n,selected=winner['label'],training=winner['score'],applied_function_rms=result['applied_function_rms'],accounting=result['accounting'])),flush=True)
            if n%3==0:
                probe={};probe_complete=True
                for p in ('left','right'):
                    ss=seeds['probe'][p][(n//3-1)*16:n//3*16];pj=[]
                    for i,s in enumerate(ss):
                        pj.append(job(directory,f'probe_g{n:02d}_{p}_start_{i:02d}',s,'start.npy',p,'phase_probe'))
                        if state['theta_hash']!=m['start_hash']:pj.append(job(directory,f'probe_g{n:02d}_{p}_current_{i:02d}',s,state['weights'],p,'phase_probe'))
                    pr,ok=dispatch(directory,pj,pool,deadline)
                    if not ok:probe_complete=False;break
                    start_hits=[int(pr[f'probe_g{n:02d}_{p}_start_{i:02d}']['report']['hits']>0) for i in range(16)]
                    current_hits=start_hits if state['theta_hash']==m['start_hash'] else [int(pr[f'probe_g{n:02d}_{p}_current_{i:02d}']['report']['hits']>0) for i in range(16)]
                    probe[p]=dict(seeds=ss,start_hits=start_hits,current_hits=current_hits,start_rate=float(np.mean(start_hits)),current_rate=float(np.mean(current_hits)))
                if not probe_complete:reason='phase probe incomplete';break
                additions=expansion(bank,winner['score'],probe)
                for a in additions:state['bank'][a['phase']].append(a['seed']);state['cached_hits'][str(a['seed'])]=a['hit']
                atomic_json(directory/f'g{n:02d}-probe.json',dict(phases=probe,appended_training_failures=additions,selected_train_score=winner['score'],state_after=state))
                atomic_json(directory/'state.json',state)
                print(json.dumps(dict(stage='fresh phase probe',generation=n,phases=probe,appended_failures=len(additions))),flush=True)
            if n==6:
                fj=[]
                for i,s in enumerate(seeds['full']):
                    for label,weights in [('start','start.npy')]+([] if state['theta_hash']==m['start_hash'] else [('current',state['weights'])]):fj.append(job(directory,f'full_{label}_{i:02d}',s,weights,'full','full_probe',False))
                fr,ok=dispatch(directory,fj,pool,deadline)
                full={}
                if ok:
                    for label in ('start','current'):
                        actual='start' if state['theta_hash']==m['start_hash'] else label
                        hits=[fr[f'full_{actual}_{i:02d}']['report']['hits'] for i in range(32)]
                        full[label]=dict(mean_hits=float(np.mean(hits)),first_return_rate=float(np.mean(np.asarray(hits)>0)),at_least_2=int(np.count_nonzero(np.asarray(hits)>=2)))
                atomic_json(directory/'full-probe.json',dict(complete=ok,policies=full));print(json.dumps(dict(stage='ordinary full-game probe',complete=ok,policies=full)),flush=True)
                if not ok:reason='ordinary full-game probe incomplete'
        if state['generation']<6 and reason=='six completed generations':reason='review wall budget'
    all_probes=all((directory/f'g{n:02d}-probe.json').exists() for n in (3,6)) and (directory/'full-probe.json').exists() and read(directory/'full-probe.json')['complete']
    summary=dict(status='review_block_complete' if state['generation']==6 and all_probes else 'partial_review_block',reason=reason,state=state,training=phases(state['bank'],state['cached_hits']) if state['cached_hits'] else None,
        accounting=accounting(directory),wall_seconds=(time.time_ns()-launch['started_wall_ns'])/1e9,source=source_signature(),automatic_next_block=False,
        interpretation='Repeated-practice TRAIN/probe progress only; used probe failures explicitly become TRAIN, no final reliability/rally/continuous-learning claim')
    atomic_json(directory/'block-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run'));p.add_argument('root',type=Path);args=p.parse_args()
    if args.mode=='stage':stage(args.root)
    else:run(args.root)
