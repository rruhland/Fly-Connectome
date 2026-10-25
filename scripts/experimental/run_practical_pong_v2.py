"""Repeated ordinary-serve practice with a small persistent phase-balanced bank."""
import argparse,copy,hashlib,json,multiprocessing as mp,os,sys,time
from dataclasses import asdict
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from fast_pong_feature_rollout import run_episode,EXECUTION_KIND
from pong_safe_mutations import harmonic,mutation_scale,balanced_buffer
from policy_search_math import sample_directions,mirrored_update,function_rms
import run_pong_policy_search_continuation as old_engine
from fly_connectome.control.session import source_signature,MODEL_HASH

EXPERIMENTS=Path('/workspace/m1b-experiments')
PARENT=EXPERIMENTS/'pong-rightward-interception-20261022-r1'
PREVIOUS=EXPERIMENTS/'pong-practical-interception-20261025-r1'
ROOTS=dict(probe_left=3472609151,probe_right=1098437627,optimizer=2791654083)
TAG=42131;MODULUS=2147483647
CONFIG=dict(generations=3,wall_seconds=1800,workers=4,representative_bank_per_phase=32,bank_total=78,
    probe_every=3,probe_per_phase=32,noise_cap=.25,step_cap=.10,directions=8,top=4,
    disagreement_targets=[.10]*4+[.25]*4,feature_episodes_per_phase=16,feature_rows_per_episode=16,
    utility='harmonic phase return rate, balancedmean tie',first_hit_ticks=1200,opponent_speed=.65)


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
    return dict(rates=rates,key=[harmonic(rates['left'],rates['right']),.5*sum(rates.values())],counts={p:dict(hits=sum(hits[str(s)] for s in bank[p]),episodes=len(bank[p])) for p in bank})


def choose(scored):
    # Input priority is incumbent, retained anchors, combined, then children. Strict >
    # keeps the incumbent on ties and the combined proposal before tied children.
    best=0
    for i in range(1,len(scored)):
        if tuple(scored[i]['score']['key'])>tuple(scored[best]['score']['key']):best=i
    return scored[best]


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
    old=read(PREVIOUS/'schedules.json');state=read(PREVIOUS/'state.json')
    bank={p:list(old['bank'][p])+list(old['probe'][p][16:32])+[v for v in state['bank'][p] if v not in old['bank'][p]] for p in ('left','right')}
    assert sum(map(len,bank.values()))==78 and len(set(sum(bank.values(),[])))==78
    return dict(bank=bank,probe={p:draw('probe_'+p,32,p) for p in ('left','right')},full=[],rejected=rejected,excluded_metadata=sources,
        reused_original_bank=old['bank'],reclassified_gen6_monitor={p:old['probe'][p][16:32] for p in bank},roots=ROOTS,tag=TAG)


def job(directory,jid,seed,weights,phase,purpose,first_hit=True):
    return dict(id=jid,seed=seed,weights=weights,phase=phase,purpose=purpose,first_hit=first_hit,
        config=dict(seed=seed,episodes=1,max_episode_ticks=1200 if first_hit else 2400,policy_kind='trained',action_hold_frames=1,opponent_speed=.65,
                    policy_scaling=read(directory/'radial-cap.json')),
        camera_grant=500 if first_hit else 1000)


def task(directory,j,deadline):
    directory=Path(directory);weights=np.load(directory/j['weights'],allow_pickle=False)
    extras={}
    if j.get('capture_features'):extras=dict(feature_matrix=np.load(directory/'search-transform.npy'),feature_rows_per_episode=16)
    result=run_episode(weights,j['config'],first_hit_cutoff=j['first_hit'],max_transitions=j['camera_grant'],deadline_monotonic=deadline,**extras)
    captured=result.pop('feature_capture',None)
    if captured is not None:
        path=directory/'features'/(j['id']+'.npz');np.savez(path,**captured)
        theta=np.load(directory/j['theta']);q=captured['X']@theta.T
        error=float(np.max(np.abs(q-captured['q'])))
        action_match=bool(np.array_equal(np.argmax(q,axis=1),captured['actions']))
        assert error<1e-10 and action_match, 'captured inputs must reproduce actual mapped-policy q/action'
        result.update(feature_file=str(path.relative_to(directory)),feature_q_max_error=error,feature_action_match=action_match)
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
    (directory/'episodes').mkdir();(directory/'features').mkdir();torch.set_num_threads(1)
    theta=np.load(PARENT/'theta_at6.npy',allow_pickle=False);matrix=np.load(PARENT/'search-transform.npy',allow_pickle=False)
    assert np.array_equal(theta@matrix,np.load(PARENT/'endpoint.npy',allow_pickle=False))
    save_array(directory/'theta_start.npy',theta);save_array(directory/'start.npy',theta@matrix);save_array(directory/'search-transform.npy',matrix)
    atomic_json(directory/'radial-cap.json',read(PARENT/'radial-cap.json'));seeds=schedule();atomic_json(directory/'schedules.json',seeds)
    anchors=[dict(label='START',theta='theta_start.npy',weights='start.npy',theta_hash=ah(theta))]
    seen={ah(theta)}
    for n in range(1,7):
        rr=read(PREVIOUS/f'g{n:02d}-result.json')
        if rr['applied_function_rms']==0:continue
        tt=np.load(PREVIOUS/f'theta_g{n:02d}.npy');h=ah(tt)
        if h in seen:continue
        seen.add(h);save_array(directory/f'anchor_g{n}.npy',tt);save_array(directory/f'anchor_policy_g{n}.npy',tt@matrix)
        anchors.append(dict(label=f'g{n}',theta=f'anchor_g{n}.npy',weights=f'anchor_policy_g{n}.npy',theta_hash=h))
    assert len(anchors)==5
    m=dict(anchors=anchors,config=CONFIG,roots=ROOTS,tag=TAG,source=source_signature(),model_hash=MODEL_HASH,harness_sha256=sha(Path(__file__).with_name('fast_pong_feature_rollout.py')),trainer_sha256=sha(__file__),
        start_theta=str(PARENT/'theta_at6.npy'),start_hash=ah(theta),matrix_hash=ah(matrix),execution_kind=EXECUTION_KIND,
        inherited_training=558832,inherited_used=532296,inherited_unused=26536,separate_interrupted_paid_unused=14879,
        assistance='broader representative TRAIN bank plus prior failures; retained anchors, harmonic utility and legal forward-disagreement mutation doses; no privileged policy inputs/backprop/new architecture',
        objective='maximize (harmonic phase return rate,equal-weight phase mean); stable incumbent/anchors/combined/flattened-child ties',
        original_direction_scale=.25,combined_step_cap=.10,probe_failures_may_become_training=False,
        mutation_helper_sha256=sha(Path(__file__).with_name('pong_safe_mutations.py')),previous_processed_observations=read(PREVIOUS/'block-summary.json')['accounting'],
        crash_semantics='completed-generation checkpoint; completed episode receipts survive; unacknowledged grants charged conservatively; no arbitrary-camera continuation')
    atomic_json(directory/'manifest.json',m)
    state=dict(generation=0,theta='theta_start.npy',weights='start.npy',theta_hash=ah(theta),bank=seeds['bank'],cached_hits=None,rng=np.random.default_rng(ROOTS['optimizer']).bit_generator.state)
    atomic_json(directory/'state.json',state)
    print(json.dumps(dict(staged=True,new_observations=0,root=str(directory),config=CONFIG,harness_sha256=m['harness_sha256'],trainer_sha256=m['trainer_sha256'])),flush=True)


def config_key(config):
    return json.dumps(asdict(PongEvaluationConfig(**config)),sort_keys=True,separators=(',',':'))


def previous_cache():
    cache={}
    for path in (PREVIOUS/'episodes').glob('*.json'):
        r=read(path)
        if not r.get('complete') or not r.get('observations_exact') or not r.get('first_hit_cutoff'):continue
        cache[(r['weights_sha256'],config_key(r['config']))]=dict(hit=int(r['report']['hits']>0),receipt=str(path))
    return cache


def prepare(directory):
    directory=Path(directory);m=read(directory/'manifest.json');assert m['config']==CONFIG and m['source']==source_signature() and m['mutation_helper_sha256']==sha(Path(__file__).with_name('pong_safe_mutations.py')) and m['trainer_sha256']==sha(__file__) and m['harness_sha256']==sha(Path(__file__).with_name('fast_pong_feature_rollout.py'))
    assert not (directory/'launch.json').exists(),'no resetting old preparation/block clock'
    atomic_json(directory/'launch.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()))
    deadline=read(directory/'launch.json')['started_monotonic']+CONFIG['wall_seconds'];state=read(directory/'state.json');bank=state['bank'];cache=previous_cache();anchors=[];reused=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for anchor in m['anchors']:
            h={};jobs=bank_jobs(directory,'anchor_'+anchor['label'],bank,anchor['weights'])
            missing=[];weight_hash=sha(directory/anchor['weights'])
            for j in jobs:
                old=cache.get((weight_hash,config_key(j['config'])))
                if old is None:missing.append(j)
                else:h[str(j['seed'])]=old['hit'];reused.append(dict(anchor=anchor['label'],seed=j['seed'],receipt=old['receipt']))
            rr,ok=dispatch(directory,missing,pool,deadline)
            if not ok:raise RuntimeError('anchor preparation partial; preserve counts and close block')
            h.update(hits_for(missing,rr));anchors.append(dict(anchor,label='anchor',anchor=anchor['label'],hits=h,score=phases(bank,h)))
        chosen=choose(anchors);state.update(theta=chosen['theta'],weights=chosen['weights'],theta_hash=chosen['theta_hash'],cached_hits=chosen['hits'])
        atomic_json(directory/'anchors.json',dict(scored=anchors,chosen=chosen['anchor'],reused_outcomes=reused))
        sample={p:[bank[p][i] for i in np.linspace(0,len(bank[p])-1,16,dtype=int)] for p in bank};jobs=bank_jobs(directory,'feature_anchor',sample,state['weights'])
        for j in jobs:j.update(capture_features=True,theta=state['theta'])
        rr,ok=dispatch(directory,jobs,pool,deadline)
        if not ok:raise RuntimeError('feature preparation partial; preserve counts and close block')
        cases=[]
        for j in jobs:
            with np.load(directory/rr[j['id']]['feature_file']) as data:cases.append(dict(phase=j['phase'],X=data['X']))
        X,weights=balanced_buffer(cases);np.savez(directory/'feature-buffer.npz',X=X,weights=weights)
    matrix=np.load(directory/'search-transform.npy');z=np.asarray(old_engine.read_calibration(old_engine.CACHE)@matrix.T);theta=np.load(directory/state['theta']);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);directions=sample_directions(rng,z,8)
    profile=[]
    for i,direction in enumerate(directions):
        scale,info=mutation_scale(theta,direction,X,weights,CONFIG['disagreement_targets'][i]);profile.append(dict(direction=i,scale=scale,**info))
    atomic_json(directory/'state.json',state)
    result=dict(complete=True,chosen_anchor=chosen['anchor'],anchor_scores={v['anchor']:v['score'] for v in anchors},reused_outcomes=len(reused),feature_rows=len(X),feature_episodes=len(jobs),
        q_max_error=max(r['feature_q_max_error'] for r in rr.values()),actions_exact=all(r['feature_action_match'] for r in rr.values()),initial_doses=profile,accounting=accounting(directory),
        elapsed_seconds=time.monotonic()-read(directory/'launch.json')['started_monotonic'])
    atomic_json(directory/'preparation.json',result);print(json.dumps(result),flush=True)


def run(directory):
    directory=Path(directory);m=read(directory/'manifest.json');assert m['mutation_helper_sha256']==sha(Path(__file__).with_name('pong_safe_mutations.py')) and m['config']==CONFIG and m['trainer_sha256']==sha(__file__) and m['harness_sha256']==sha(Path(__file__).with_name('fast_pong_feature_rollout.py')) and m['source']==source_signature()
    assert not (directory/'block-summary.json').exists(),'new block needs explicit new prospective budget'
    assert read(directory/'preparation.json')['complete'],'prepare and review legal buffer before training'
    launch=read(directory/'launch.json');deadline=launch['started_monotonic']+CONFIG['wall_seconds'];state=read(directory/'state.json');matrix=np.load(directory/'search-transform.npy');theta=np.load(directory/state['theta'])
    z=np.asarray(old_engine.read_calibration(old_engine.CACHE)@matrix.T);seeds=read(directory/'schedules.json');reason='three completed generations';anchors=read(directory/'anchors.json')['scored']
    with np.load(directory/'feature-buffer.npz') as b:X=b['X'];buffer_weights=b['weights']
    with ProcessPoolExecutor(max_workers=CONFIG['workers'],mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        while state['cached_hits'] is not None and state['generation']<CONFIG['generations'] and time.monotonic()<deadline-2:
            n=state['generation']+1;bank=state['bank'];rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);directions=sample_directions(rng,z,8)
            save_array(directory/f'g{n:02d}-directions.npy',directions);jobs=[];scales=[];dose=[]
            for i,direction in enumerate(directions):
                scale,info=mutation_scale(theta,direction,X,buffer_weights,CONFIG['disagreement_targets'][i]);scales.append(scale);dose.append(dict(direction=i,scale=scale,**info))
            atomic_json(directory/f'g{n:02d}-dose.json',dose)
            for d in range(8):
                for sign in range(2):
                    candidate=theta+scales[d]*directions[d]*(1 if sign==0 else -1);name=f'g{n:02d}_d{d}_{sign}.npy';save_array(directory/name,candidate@matrix)
                    jobs+=bank_jobs(directory,f'g{n:02d}_d{d}_{sign}',bank,name)
            reports,complete=dispatch(directory,jobs,pool,deadline)
            if not complete:reason='candidate population incomplete';break
            fitness=np.zeros((8,2));children=[]
            for d in range(8):
                for sign in range(2):
                    js=[j for j in jobs if j['weights']==f'g{n:02d}_d{d}_{sign}.npy'];h=hits_for(js,reports);score=phases(bank,h);fitness[d,sign]=score['key'][0]
                    children.append(dict(label='child',direction=d,sign=sign,hits=h,score=score))
            proposed,diagnostics=mirrored_update(theta,directions,fitness,z);save_array(directory/f'g{n:02d}_combined.npy',proposed@matrix)
            js=bank_jobs(directory,f'g{n:02d}_combined',bank,f'g{n:02d}_combined.npy');rr,complete=dispatch(directory,js,pool,deadline)
            if not complete:reason='combined proposal incomplete';break
            h=hits_for(js,rr);incumbent=dict(label='incumbent',hits=state['cached_hits'],score=phases(bank,state['cached_hits']))
            combined=dict(label='combined',hits=h,score=phases(bank,h));winner=choose([incumbent]+anchors+[combined]+children)
            before=theta.copy()
            if winner['label']=='combined':theta=proposed
            elif winner['label']=='child':theta=theta+scales[winner['direction']]*directions[winner['direction']]*(1 if winner['sign']==0 else -1)
            elif winner['label']=='anchor':theta=np.load(directory/winner['theta'])
            save_array(directory/f'theta_g{n:02d}.npy',theta);save_array(directory/f'policy_g{n:02d}.npy',theta@matrix)
            state=dict(generation=n,theta=f'theta_g{n:02d}.npy',weights=f'policy_g{n:02d}.npy',theta_hash=ah(theta),bank=bank,cached_hits=winner['hits'],rng=copy.deepcopy(rng.bit_generator.state))
            result=dict(generation=n,before=incumbent['score'],after=winner['score'],selected={k:v for k,v in winner.items() if k not in ('hits',)},fitness=fitness.tolist(),combined_diagnostics=diagnostics,
                applied_function_rms=function_rms(theta-before,z),theta_hash=ah(theta),state_after=state,accounting=accounting(directory))
            atomic_json(directory/f'g{n:02d}-result.json',result);atomic_json(directory/'state.json',state)
            print(json.dumps(dict(stage='generation',generation=n,selected=winner['label'],training=winner['score'],applied_function_rms=result['applied_function_rms'],accounting=result['accounting'])),flush=True)
            if n%3==0:
                probe={};probe_complete=True
                for p in ('left','right'):
                    ss=seeds['probe'][p];pj=[]
                    for i,s in enumerate(ss):
                        pj.append(job(directory,f'probe_g{n:02d}_{p}_start_{i:02d}',s,'start.npy',p,'phase_probe'))
                        if state['theta_hash']!=m['start_hash']:pj.append(job(directory,f'probe_g{n:02d}_{p}_current_{i:02d}',s,state['weights'],p,'phase_probe'))
                    pr,ok=dispatch(directory,pj,pool,deadline)
                    if not ok:probe_complete=False;break
                    start_hits=[int(pr[f'probe_g{n:02d}_{p}_start_{i:02d}']['report']['hits']>0) for i in range(32)]
                    current_hits=start_hits if state['theta_hash']==m['start_hash'] else [int(pr[f'probe_g{n:02d}_{p}_current_{i:02d}']['report']['hits']>0) for i in range(32)]
                    probe[p]=dict(seeds=ss,start_hits=start_hits,current_hits=current_hits,start_rate=float(np.mean(start_hits)),current_rate=float(np.mean(current_hits)))
                if not probe_complete:reason='phase probe incomplete';break
                atomic_json(directory/f'g{n:02d}-probe.json',dict(phases=probe,no_expansion_this_block=True))
                print(json.dumps(dict(stage='fresh phase probe',generation=n,phases=probe)),flush=True)
        if state['generation']<3 and reason=='three completed generations':reason='review wall budget'
    all_probes=(directory/'g03-probe.json').exists()
    summary=dict(status='review_block_complete' if state['generation']==3 and all_probes else 'partial_review_block',reason=reason,state=state,training=phases(state['bank'],state['cached_hits']) if state['cached_hits'] else None,
        accounting=accounting(directory),wall_seconds=(time.time_ns()-launch['started_wall_ns'])/1e9,source=source_signature(),automatic_next_block=False,
        interpretation='Broader TRAIN context/anchor/forward-dose progress only; previous gen6 monitor now TRAIN, current fresh probe separate; no final reliability/rally/continuous-learning claim')
    atomic_json(directory/'block-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','prepare','run'));p.add_argument('root',type=Path);args=p.parse_args()
    if args.mode=='stage':stage(args.root)
    elif args.mode=='prepare':prepare(args.root)
    else:run(args.root)
