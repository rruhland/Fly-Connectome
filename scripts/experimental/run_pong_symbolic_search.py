"""One fixed structural-program search with complete actual-game reward fitness."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from pong_fixed_rbf import load_readout
from pong_continuous_visual import load_scenes
from pong_symbolic_fitness import evaluate, fitness_key
from run_rally_option_memory import ROOT, MODEL, BANK, EXPECTED, read, write, sha
from run_rally_repertoire_collection import select_starts, exclusions
from stateful_pong_comparison import native_task, comparison
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import aggregate
from run_rally_raw_optimizer import more
import run_practical_pong_v2 as base

DEST=ROOT/'pong-symbolic-search-20261029-r1'
CALIBRATION=ROOT/'pong-rally-continuous-actor-20261028-r1/calibration'
ADAPTER='pong_symbolic_policy_adapter'
SECONDS=900; CAMERAS=320000; RESERVED_SECONDS=150; RESERVED_CAMERAS=40000


class CameraBudget:
    def __init__(self,limit=CAMERAS):
        self.limit=limit;self.used=0;self.reserved=0;self.initials=0
    def reserve(self,grant,limit=None):
        bound=self.limit if limit is None else min(self.limit,limit)
        if self.used+self.reserved+grant>bound:return False
        self.reserved+=grant;return True
    def finish(self,grant,result):
        self.reserved-=grant
        self.used+=result.get('transition_camera_upper_bound',result['transitions'])
        self.initials+=result.get('initial_camera_upper_bound',result['initial_cameras'])
    def state(self):
        return dict(transition_upper_bound=self.used,inflight_transition_grants=self.reserved,
            initial_camera_upper_bound=self.initials,transition_limit=self.limit)


def calibration_observations():
    baseline=load_readout(MODEL/'rbf-model.npz');observations=[]
    for episode in range(8):
        stem=CALIBRATION/f'calibration_{episode:02d}'
        scenes=load_scenes(str(stem)+'-scenes.npz');raw=sp.load_npz(str(stem)+'-raw.npz').tocsr()
        assert len(scenes)==raw.shape[0]==256
        for i in range(0,256,8):
            lo,hi=raw.indptr[i:i+2];_,q=baseline.choose(raw.indices[lo:hi],raw.data[lo:hi])
            observations.append((scenes[i],q,i%5,True))
    return observations


def prepare():
    from pong_symbolic_program import Program, CONFIG
    assert not DEST.exists() and sha(MODEL/'rbf-model.npz')==EXPECTED
    bank=read(BANK);assert bank['source']==base.source_signature()
    phased={p:[e for e in bank['bank'] if base.leftward(e['seed'])==(p=='left')][:16] for p in ('left','right')}
    assert all(len(v)==16 for v in phased.values())
    entries=[e for i in range(16) for e in (phased['left'][i],phased['right'][i])]
    for e in entries:
        r=read(e['receipt']);assert r['complete'] and e['source_policy_sha256']==EXPECTED and r['learned_contents_unchanged']
    DEST.mkdir();(DEST/'fitness').mkdir();(DEST/'programs').mkdir();(DEST/'final32').mkdir();(DEST/'fresh').mkdir()
    schedule=select_starts(1943906127,63297,exclusions(DEST),32);seeds=schedule.pop('TRAIN')
    schedule.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='frozen symbolic endpoint qualification;never fitness')
    write(DEST/'fresh/schedules.json',schedule)
    files=[Path(__file__).with_name(n) for n in ('run_pong_symbolic_search.py','pong_symbolic_fitness.py',
        'pong_symbolic_program.py','pong_symbolic_policy_adapter.py','stateful_pong_comparison.py',
        'pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_continuous_visual.py','run_rally_option_memory.py',
        'pong_option_memory.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py',
        'run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')]
    assets=[MODEL/'rbf-model.npz',MODEL/'radial-cap.json',BANK,*[Path(e['receipt']) for e in entries]]
    assets += [CALIBRATION/f'calibration_{i:02d}-{kind}.npz' for i in range(8) for kind in ('raw','scenes')]
    spec=dict(grammar=CONFIG,seconds=SECONDS,transition_camera_cap=CAMERAS,initial_cameras_counted_separately=True,
        search_seconds=SECONDS-RESERVED_SECONDS,search_transition_cap=CAMERAS-RESERVED_CAMERAS,
        final_reserved_seconds=RESERVED_SECONDS,final_reserved_transition_cameras=RESERVED_CAMERAS,
        generations=16,challengers_per_generation=8,workers=4,optimizer_seed=2026102907,
        bank=entries,generation_starts=[[e['seed'] for e in entries[4*(g%8):4*(g%8)+4]] for g in range(16)],
        selection='completeactual4gameownhits;currentANDcachedzero40069contendeveryrotation;strictlymorehitsorfeweractivenodesonexactrewardtie;currentretainedonexacttie;allcompletedchallengersusableinpartiallastgeneration',
        mutation='1to3activeoperator/edge/output/poolingchanges;gainmayalsochange;max64attemptsperslot;nosweep',
        dedup='activeprogramhashglobal;256fixedunlabeledTRAINcalibrationaction-signaturesonlyskip,noinheritedfitness',
        fitness_cache='ONLYexactactiveprogramhash+seed+frozenruntimeidentity;completedgamesonly;zeroexactbaselinecachedbank',
        runtime='exact40069untilactualownhit;thenprogramresidualeachcamera;normalfullgame;notrewardfilledprefix',
        fitness_archives='compactactualaction/hit/dt/endtapewithdecisiondigest;nochallengerraw/scenecapture;digestnotreplaycertificate',
        final='ONEselectedfrozenendpoint32completeTRAINgamesfullrawcapture/replayvscached40069;LITERALzeroactivehashreusesexactcachedbaseline32withoutnewnative;total/latergainfirstparitysecond/laterfractionnotworse',
        fresh='conditional32pairsONLYifTRAINqualifiesANDremainingtransitionbudget>=64000ANDremainingwall>=90s;elseTRAINqualifiedawaitseparateallocation',
        fresh_tag=63297,source=base.source_signature(),files={str(p):sha(p) for p in files},
        assets={str(p):sha(p) for p in assets},no_backprop=True,no_expert_labels=True,no_runtime_simulator_lookahead=True,
        no_automatic_promotion=True,no_automatic_extension=True)
    identity=dict(source=spec['source'],files=spec['files'],baseline=EXPECTED,scaling=sha(MODEL/'radial-cap.json'),grammar=CONFIG)
    spec['policy_identity']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(DEST/'spec.json',spec);write(DEST/'programs/initial.json',Program.zero().to_dict())
    print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),starts=len(entries)),flush=True)


def fitness_dispatch(pool,program,seeds,cache,budget,spec,deadline,label):
    key=lambda seed:fitness_key(program.active_hash(),seed,spec['policy_identity'])
    jobs=[s for s in seeds if key(s) not in cache];active={};pending=list(jobs);results=[]
    def fill():
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            if not budget.reserve(1000,spec['search_transition_cap']):break
            seed=pending.pop(0)
            f=pool.submit(evaluate,program.to_dict(),seed,str(DEST/'fitness'),deadline-1,1000,spec['policy_identity'])
            active[f]=seed
    fill()
    while active:
        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
        for f in done:
            seed=active.pop(f)
            try:r=f.result()
            except Exception as error:
                r=dict(seed=seed,complete=False,transitions=0,initial_cameras=0,
                    transition_camera_upper_bound=1000,initial_camera_upper_bound=1,error=repr(error),unacknowledged_grant_charged=True)
            budget.finish(1000,r);results.append(r)
            if r.get('fitness_eligible'):
                cache[r['fitness_key']]=r
        write(DEST/'progress.json',dict(phase='fitness',label=label,cost=budget.state(),
            active=[dict(seed=s,transition_grant=1000) for s in active.values()],completed_new=len(results)))
        fill()
    if not all(key(seed) in cache for seed in seeds):return None
    selected=[cache[key(seed)] for seed in seeds]
    return dict(own_hits=sum(r['own_hits'] for r in selected),later_hits=sum(r['later_hits'] for r in selected),
        games_with_own_hit=sum(r['games_with_own_hit'] for r in selected),
        per_seed={str(r['seed']):r['own_hits'] for r in selected},new_games=len(results))


def qualifies(stats):
    a,b=stats['baseline'],stats['candidate'];x,y=stats['baseline_more'],stats['candidate_more']
    return (b['own_returns']>a['own_returns'] and y['later_returns']>x['later_returns']
        and b['first_incoming_returns']==a['first_incoming_returns']
        and y['tail']['2']>=x['tail']['2'] and y['later_fraction']>=x['later_fraction'])


def endpoint_dispatch(pool,payload_path,seeds,directory,budget,deadline,paired=False):
    jobs=[(i,seed,kind) for i,seed in enumerate(seeds) for kind in ((False,True) if paired else (True,))]
    active={};pending=list(jobs);results=[]
    def fill():
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            if not budget.reserve(1000):break
            i,seed,kind=pending.pop(0)
            f=pool.submit(native_task,ADAPTER,str(payload_path),str(directory),seed,i,deadline-1,kind)
            active[f]=(seed,kind)
    fill()
    while active:
        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
        for f in done:
            seed,kind=active.pop(f)
            try:r=f.result()
            except Exception as error:
                r=dict(seed=seed,complete=False,transitions=0,initial_cameras=0,
                    transition_camera_upper_bound=1000,initial_camera_upper_bound=1,error=repr(error),unacknowledged_grant_charged=True)
            budget.finish(1000,r);results.append(r)
        write(DEST/'progress.json',dict(phase='fresh' if paired else 'final32',cost=budget.state(),
            active=[dict(seed=s,candidate=c,transition_grant=1000) for s,c in active.values()],completed=len(results)))
        fill()
    return results


def run(approved):
    from pong_symbolic_program import Program
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['assets'].items())
    began=time.monotonic();deadline=began+spec['seconds'];search_deadline=began+spec['search_seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,
        deadline_monotonic=deadline,search_deadline_monotonic=search_deadline,seconds=spec['seconds']))
    observations=calibration_observations();rng=np.random.default_rng(spec['optimizer_seed']);incumbent=Program.zero()
    cache={};budget=CameraBudget(spec['transition_camera_cap']);rounds=[];seen_hash={incumbent.active_hash()};seen_sig={incumbent.action_signature(observations)}
    for entry in spec['bank']:
        r=read(entry['receipt']);h=int(r['report']['hits']);key=fitness_key(incumbent.active_hash(),r['seed'],spec['policy_identity'])
        cache[key]=dict(seed=r['seed'],complete=True,fitness_eligible=True,own_hits=h,later_hits=max(0,h-1),
            games_with_own_hit=int(h>0),fitness_key=key,cached_baseline_receipt=entry['receipt'])
    selected_path=DEST/'programs/selected.json';write(selected_path,incumbent.to_dict())
    summary=dict(status='searching',baseline_cache_games=len(cache),rounds=rounds)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for g in range(spec['generations']):
            if time.monotonic()>=search_deadline-2 or budget.used+1000>spec['search_transition_cap']:break
            seeds=spec['generation_starts'][g];parent=incumbent;challengers=[];skipped=0
            for slot in range(spec['challengers_per_generation']):
                if time.monotonic()>=search_deadline-2:break
                for _ in range(64):
                    if time.monotonic()>=search_deadline-2:break
                    child=parent.mutate(rng);h=child.active_hash()
                    if h in seen_hash:skipped+=1;continue
                    seen_hash.add(h);signature=child.action_signature(observations)
                    if signature in seen_sig:skipped+=1;continue
                    seen_sig.add(signature);challengers.append(child);break
                else:break
            write(DEST/'programs'/f'g{g:02d}.json',dict(parent=parent.to_dict(),challengers=[p.to_dict() for p in challengers],skipped=skipped))
            current=fitness_dispatch(pool,parent,seeds,cache,budget,spec,search_deadline,f'g{g:02d}-incumbent')
            if current is None:break
            best=parent;best_result=current;outcomes=[]
            zero=Program.zero();zero_result=fitness_dispatch(pool,zero,seeds,cache,budget,spec,search_deadline,f'g{g:02d}-cached-zero')
            if (zero_result['own_hits'],-zero.active_node_count)>(best_result['own_hits'],-best.active_node_count):
                best,best_result=zero,zero_result
            for i,child in enumerate(challengers):
                outcome=fitness_dispatch(pool,child,seeds,cache,budget,spec,search_deadline,f'g{g:02d}-c{i:02d}')
                if outcome is None:break
                outcomes.append(dict(program_hash=child.active_hash(),active_nodes=child.active_node_count,**outcome))
                if (outcome['own_hits'],-child.active_node_count)>(best_result['own_hits'],-best.active_node_count):
                    best,best_result=child,outcome
            incumbent=best;write(selected_path,incumbent.to_dict())
            row=dict(generation=g,starts=seeds,parent_hash=parent.active_hash(),selected_hash=incumbent.active_hash(),
                parent_fitness=current,zero_fitness=zero_result,selected_fitness=best_result,challengers=outcomes,
                generated=len(challengers),skipped_signatures_or_hashes=skipped,cost=budget.state(),elapsed_seconds=time.monotonic()-began)
            rounds.append(row);write(DEST/f'generation_{g:02d}.json',row)
            write(DEST/'checkpoint.json',dict(selected=incumbent.to_dict(),completed_generations=len(rounds),
                rng_state=rng.bit_generator.state,cost=budget.state(),cache={k:v.get('receipt_file',v.get('cached_baseline_receipt')) for k,v in cache.items()}))
            print(dict(stage='generation',generation=g,parent_hits=current['own_hits'],selected_hits=best_result['own_hits'],
                changed=incumbent.active_hash()!=parent.active_hash(),evaluated=len(outcomes),cost=budget.state()),flush=True)
            if len(outcomes)<len(challengers):break
        summary.update(status='endpoint_frozen',selected=incumbent.to_dict(),search_elapsed_seconds=time.monotonic()-began,search_cost=budget.state())
        endpoint=DEST/'programs/final.json';write(endpoint,incumbent.to_dict())
        seeds=[e['seed'] for e in spec['bank']]
        literal_zero=incumbent.active_hash()==Program.zero().active_hash()
        if literal_zero:
            baseline=[read(e['receipt']) for e in spec['bank']]
            outcome=aggregate([r['annotation'] for r in baseline]);tail=more(baseline)
            stats=dict(complete=True,baseline=outcome,candidate=outcome,baseline_more=tail,candidate_more=tail,
                qualified=False,literal_zero_active_hash_reuse=True,cached_complete_games=32,
                new_transition_cameras=0,new_initial_cameras=0,no_new_policy_replay_claim=True)
            complete=True
        else:
            results=endpoint_dispatch(pool,endpoint,seeds,DEST/'final32',budget,deadline)
            complete=len(results)==32 and all(r['complete'] for r in results)
            stats=dict(complete=complete)
            if complete:
                byseed={r['seed']:r for r in results};stats.update(comparison([read(e['receipt']) for e in spec['bank']],[byseed[s] for s in seeds]))
                stats['qualified']=qualifies(stats)
        write(DEST/'final32/summary.json',stats);summary.update(final32=stats,
            status='unchanged_literal_zero' if literal_zero else 'TRAIN32_complete' if complete else 'TRAIN32_partial')
        print(dict(stage='final32',**stats),flush=True)
        if complete and stats['qualified']:
            enough=spec['transition_camera_cap']-budget.used>=64000 and deadline-time.monotonic()>=90
            summary['fresh_budget_available']=enough
            if enough:
                fresh_seeds=read(DEST/'fresh/schedules.json')['probe'];rs=endpoint_dispatch(pool,endpoint,fresh_seeds,DEST/'fresh',budget,deadline,True)
                groups={kind:{r['seed']:r for r in rs if not r.get('not_started') and (r.get('model_sha256') is not None)==kind} for kind in (False,True)}
                done=all(len(v)==32 and all(r['complete'] for r in v.values()) for v in groups.values());fresh=dict(complete=done)
                if done:
                    a=[groups[False][s] for s in fresh_seeds];b=[groups[True][s] for s in fresh_seeds]
                    fresh.update(comparison(a,b),paired=paired_metrics(a,b));fresh['qualified']=qualifies(fresh)
                write(DEST/'fresh/summary.json',fresh);summary.update(fresh=fresh,status='fresh_complete' if done else 'fresh_partial')
            else:summary['status']='TRAINqualified_fresh_requires_separate_allocation'
    summary.update(cost=budget.state(),wall_seconds=time.monotonic()-began,post_deadline_cleanup_seconds=max(0,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),
        selected_file_sha256=sha(DEST/'programs/final.json'),no_automatic_promotion=True)
    write(DEST/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');args=p.parse_args()
    prepare() if args.operation=='prepare' else run(args.approved_spec_sha256)
