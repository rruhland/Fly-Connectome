"""One compact servo search on fixed nested TRAIN starts, actual own hits only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from stateful_reward_fitness import CameraBudget,evaluate,fitness_key
from pong_relational_feedback import RelationalPolicy,structural_choices,load_calibration_scales
from pong_relational_policy_adapter import from_payload,to_payload,model_hash
from stateful_pong_comparison import native_task,comparison
from run_pong_symbolic_search import qualifies
from run_rally_option_memory import ROOT,MODEL,BANK,EXPECTED,read,write,sha
from run_rally_repertoire_collection import select_starts,exclusions
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import aggregate
from run_rally_raw_optimizer import more
import run_practical_pong_v2 as base

DEST=ROOT/'pong-relational-search-20261029-r1'
CALIBRATION=ROOT/'pong-rally-continuous-actor-20261028-r1/calibration-report.json'
ADAPTER='pong_relational_policy_adapter'
BASELINE_ID='literal40069'
LOW=np.array([-3.,-3.,-3.,0.,0.]);HIGH=np.array([3.,3.,3.,2.,2.])


def parameter_payload(choices,coefficients,scales):
    return to_payload(RelationalPolicy(*map(int,choices),coefficients,scales))


def proposals(round_index,center,scales,rng):
    bindings=[b for b in structural_choices() if b[0] in (2,3)]
    result=[]
    if round_index==0:
        for b in bindings:
            result += [parameter_payload(b,np.zeros(5),scales),parameter_payload(b,rng.uniform(LOW,HIGH),scales)]
    else:
        if center is None:raise ValueError('second round needs a completed16-seed numeric center')
        for _ in range(16):result.append(parameter_payload(bindings[int(rng.integers(len(bindings)))],rng.uniform(LOW,HIGH),scales))
        for i in range(16):
            binding=tuple(center['choices'])
            if i>=8:
                alternatives=[b for b in bindings if b!=binding];binding=alternatives[int(rng.integers(len(alternatives)))]
            coefficients=np.clip(np.asarray(center['coefficients'])+rng.normal(0,.5,5),LOW,HIGH)
            result.append(parameter_payload(binding,coefficients,scales))
    # Fixed RNG priority breaks equal-reward screening ties without a size bias.
    order=rng.permutation(len(result));return [result[int(i)] for i in order]


def unique_population(population,seen):
    result=[]
    for payload in population:
        h=model_hash(payload)
        if h not in seen:
            seen.add(h);result.append(payload)
    return result


def prepare():
    assert not DEST.exists() and sha(MODEL/'rbf-model.npz')==EXPECTED
    scales=load_calibration_scales(CALIBRATION);bank=read(BANK);assert bank['source']==base.source_signature()
    phases={p:[e for e in bank['bank'] if base.leftward(e['seed'])==(p=='left')][:16] for p in ('left','right')}
    assert all(len(v)==16 for v in phases.values())
    entries=[e for i in range(16) for e in (phases['left'][i],phases['right'][i])]
    for e in entries:
        r=read(e['receipt']);assert r['complete'] and r['learned_contents_unchanged'] and e['source_policy_sha256']==EXPECTED
    DEST.mkdir();[(DEST/n).mkdir() for n in ('fitness','models','final32','fresh','failures')]
    schedule=select_starts(1943906127,63313,exclusions(DEST),32);seeds=schedule.pop('TRAIN')
    schedule.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='frozen relational endpoint fresh qualification;never training')
    write(DEST/'fresh/schedules.json',schedule)
    names=('run_pong_relational_search.py','stateful_reward_fitness.py','exact_visual_transport_copy.py','pong_relational_policy_adapter.py',
        'pong_relational_feedback.py','stateful_pong_comparison.py','run_pong_symbolic_search.py',
        'pong_symbolic_fitness.py','pong_symbolic_program.py','pong_fixed_rbf.py','pong_fixed_pairwise.py',
        'pong_continuous_visual.py','run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py',
        'return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py',
        'run_rally_repertoire_collection.py','run_practical_pong_v2.py')
    files=[Path(__file__).with_name(n) for n in names]
    assets=[MODEL/'rbf-model.npz',MODEL/'radial-cap.json',BANK,CALIBRATION,*[Path(e['receipt']) for e in entries]]
    spec=dict(seconds=900,search_seconds=750,transition_camera_cap=320000,search_transition_cap=280000,
        initial_cameras_counted_separately=True,final_reserved_seconds=150,final_reserved_transition_cameras=40000,
        rounds=2,candidates_per_round=32,survivors=[32,8,2],stages=[4,8,16],workers=4,optimizer_seed=2026102908,
        train_seeds=[e['seed'] for e in entries[:16]],bank=entries,scales=scales.tolist(),
        observed_calibration_cardinality_counts={'0':16,'1':62,'2':327,'3':1643},learned_cardinalities=[2,3],
        round0='all16bindingchoicesonce5numericzeroandonceuniformboxdraw',
        round1='16globalboxdraws+16clippednormalperturbationssigma.5;8retainbinding8changebinding;centerfull16incumbentorelsebestcompleted16challenger',
        numeric_lower=LOW.tolist(),numeric_upper=HIGH.tolist(),screen_ties='fixedRNGproposalpriority;noactive-node-sizecomparison',
        selection='samefixednested4then8then16seedsEVERYround;top8newchallengers4→8,top2newchallengers8→16;ONLYcomplete16scoresreplaceincumbent;incumbentandliteral40069alwayscontend;incumbentretainedexacttie',
        incomplete='allstartedgamesretained/charged;onlycompletesameseedsetsfitness;lastcomplete16incumbentonly;secondroundrequirescompleted16center',
        runtime='40069untilactualownhit;currentobservedanonymouspairrelation;actualpreviouscommandhysteresis;fallbackcardinalitymismatch',
        cache='exactparameterhash+seed+frozenruntimeidentity;completegamesonly;nocrosspolicyaction-signaturecredit',
        fitness='completeactualgamescalarownhitcountonly;noshaping/noexpertlabels',
        final='freezeoneendpoint;32matchedTRAINgamesfullraw/replayagainstcached40069;literalbaselineendpointreusewithoutnewnative;total/latergainfirstparitysecond/laterfractionnotworse',
        fresh='onlyTRAINqualificationANDremaining>=64000transitioncamsAND>=90seconds;elseexplicitseparateallocation',
        fresh_tag=63313,source=base.source_signature(),files={str(p):sha(p) for p in files},assets={str(p):sha(p) for p in assets},
        no_backprop=True,no_expert_labels=True,no_runtime_simulator_lookahead=True,no_automatic_promotion=True,no_automatic_extension=True)
    spec['identity']=hashlib.sha256(json.dumps({k:spec[k] for k in ('source','files','assets','scales')},sort_keys=True).encode()).hexdigest()
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def dispatch(pool,worker,jobs,budget,deadline,limit,label):
    pending=list(jobs);active={};results=[]
    def fill():
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            if not budget.reserve(1000,limit):break
            job=pending.pop(0);active[pool.submit(worker,*job['args'])]=job
    fill()
    while active:
        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
        for f in done:
            job=active.pop(f)
            try:r=f.result()
            except Exception as error:
                r=dict(seed=job['seed'],complete=False,transitions=0,initial_cameras=0,
                    transition_camera_upper_bound=1000,initial_camera_upper_bound=1,error=repr(error),unacknowledged_grant_charged=True)
                write(DEST/'failures'/(hashlib.sha256(job['key'].encode()).hexdigest()+'.json'),dict(job=job,result=r))
            budget.finish(1000,r);r['job_key']=job['key'];results.append(r)
        write(DEST/'progress.json',dict(phase=label,cost=budget.state(),completed=len(results),
            active=[dict(key=j['key'],seed=j['seed'],transition_grant=1000) for j in active.values()]))
        fill()
    return results


def complete_score(payload,seeds,cache,spec):
    h=BASELINE_ID if payload is None else model_hash(payload)
    keys=[fitness_key(h,s,spec['identity']) for s in seeds]
    if not all(k in cache for k in keys):return None
    return sum(cache[k]['own_hits'] for k in keys)


def evaluate_stage(pool,population,seeds,cache,budget,spec,deadline,label):
    jobs=[]
    for payload in population:
        h=model_hash(payload)
        for seed in seeds:
            key=fitness_key(h,seed,spec['identity'])
            if key not in cache:
                jobs.append(dict(key=key,seed=seed,args=(ADAPTER,payload,seed,str(DEST/'fitness'),deadline-1,1000,spec['identity'])))
    for r in dispatch(pool,evaluate,jobs,budget,deadline,spec['search_transition_cap'],label):
        if r.get('fitness_eligible'):cache[r['fitness_key']]=r
    return [(p,complete_score(p,seeds,cache,spec)) for p in population if complete_score(p,seeds,cache,spec) is not None]


def rank_complete(scored):
    # Python's stable sort preserves the fixed pre-outcome proposal priority.
    return sorted(scored,key=lambda pair:-pair[1])


def endpoint_jobs(path,seeds,folder,deadline,paired=False):
    return [dict(key=f'{kind}:{seed}',seed=seed,args=(ADAPTER,str(path),str(folder),seed,i,deadline-1,kind))
        for i,seed in enumerate(seeds) for kind in ((False,True) if paired else (True,))]


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['assets'].items())
    began=time.monotonic();deadline=began+spec['seconds'];search_deadline=began+spec['search_seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,
        deadline_monotonic=deadline,search_deadline_monotonic=search_deadline,seconds=spec['seconds']))
    cache={};budget=CameraBudget(spec['transition_camera_cap']);rng=np.random.default_rng(spec['optimizer_seed'])
    incumbent=None;center=None;rounds=[];train=spec['train_seeds'];seen=set()
    for e in spec['bank'][:16]:
        r=read(e['receipt']);key=fitness_key(BASELINE_ID,e['seed'],spec['identity'])
        cache[key]=dict(own_hits=int(r['report']['hits']),receipt_file=e['receipt'],cached_baseline=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for round_index in range(spec['rounds']):
            if time.monotonic()>=search_deadline-2:break
            population=proposals(round_index,center,spec['scales'],rng)
            population=unique_population(population,seen)
            write(DEST/f'round{round_index}-proposals.json',dict(population=population,incumbent=incumbent,rng_state=rng.bit_generator.state))
            stages=[];scored=[]
            for size,survivors in zip(spec['stages'],(8,2,None)):
                scored=rank_complete(evaluate_stage(pool,population,train[:size],cache,budget,spec,search_deadline,f'round{round_index}-stage{size}'))
                row=dict(size=size,completed=len(scored),incumbent_hits=complete_score(incumbent,train[:size],cache,spec),
                    baseline_hits=complete_score(None,train[:size],cache,spec),
                    scored=[dict(model_hash=model_hash(p),own_hits=h) for p,h in scored])
                stages.append(row);write(DEST/f'round{round_index}-stage{size}.json',row)
                print(dict(stage=f'round{round_index}-stage{size}',completed=len(scored),
                    best_hits=scored[0][1] if scored else None,incumbent_hits=row['incumbent_hits'],baseline_hits=row['baseline_hits'],cost=budget.state()),flush=True)
                if not scored:break
                if survivors is not None:population=[p for p,_ in scored[:survivors]]
            if not stages or stages[-1]['size']!=16 or not scored:break
            before=complete_score(incumbent,train,cache,spec);baseline=complete_score(None,train,cache,spec)
            best_payload,best_hits=scored[0]
            if baseline>before:incumbent=None;before=baseline
            if best_hits>before:incumbent=best_payload
            center=incumbent if incumbent is not None else best_payload
            record=dict(round=round_index,stages=stages,selected=incumbent,
                selected_hits=complete_score(incumbent,train,cache,spec),baseline_hits=baseline,cost=budget.state(),elapsed_seconds=time.monotonic()-began)
            rounds.append(record);write(DEST/f'round{round_index}-result.json',record)
            write(DEST/'checkpoint.json',dict(selected=incumbent,rounds=rounds,cost=budget.state(),rng_state=rng.bit_generator.state,
                cache={k:v['receipt_file'] for k,v in cache.items()}))
        summary=dict(status='endpoint_frozen',rounds=rounds,selected=incumbent,search_cost=budget.state(),search_elapsed_seconds=time.monotonic()-began)
        baseline=[read(e['receipt']) for e in spec['bank']];seeds=[e['seed'] for e in spec['bank']]
        if incumbent is None:
            outcome=aggregate([r['annotation'] for r in baseline]);tail=more(baseline)
            final=dict(complete=True,baseline=outcome,candidate=outcome,baseline_more=tail,candidate_more=tail,
                qualified=False,literal_baseline_reuse=True,new_transition_cameras=0,new_initial_cameras=0)
            summary['status']='unchanged_literal_baseline'
        else:
            path=DEST/'models/final-policy.npz';from_payload(incumbent).save(path)
            write(DEST/'models/frozen-endpoint.json',dict(payload=incumbent,model_sha256=sha(path),
                model_hash=model_hash(incumbent),selected_from_complete16_only=True,before_final_native=True))
            rs=dispatch(pool,native_task,endpoint_jobs(path,seeds,DEST/'final32',deadline),budget,deadline,budget.limit,'final32')
            done=len(rs)==32 and all(r['complete'] for r in rs);final=dict(complete=done)
            if done:
                byseed={r['seed']:r for r in rs};final.update(comparison(baseline,[byseed[s] for s in seeds]));final['qualified']=qualifies(final)
            summary['status']='TRAIN32_complete' if done else 'TRAIN32_partial'
            if done and final['qualified']:
                enough=budget.limit-budget.used>=64000 and deadline-time.monotonic()>=90;summary['fresh_budget_available']=enough
                if enough:
                    fresh_seeds=read(DEST/'fresh/schedules.json')['probe']
                    rs=dispatch(pool,native_task,endpoint_jobs(path,fresh_seeds,DEST/'fresh',deadline,True),budget,deadline,budget.limit,'fresh')
                    groups={kind:{r['seed']:r for r in rs if not r.get('not_started') and r['job_key'].startswith(str(kind)+':')} for kind in (False,True)}
                    done=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fresh=dict(complete=done)
                    if done:
                        a=[groups[False][s] for s in fresh_seeds];b=[groups[True][s] for s in fresh_seeds]
                        fresh.update(comparison(a,b),paired=paired_metrics(a,b));fresh['qualified']=qualifies(fresh)
                    write(DEST/'fresh/summary.json',fresh);summary.update(fresh=fresh,status='fresh_complete' if done else 'fresh_partial')
                else:summary['status']='TRAINqualified_fresh_requires_separate_allocation'
        write(DEST/'final32/summary.json',final);summary['final32']=final
        print(dict(stage='final32',**final),flush=True)
    summary.update(cost=budget.state(),wall_seconds=time.monotonic()-began,post_deadline_cleanup_seconds=max(0,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),no_automatic_promotion=True)
    write(DEST/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    prepare() if a.operation=='prepare' else run(a.approved_spec_sha256)
