"""One concrete aggregate/discover/soft-margin cycle around an earned rally head.

Reuses native collector, prefix discovery worker and convex fitter. Earlier
rewarded actions are recomputed against the current carry, never treated as its
current q values or privileged labels. No fresh qualification is fitted here.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from run_rally_baseline_pong import collect_one
from run_rally_discovery_pong import case_task
from pong_rally_imitation import diagnostics
from pong_rally_margin import fit
from return_metrics import aggregate
import run_practical_pong_v2 as base


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_aggregate_cycle.py','run_rally_discovery_pong.py','pong_rally_credit.py','pong_rally_margin.py','pong_rally_imitation.py','run_rally_baseline_pong.py')}


def read_rows(folder):
    rows=[]
    for p in sorted((folder/'episodes').glob('*.json')):
        r=base.read(p);r=dict(r,receipt=str(p),feature_path=str(folder/r['feature_file']));rows.append(r)
    return rows


def plan_bursts(n,positions):
    seen=set();out=[]
    for hold in (20,40,80):
        for t in positions:
            effective=min(hold,n-t)
            for action in range(3):
                key=t,effective,action
                if key not in seen:seen.add(key);out.append(dict(transition=t,hold=hold,effective_hold=effective,action=action))
    return out


def dispatch(pool,jobs,deadline,submit,callback=None):
    pending=iter(jobs);active={};results=[]
    def fill():
        while len(active)<4 and time.monotonic()<deadline-2:
            try:j=next(pending)
            except StopIteration:break
            active[submit(pool,j,deadline)]=j
    fill()
    while active:
        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
        for f in done:active.pop(f);results.append(f.result())
        if callback:callback(results)
        fill()
    return results


def run(directory,carry):
    directory=Path(directory);carry=Path(carry);new=directory/'new_train';new_summary=base.read(new/'summary.json');assert new_summary['status']=='complete'
    assert not (directory/'cycle-clock.json').exists();began=time.monotonic();base.atomic_json(directory/'cycle-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+600
    for source,name in [('proposal.npy','carry.npy'),('proposal_theta.npy','carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json'),('old-transform.npy','old-transform.npy'),('extra-transform.npy','extra-transform.npy')]:(directory/name).write_bytes((carry/source).read_bytes())
    theta=np.load(directory/'carry_theta.npy');weight_hash=base.sha(directory/'carry.npy');assert weight_hash==base.sha(new/'carry.npy')
    old=read_rows(carry/'native_candidate');added=read_rows(new);rows=old+added;byseed={r['seed']:r for r in rows};assert len(rows)==len(byseed)==96
    assert all(r['complete'] and r['weights_sha256']==weight_hash for r in rows)
    original_root=base.EXPERIMENTS/'pong-rally-baseline-20261027-r1';original={r['seed']:r for r in read_rows(original_root)}
    old_discovery=base.EXPERIMENTS/'pong-rally-discovery-20261027-r1';old_ds=base.read(old_discovery/'summary.json');old_cases={c['seed']:c for c in base.read(old_discovery/'schedules.json')['cases']}
    memories=[]
    for w in old_ds['witnesses']:
        target=old_cases[w['seed']]['previous_hits']+1
        if byseed[w['seed']]['report']['hits']>=target:continue
        with np.load(old_discovery/w['feature_file']) as f:length=len(f['X'])
        hit_indices=[i for i,r in enumerate(original[w['seed']]['records'][:w['transition']]) if r['hits']]+[length-1]
        memories.append(dict(seed=w['seed'],phase=w['phase'],feature_path=str(old_discovery/w['feature_file']),hit_indices=hit_indices,source='old_burst_success',target_hits=target))
    for seed,r in original.items():
        if byseed[seed]['report']['hits'] or not r['report']['hits']:continue
        index=next(i for i,v in enumerate(r['records']) if v['hits']);memories.append(dict(seed=seed,phase='left' if base.leftward(seed) else 'right',feature_path=r['feature_path'],slice_end=index+1,hit_indices=[index],source='old_learned_first_return',target_hits=1))
    known={m['seed'] for m in memories};new_ids={r['seed'] for r in added};rng=np.random.default_rng(2861759031);cases=[]
    for role in ('later','opening'):  # A later case first retains the established delta-hit null check.
        eligible=[r for r in rows if r['report']['losses'] and (r['report']['hits']>0)==(role=='later') and r['seed'] not in known]
        selected=[]
        for phase in ('left','right'):
            values=[r for r in eligible if base.leftward(r['seed'])==(phase=='left')];values=[values[int(i)] for i in rng.permutation(len(values))];values.sort(key=lambda r:r['seed'] not in new_ids);selected+=values[:6]
        if len(selected)<12:
            rest=[r for r in eligible if r['seed'] not in {v['seed'] for v in selected}];rest.sort(key=lambda r:r['seed'] not in new_ids);selected+=rest[:12-len(selected)]
        for r in selected:
            hits=[i for i,v in enumerate(r['records']) if v['hits']];last=hits[-1] if hits else -1;floor=max(last+1,5 if role=='opening' else 0)
            positions=sorted({max(floor,r['transitions']-lead) for lead in (40,80,120) if max(floor,r['transitions']-lead)<r['transitions']})
            cases.append(dict(seed=r['seed'],phase='left' if base.leftward(r['seed']) else 'right',role=role,baseline_receipt=r['receipt'],baseline_feature=r['feature_path'],baseline_transitions=r['transitions'],previous_hits=r['report']['hits'],last_own_hit_transition=last,positions=positions,k1_control=False,burst_plan=plan_bursts(r['transitions'],positions)))
    assert cases and cases[0]['previous_hits']>0
    discovery=directory/'discovery';discovery.mkdir()
    for name in ('cases','branches','witnesses'):(discovery/name).mkdir()
    for name in ('carry.npy','carry_theta.npy','search-transform.npy','radial-cap.json'):(discovery/name).write_bytes((directory/name).read_bytes())
    reservation=sum(c['baseline_transitions']+sum(c['baseline_transitions']-p['transition'] for p in c['burst_plan']) for c in cases)+max(cases[0]['baseline_transitions']-t for t in cases[0]['positions']);assert reservation<=64000
    schedule=dict(train=[r['seed'] for r in rows],new_train=base.read(new/'schedules.json'),cases=cases,discovery_camera_reservation=reservation);base.atomic_json(directory/'schedules.json',schedule)
    manifest=dict(files=pins(),source=base.source_signature(),carry_source=str(carry/'proposal.npy'),carry_sha256=weight_hash,new_train_root=str(new),cached_current_receipts=[r['receipt'] for r in old],
        new_current_receipts=[r['receipt'] for r in added],memories=memories,discovery_seconds=300,native_seconds=300,cycle_seconds=600,
        budgets=dict(new_train_camera_cap=32000,discovery_camera_cap=64000,native_camera_cap=96000),selection='up to12opening+12later, balancephasewhereavailable, prefernewcontexts, skipknownrewardedrepairmemory',
        memory='keepalloldprovenance; fitunresolvedoldwitnesses/currentfirstregressions; current-relativecontrasts arepositive labels, onlycurrentmatches arepreserved',
        no_holdout_fitted=True,no_new_architecture=True,no_backprop=True)
    base.atomic_json(directory/'manifest.json',manifest);base.atomic_json(directory/'baseline96.json',dict(aggregate=aggregate([r['annotation'] for r in rows]),games=96,cached_games=64,new_games=32,first_returns=sum(r['report']['hits']>0 for r in rows)))
    print(json.dumps(dict(staged=True,cases=len(cases),roles={role:sum(c['role']==role for c in cases) for role in ('opening','later')},new_target_contexts=sum(c['seed'] in new_ids for c in cases),memory=len(memories),memory_sources={s:sum(m['source']==s for m in memories) for s in ('old_burst_success','old_learned_first_return')},camera_reservation=reservation)),flush=True)
    discovery_deadline=began+300
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        found=dispatch(pool,list(enumerate(cases)),discovery_deadline,lambda pool,item,limit:pool.submit(case_task,str(discovery),item[1],item[0],limit-2),
            lambda results:base.atomic_json(directory/'progress.json',dict(phase='discovery',cases=len(results),repaired=sum(bool(r['witnesses']) for r in results),policy_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results))))
    witnesses=[w for r in found for w in r['witnesses']];discovery_cost=sum(r['prefix_cameras']+r['suffix_cameras'] for r in found);assert discovery_cost<=64000
    base.atomic_json(directory/'discovery-result.json',dict(cases=len(found),repaired=len({w['seed'] for w in witnesses}),witnesses=witnesses,policy_cameras=discovery_cost,initials=sum(r['initial_cameras'] for r in found),censored=sum(b['outcome'] is None for r in found for b in r['branches'])))
    for w in witnesses:
        with np.load(discovery/w['feature_file']) as f:length=len(f['X'])
        source=byseed[w['seed']];hit_indices=[i for i,r in enumerate(source['records'][:w['transition']]) if r['hits']]+[length-1]
        memories.append(dict(seed=w['seed'],phase=w['phase'],feature_path=str(discovery/w['feature_file']),hit_indices=hit_indices,source='current_burst_success',target_hits=len(hit_indices)))
    positive=[];matching=[];preservation=[];allcases=[]
    for memory in memories:
        with np.load(memory['feature_path']) as f:X=f['X'];actions=f['actions']
        if memory.get('slice_end'):X=X[:memory['slice_end']];actions=actions[:memory['slice_end']]
        current_q=X@theta.T;contrast=actions!=np.argmax(current_q,axis=1);start=0
        for ordinal,end in enumerate(memory['hit_indices']):
            group=('first_' if ordinal==0 else 'later_')+memory['phase'];segment=np.arange(start,end+1);start=end+1
            for mask,target in ((contrast,positive),(~contrast,matching)):
                index=segment[mask[segment]]
                if len(index):target.append(dict(seed=memory['seed'],group=group,X=X[index],actions=actions[index]))
    for r in rows:
        with np.load(r['feature_path']) as f:X=f['X'];actions=f['actions']
        phase='left' if base.leftward(r['seed']) else 'right';allcases.append(dict(seed=r['seed'],group=phase,X=X,actions=actions));start=0
        for ordinal,end in enumerate(i for i,row in enumerate(r['records']) if row['hits']):
            preservation.append(dict(seed=r['seed'],group=('first_' if ordinal==0 else 'later_')+phase,X=X[start:end+1],actions=actions[start:end+1]));start=end+1
    assert positive;normal_path=base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json';normal=base.read(normal_path);normal={k:normal[k] for k in ('input_rms','kappa')}
    delta,report=fit(theta,positive,preservation,matching,normal);report['whole_baseline_distribution']=diagnostics(theta,delta,allcases);report.update(memory_sources={s:sum(m['source']==s for m in memories) for s in sorted({m['source'] for m in memories})},new_reward_witnesses=len(witnesses),current_earned_segments=len(preservation));base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    proposal=theta+delta;base.save_array(directory/'proposal_theta.npy',proposal);base.save_array(directory/'proposal.npy',proposal[:,:214]@np.load(directory/'old-transform.npy')+proposal[:,214:]@np.load(directory/'extra-transform.npy'));base.save_array(directory/'applied_delta.npy',delta)
    native=directory/'native_candidate';native.mkdir();(native/'episodes').mkdir();(native/'features').mkdir()
    for source,name in [('proposal.npy','carry.npy'),('proposal_theta.npy','carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json')]:(native/name).write_bytes((directory/source).read_bytes())
    native_start=time.monotonic();native_deadline=min(deadline,native_start+300);base.atomic_json(directory/'native-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=native_start,deadline_monotonic=native_deadline))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        outcomes=dispatch(pool,list(enumerate(schedule['train'])),native_deadline,lambda pool,item,limit:pool.submit(collect_one,str(native),item[1],item[0],limit-1,False),
            lambda results:base.atomic_json(directory/'progress.json',dict(phase='native96',games=len(results),policy_cameras=sum(r['transitions'] for r in results))))
    candidate=aggregate([r['annotation'] for r in outcomes]);baseline_metrics=aggregate([r['annotation'] for r in rows]);by_result={r['seed']:r for r in outcomes}
    def extra_metrics(values):
        a=aggregate([r['annotation'] for r in values]);returns=a['own_returns']-a['first_incoming_returns'];opps=a['own_opportunities']-a['first_incoming_opportunities']
        return dict(later_returns=returns,later_opportunities=opps,later_fraction=returns/opps if opps else None,tail={str(k):sum(r['report']['hits']>=k for r in values) for k in range(1,8)},timeouts=sum(r['report']['truncations'] for r in values),opponent_ended=sum(r['report']['wins'] for r in values))
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=baseline_metrics,candidate=candidate,baseline_more=extra_metrics(rows),candidate_more=extra_metrics(outcomes),
        discovery_repaired=len({w['seed'] for w in witnesses}),new_witness_actual_extra_returns=sum(by_result[w['seed']]['report']['hits']>byseed[w['seed']]['report']['hits'] for w in witnesses if w['seed'] in by_result),
        cost=dict(new_train_cameras=new_summary['train_transition_cameras'],discovery_cameras=discovery_cost,candidate_cameras=sum(r['transitions'] for r in outcomes),initials=new_summary['initial_cameras']+sum(r['initial_cameras'] for r in found)+sum(r['initial_cameras'] for r in outcomes)),
        reused_cached_games=64,no_holdout_fitted=True,no_automatic_promotion=True,cycle_wall_seconds=time.monotonic()-began,end_to_end_since_new_collection=time.monotonic()-base.read(new/'work-clock.json')['started_monotonic'],
        source_frozen=manifest['files']==pins() and manifest['source']==base.source_signature(),carry_immutable=base.sha(directory/'carry.npy')==weight_hash,proposal_sha256=base.sha(directory/'proposal.npy'))
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--carry',default='/workspace/m1b-experiments/pong-rally-margin-20261027-r1');a=p.parse_args();run(a.directory,a.carry)
