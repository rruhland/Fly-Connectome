"""Fixed candidate-only duration validation after BOTH acquisition endpoints freeze."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

import compact_public_pong as worker
from stateful_reward_fitness import CameraBudget
from return_metrics import aggregate
from run_rally_option_memory import ROOT,MODEL,read,write,sha
from run_pong_relational_categorical_replication import validate_schedules
import run_practical_pong_v2 as base

ACQUISITION=ROOT/'pong-relational-categorical-replication-20261029-r1'
DEST=ROOT/'pong-relational-replication-validation-20261029-r1'


def require_learned_endpoints(barrier):
    if not barrier.get('both_endpoints_frozen'):raise ValueError('both endpoints must freeze before either validation')
    endpoints=barrier['endpoints']
    if set(endpoints)!= {'0','1'}:raise ValueError('exactly two frozen acquisition endpoints required')
    for replica,e in endpoints.items():
        if not e['acquisition_complete'] or not e['learned_endpoint'] or e['model_path'] is None:
            raise ValueError(f'replica {replica} has no completed learned acquisition; do not spend fresh budget')
    return endpoints


def ordered_assignments(schedules):
    jobs=[]
    for stratum in ('long60','standard20'):
        rows={r:[a for a in schedules[r]['fresh']['assignments'] if a['stratum']==stratum] for r in ('0','1')}
        if any(len(v)!=32 for v in rows.values()):raise ValueError('32 fixed starts per duration and replica required')
        for i in range(32):
            for replica in ('0','1'):jobs.append(dict(rows[replica][i],replica=replica))
    return jobs


def aggregate_time_bins(rows):
    result=[]
    for i in range(3):
        bins=[r['time_bins'][i] for r in rows]
        total={k:sum(b[k] for b in bins) for k in ('exposure_seconds','own_opportunities','own_returns',
            'first_opportunities','first_returns','later_opportunities','later_returns')}
        if abs(total['exposure_seconds'])<1e-9:total['exposure_seconds']=0.
        total.update(start_seconds=20*i,end_seconds=20*(i+1),
            games_with_exposure=sum(b['exposure_seconds']>1e-9 for b in bins),
            later_fraction=total['later_returns']/total['later_opportunities'] if total['later_opportunities'] else None)
        result.append(total)
    return result


def summarize_stratum(rows,expected_games):
    complete=len(rows)==expected_games and all(r['complete'] and r['fitness_eligible'] for r in rows)
    actual=aggregate([r['annotation'] for r in rows]);first=actual['first_incoming_returns']
    later_returns=actual['own_returns']-first
    later_opportunities=actual['own_opportunities']-actual['first_incoming_opportunities']
    hits=[int(r['own_hits']) for r in rows]
    return dict(complete=complete,games=len(rows),metrics=actual,later_returns=later_returns,
        later_opportunities=later_opportunities,later_fraction=later_returns/later_opportunities if later_opportunities else None,
        timeouts=sum(r['report']['truncations'] for r in rows),tail_counts={str(k):sum(h>=k for h in hits) for k in (1,2,3,5,10,15,20)},
        time_bins=aggregate_time_bins(rows),streaming_checked_rows=sum(r['streaming_audit']['checked_rows'] for r in rows),
        full_archived_replay=False)


def replica_summary(rows):
    groups={name:summarize_stratum([r for r in rows if r['validation_stratum']==name],32) for name in ('long60','standard20')}
    combined=aggregate([r['annotation'] for r in rows]);long,short=groups['long60'],groups['standard20']
    complete=long['complete'] and short['complete']
    first=combined['first_incoming_return_fraction']
    meets=(complete and first is not None and first>=.95 and long['later_opportunities']>=200
        and long['later_fraction'] is not None and long['later_fraction']>=.95
        and short['later_fraction'] is not None and short['later_fraction']>=.95)
    return dict(complete=complete,strata=groups,combined_first_returns=combined['first_incoming_returns'],
        combined_first_opportunities=combined['first_incoming_opportunities'],combined_first_fraction=first,
        long_later_exposure_sufficient=long['later_opportunities']>=200,meets_declared_point_targets=bool(meets),
        interpretation='Inspect late-bin rates and exposure separately; pooled success does not establish consistent late play.')


def prepare():
    if DEST.exists():raise FileExistsError('preserve every prepared validation and attempted clock')
    acquisition=read(ACQUISITION/'spec.json');barrier_path=ACQUISITION/'endpoints-frozen.json'
    for replica in ('0','1'):
        summary_path=ACQUISITION/f'replica{replica}/summary.json'
        if summary_path.exists():
            result=read(summary_path);endpoint=result['endpoint']
            if result['status']!='acquisition_complete' or not endpoint['learned_endpoint']:
                DEST.mkdir();write(DEST/'validation-skipped.json',dict(status='failed_acquisition_no_validation',
                    replica=replica,reason='Acquisition was incomplete or retained literal baseline.',new_cameras=0))
                print(dict(status='failed_acquisition_no_validation',replica=replica,new_cameras=0),flush=True);return
    if not barrier_path.exists():raise RuntimeError('wait for the explicit both-endpoints freeze barrier; no validation started')
    barrier=read(barrier_path)
    try:endpoints=require_learned_endpoints(barrier)
    except ValueError as error:
        DEST.mkdir();write(DEST/'validation-skipped.json',dict(status='failed_acquisition_no_validation',reason=str(error),new_cameras=0));raise
    assert barrier['spec_sha256']==sha(ACQUISITION/'spec.json') and barrier['schedules_sha256']==acquisition['schedules_sha256']
    assert acquisition['source']==base.source_signature()
    assert all(sha(p)==h for p,h in {**acquisition['files'],**acquisition['assets']}.items())
    schedules=read(acquisition['schedules_path']);excluded=read(ACQUISITION/'seed-exclusion-proof.json')['canonical_ids']
    validate_schedules(schedules,excluded)
    assets=[ACQUISITION/'spec.json',Path(acquisition['schedules_path']),barrier_path,
        MODEL/'rbf-model.npz',MODEL/'radial-cap.json']
    for replica,e in endpoints.items():
        endpoint_path=ACQUISITION/f'replica{replica}/endpoint.json';summary_path=ACQUISITION/f'replica{replica}/summary.json'
        assert sha(endpoint_path)==e['endpoint_receipt_sha256'] and sha(summary_path)==e['acquisition_summary_sha256']
        assert sha(e['model_path'])==e['model_sha256']
        assets += [endpoint_path,summary_path,Path(e['model_path'])]
    DEST.mkdir();(DEST/'failures').mkdir()
    for replica in ('0','1'):
        (DEST/f'replica{replica}').mkdir()
        for stratum in ('long60','standard20'):(DEST/f'replica{replica}'/stratum).mkdir()
    names=set(worker.FILES)|{Path(__file__).name,'run_pong_relational_categorical_replication.py','stateful_reward_fitness.py'}
    files=[Path(__file__).with_name(n) for n in sorted(names)];assignments=ordered_assignments(schedules)
    assert sum(a['max_transition_cameras'] for a in assignments)==256000 and len(assignments)==128
    spec=dict(seconds=1200,transition_camera_cap=256000,initial_camera_cap=128,workers=4,
        acquisition_spec=str(ACQUISITION/'spec.json'),acquisition_spec_sha256=sha(ACQUISITION/'spec.json'),
        freeze_barrier=str(barrier_path),freeze_barrier_sha256=sha(barrier_path),endpoints=endpoints,
        assignments=assignments,dispatch='Interleave replicas within each fixed duration stratum: long60, then standard20.',
        candidate_only=True,no_baseline_rerun=True,no_model_selection=True,no_refit=True,no_new_starts=True,
        targets='Each replica: at least 95% first returns across its 64 starts; at least 95% later success in BOTH duration strata; at least 200 later opportunities in long32.',
        reporting='Report replicas separately, all duration strata, tails, timeouts, and 0–20 / 20–40 / 40–60 second exposure and events. No selecting the better replica.',
        audit='Independent deserialized controllers compared on every actual public input and actual feedback. Streaming checks; not full archived replay. Bounded miss tails only.',
        insufficient_exposure='Incomplete evidence; no outcome-dependent new games or time extension.',
        no_automatic_promotion=True,source=base.source_signature(),files={str(p):sha(p) for p in files},assets={str(p):sha(p) for p in assets})
    spec['identity']=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    require_learned_endpoints(dict(endpoints=spec['endpoints'],both_endpoints_frozen=True))
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in {**spec['files'],**spec['assets']}.items())
    began=time.monotonic();deadline=began+spec['seconds'];budget=CameraBudget(spec['transition_camera_cap']);results=[]
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=spec['seconds']))
    pending=list(spec['assignments']);active={}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        def fill():
            while pending and len(active)<4 and time.monotonic()<deadline-2:
                job=pending[0];grant=job['max_transition_cameras']
                if not budget.reserve(grant):break
                pending.pop(0);endpoint=spec['endpoints'][job['replica']]
                args=(endpoint['model_path'],job['seed'],job['max_episode_ticks'],
                    str(DEST/f"replica{job['replica']}"/job['stratum']),deadline-1,grant,spec['identity'])
                active[pool.submit(worker.evaluate,*args)]=job
        def progress():
            write(DEST/'progress.json',dict(completed=len(results),cost=budget.state(),
                active=[dict(replica=j['replica'],seed=j['seed'],stratum=j['stratum'],transition_grant=j['max_transition_cameras']) for j in active.values()]))
        fill();progress()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future);grant=job['max_transition_cameras'];endpoint=spec['endpoints'][job['replica']]
                try:
                    r=future.result()
                    if (r['seed']!=job['seed'] or r['max_episode_ticks']!=job['max_episode_ticks']
                            or r['runtime_identity']!=spec['identity'] or r['model_sha256']!=endpoint['model_sha256']):
                        raise ValueError('worker identity differs from frozen validation assignment')
                    if r['transition_camera_upper_bound']>grant:raise ValueError('worker exceeded declared camera grant')
                    if r['complete'] and (not r['streaming_audit']['passed'] or r['streaming_audit']['checked_rows']!=r['transitions']):
                        raise ValueError('completed game lacks every streaming check')
                except Exception as error:
                    r=dict(seed=job['seed'],complete=False,transitions=0,initial_cameras=0,
                        transition_camera_upper_bound=grant,initial_camera_upper_bound=1,error=repr(error),unacknowledged_grant_charged=True)
                    write(DEST/'failures'/f"replica{job['replica']}-game{job['index']:02d}.json",dict(job=job,result=r))
                budget.finish(grant,r);r.update(validation_replica=job['replica'],validation_stratum=job['stratum']);results.append(r)
            fill();progress()
    replicas={}
    for replica in ('0','1'):
        rows=[r for r in results if r['validation_replica']==replica and 'annotation' in r]
        replicas[replica]=replica_summary(rows);write(DEST/f'replica{replica}/summary.json',replicas[replica])
    summary=dict(status='validation_complete' if all(r['complete'] for r in replicas.values()) else 'validation_partial',
        replicas=replicas,both_meet_declared_point_targets=all(r['meets_declared_point_targets'] for r in replicas.values()),
        cost=budget.state(),started_jobs=len(results),wall_seconds=time.monotonic()-began,
        post_deadline_cleanup_seconds=max(0,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),
        no_model_selection=True,no_refit=True,no_outcome_dependent_extension=True,no_automatic_promotion=True)
    assert budget.used<=256000 and budget.initials<=128
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    prepare() if a.operation=='prepare' else run(a.approved_spec_sha256)
