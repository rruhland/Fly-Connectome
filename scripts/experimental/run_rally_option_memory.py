"""One reviewed paired option-memory acquisition and actual stateful deployment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_action_credit import clone_session, step_action
from pong_continuous_visual import canonical_scene, load_scenes, save_scenes
from pong_option_memory import (OptionMemory, OptionMemoryEvaluator, MemoryController,
    scene_distance, save_memory, load_memory, HOLD)
from rally_raw_capture import BOOKKEEPING, SparseRows
from return_metrics import ReturnCounter, aggregate, native_count_status
from run_rally_raw_optimizer import more
from run_rally_raw_fresh_screen import paired_metrics
from run_rally_repertoire_collection import exclusions, select_starts
import run_practical_pong_v2 as base

ROOT = Path('/workspace/m1b-experiments')
DEST = ROOT/'pong-local-option-memory-20261028-r1'
MODEL = ROOT/'pong-rally-joint-output-20261028-r1/full'
BANK = ROOT/'pong-rally-TRAIN-bank-20261028-r7/admission.json'
CALIBRATION = ROOT/'pong-rally-continuous-actor-20261028-r1/calibration'
SCREEN8 = ROOT/'pong-value-feasibility-20261028-r1/screen8-bank.json'
EXPECTED = '40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'
FILES = ('pong_option_memory.py', 'run_rally_option_memory.py', 'pong_continuous_visual.py',
    'pong_fixed_rbf.py', 'pong_fixed_pairwise.py', 'pong_action_credit.py',
    'rally_raw_capture.py', 'return_metrics.py', 'run_rally_raw_optimizer.py',
    'run_rally_raw_fresh_screen.py', 'run_rally_repertoire_collection.py', 'run_practical_pong_v2.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value): base.atomic_json(Path(path), value)


def seed_rngs():
    random.seed(BOOKKEEPING); np.random.seed(BOOKKEEPING); torch.manual_seed(BOOKKEEPING)


def prepare():
    assert not DEST.exists() and sha(MODEL/'rbf-model.npz') == EXPECTED
    bank = read(BANK); used = set(); sites = []
    for kind in ('later_miss', 'later_success'):
        for phase in ('left', 'right'):
            found = 0
            for entry in bank['bank']:
                if entry['seed'] in used or base.leftward(entry['seed']) != (phase == 'left'):
                    continue
                r = read(entry['receipt'])
                assert r['complete'] and r['weights_sha256'] == EXPECTED
                hits = [i for i, row in enumerate(r['records']) if row['hits']]
                candidates = [(h, hits[j-1]) for j, h in enumerate(hits) if j]
                if kind == 'later_miss':
                    final = r['records'][-1]
                    candidates = [(len(r['records'])-1, hits[-1])] if hits and final['terminated'] and final['reward'] < 0 else []
                candidates = [(outcome, prev) for outcome, prev in candidates if outcome-64 > prev]
                if not candidates:
                    continue
                outcome, previous = candidates[0]
                sites.append(dict(index=len(sites), seed=entry['seed'], phase=phase, kind=kind,
                    anchor=outcome-64, outcome=outcome, previous_hit=previous,
                    receipt=entry['receipt'], receipt_sha256=sha(entry['receipt'])))
                used.add(entry['seed']); found += 1
                if found == 2:
                    break
            if found != 2:
                raise RuntimeError(f'only {found}/2 distinct sites available for {kind}/{phase}; no substitute design')
    paths = [CALIBRATION/f'calibration_{i:02d}-scenes.npz' for i in range(8)]
    episodes = [load_scenes(p) for p in paths]
    assert list(map(len, episodes)) == [256]*8
    pairs = []; distances = []; rng = np.random.default_rng(2026102891)
    for i, scenes in enumerate(episodes):
        for t in sorted(rng.choice(len(scenes)-16, 32, replace=False).tolist()):
            distance = scene_distance(scenes[t], scenes[t+16])
            pairs.append(dict(episode=i, first=t, second=t+16, distance=distance)); distances.append(distance)
    radius = float(np.median(distances)); assert radius > 0 and np.isfinite(radius)
    first = [e['seed'] for e in read(SCREEN8)['bank']]
    screen = first + [e['seed'] for e in bank['bank'] if e['seed'] not in first][:24]
    DEST.mkdir(); (DEST/'acquisition').mkdir(); (DEST/'fresh').mkdir()
    schedule = select_starts(1943906127, 63137, exclusions(DEST), 32)
    seeds = schedule.pop('TRAIN'); schedule.update(probe=seeds[:32], unused_reserved=seeds[32:],
        purpose='unused paired full-game option-memory qualification; never fit')
    write(DEST/'fresh/schedules.json', schedule)
    pinned = {str(p):sha(p) for p in [BANK, SCREEN8, *paths,
        MODEL/'rbf-model.npz', MODEL/'proposal.npy', MODEL/'radial-cap.json']}
    pinned.update({s['receipt']:s['receipt_sha256'] for s in sites})
    spec = dict(method='anonymous_assignment_local_mean_paired_option_advantages', sites=sites,
        schema='canonical_scene_exact_six_fields_no_age_no_context',
        metric='mean minimum assignment squared6field cost/4; unmatched1; positionclip1; v/(1+abs(v)); IDsdiscarded',
        metric_amendment='use exact existing canonical6fields only; omit camera context to avoid reconstructing unavailable structured calibration context',
        radius=radius, radius_calibration_pairs=pairs, radius_source='existing2048scenes;256fixed16apart pairs;no new cameras',
        hold_cameras=16, baseline_reconsideration_cameras=1, suffix_cap=160,
        outcomes='first own hit .95**elapsed_seconds; natural termination0; timeout/horizon/deadline censored',
        learning='all completed paired option-minus-baseline outcomes; uniform mean over scenes within radius; no max/winner-only',
        acquisition_seconds=120, acquisition_transition_cap=30000, per_site_transition_cap=3750,
        evaluation_seconds=180, workers=4, screen8=first, screen32=screen, fresh_tag=63137,
        source=base.source_signature(), files={n:sha(Path(__file__).with_name(n)) for n in FILES},
        pinned_assets=pinned, no_backprop=True, no_privileged_policy_inputs=True,
        no_expert_labels=True, no_runtime_simulator_lookahead=True, cloned_TRAIN_state_scaffold=True,
        raw_only_episode_certificates_disabled=True, stateful_candidate_evaluated_from_reset=True,
        no_fresh_fitting=True, no_sweep=True, no_automatic_promotion=True,
        implementation_seconds_limit=900, parent_native_review_required=True)
    write(DEST/'spec.json', spec)
    print(json.dumps(dict(status='prepared_no_native', sites=sites, radius=radius,
        spec_sha256=sha(DEST/'spec.json'))), flush=True)


def acquisition_task(site, deadline, spec):
    began = time.monotonic(); seed_rngs(); prefix = []; outputs = []
    original = load_readout(MODEL/'rbf-model.npz')
    receipt = read(site['receipt']); session = FreshRBFEvaluator(original, PongEvaluationConfig(**receipt['config']))
    if time.monotonic() >= deadline:
        return dict(site=site, status='not_started_deadline', transitions=0, initials=0, outcomes=[])
    session._begin_episode()
    while session.transitions < site['anchor'] and time.monotonic() < deadline:
        row = session.step(check_frozen=False); prefix.append(row)
        assert row == receipt['records'][len(prefix)-1], ('prefix mismatch', site['seed'], len(prefix)-1)
    session.verify_frozen()
    paid = len(prefix); initials = session.initial_cameras
    if paid != site['anchor']:
        result = dict(site=site, status='prefix_deadline_partial', transitions=paid, initials=initials,
            prefix_records=prefix, outcomes=[])
        write(DEST/'acquisition'/f"site_{site['index']:02d}.json", result); return result
    scene = canonical_scene(session.latest_encoded)
    save_scenes(DEST/'acquisition'/f"site_{site['index']:02d}-scene.npz", [scene])
    for option in (3, 0, 1, 2):
        if time.monotonic() >= deadline or paid >= spec['per_site_transition_cap']:
            break
        branch = clone_session(session); records = []; elapsed = 0.; reason = 'horizon_censored'
        for t in range(spec['suffix_cap']):
            if time.monotonic() >= deadline:
                reason = 'deadline_censored'; break
            if paid >= spec['per_site_transition_cap']:
                reason = 'camera_quota_censored'; break
            row = step_action(branch, option) if option != 3 and t < HOLD else branch.step(check_frozen=False)
            records.append(row); paid += 1; elapsed += row['elapsed_seconds']
            if row['hits']:
                reason = 'own_hit'; break
            if row['terminated']:
                reason = 'natural_termination'; break
            if row['truncated']:
                reason = 'environment_timeout_censored'; break
        branch.verify_frozen()
        observed = reason in ('own_hit', 'natural_termination')
        value = float(.95**elapsed) if reason == 'own_hit' else 0. if observed else None
        if option == 3:
            assert records == receipt['records'][site['anchor']:site['anchor']+len(records)]
        outputs.append(dict(option=option, observed=observed, reason=reason, value=value,
            elapsed_seconds=elapsed, transitions=len(records), records=records,
            tail_policy_sha256=EXPECTED, clone_initial_cameras=branch.initial_cameras-initials))
        write(DEST/'acquisition'/f"site_{site['index']:02d}-progress.json",
            dict(site=site, transitions=paid, initials=initials, outcomes=outputs))
    result = dict(site=site, status='complete' if len(outputs)==4 else 'bounded_partial',
        transitions=paid, initials=initials, prefix_transitions=len(prefix), prefix_records=prefix,
        outcomes=outputs, scene_file=str(DEST/'acquisition'/f"site_{site['index']:02d}-scene.npz"),
        exact_prefix_native_replay=True, clone_physics_never_policy_input=True,
        wall_seconds=time.monotonic()-began)
    assert paid <= spec['per_site_transition_cap'] and all(o['clone_initial_cameras']==0 for o in outputs)
    write(DEST/'acquisition'/f"site_{site['index']:02d}.json", result); return result


def parallel_jobs(worker, jobs, deadline, workers=4, progress=None):
    results = []; errors = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        pending = iter(jobs); active = {}
        def fill():
            while len(active)<workers and time.monotonic()<deadline-2:
                try: job = next(pending)
                except StopIteration: break
                active[pool.submit(worker, *job)] = job
        fill()
        while active:
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try: results.append(future.result())
                except Exception as error: errors.append(dict(job=str(job), error=repr(error)))
            if progress: write(progress, dict(completed=len(results), errors=errors,
                transitions=sum(r['transitions'] for r in results)))
            fill()
    if errors:
        raise RuntimeError(f'worker errors, inspect progress and per-site receipts: {errors}')
    return results


def validate_spec(approval):
    assert approval == sha(DEST/'spec.json')
    spec = read(DEST/'spec.json')
    assert spec['source']==base.source_signature()
    assert all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items())
    assert all(sha(p)==h for p,h in spec['pinned_assets'].items())
    return spec


def acquire(approval):
    spec = validate_spec(approval); assert not (DEST/'acquisition-clock.json').exists()
    began = time.monotonic(); deadline = began+120
    write(DEST/'acquisition-clock.json', dict(started_monotonic=began, deadline_monotonic=deadline,
        started_wall_ns=time.time_ns(), seconds=120, transition_cap=30000))
    results = parallel_jobs(acquisition_task, [(s, deadline-1, spec) for s in spec['sites']],
        deadline, progress=DEST/'acquisition-progress.json')
    results.sort(key=lambda r:r['site']['index'])
    scenes=[]; advantages=[]; valid=[]; provenance=[]; history=[]
    for result in results:
        if not result.get('scene_file'):
            continue
        scenes.append(load_scenes(result['scene_file'])[0]); row=np.zeros(3); mask=np.zeros(3,dtype=bool)
        options={o['option']:o for o in result['outcomes']}; baseline=options.get(3)
        for action in range(3):
            other=options.get(action)
            if baseline and other and baseline['observed'] and other['observed']:
                row[action]=other['value']-baseline['value']; mask[action]=True
                history.append(dict(site=result['site']['index'], action=action,
                    paired_advantage=float(row[action]), acquired_from_own_reward=True))
        advantages.append(row); valid.append(mask); provenance.append(result['site'])
    memory=OptionMemory(scenes,np.asarray(advantages).reshape(-1,3),np.asarray(valid).reshape(-1,3),spec['radius'])
    save_memory(DEST/'option-memory.npz',memory)
    write(DEST/'memory-provenance.json',dict(sites=provenance, online_acquisition_updates=history,
        all_completed_positive_negative_and_zero_pairs_retained=True,
        no_state_identifiers_used_by_policy=True,tail_policy=EXPECTED))
    for name in ('proposal.npy','radial-cap.json','rbf-model.npz'):
        (DEST/name).write_bytes((MODEL/name).read_bytes())
    summary=dict(status='complete' if len(results)==8 and all(r['status']=='complete' for r in results) else 'bounded_partial',
        sites=len(scenes), completed_pairs=int(memory.valid.sum()),
        positive_pairs=int(np.sum(memory.valid & (memory.advantages>0))),
        negative_pairs=int(np.sum(memory.valid & (memory.advantages<0))),
        zero_pairs=int(np.sum(memory.valid & (memory.advantages==0))),
        advantages=memory.advantages.tolist(), valid=memory.valid.tolist(),
        transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initials'] for r in results),
        prefix_cameras=sum(r.get('prefix_transitions',r['transitions']) for r in results),
        wall_seconds=time.monotonic()-began,memory_sha256=sha(DEST/'option-memory.npz'),
        no_native_deployment_yet=True,raw_only_certificates_disabled=True,no_automatic_promotion=True)
    assert summary['transition_cameras']<=30000
    write(DEST/'acquisition-summary.json',summary);print(json.dumps(summary),flush=True)


def native_task(directory, seed, index, deadline, candidate=True):
    directory=Path(directory);began=time.monotonic();seed_rngs()
    original=load_readout(MODEL/'rbf-model.npz')
    config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,
        policy_scaling=read(MODEL/'radial-cap.json'))
    memory=load_memory(DEST/'option-memory.npz') if candidate else None
    session=OptionMemoryEvaluator(original,memory,config) if candidate else FreshRBFEvaluator(original,config)
    game=session.env.game;original_step=game.step;counter=ReturnCounter();records=[];events=[];raw=SparseRows();scenes=[]
    def instrument(drive,**kwargs):
        own=counter.opportunities;opponent=counter.opponent_opportunities;tick=int(game.rally_steps[0])+1
        out=counter.step(game,original_step,drive,**kwargs)
        if counter.opportunities>own:events.append(dict(kind='own',ordinal=counter.opportunities,success=bool(out.hits[0]),tick=tick,transition=session.transitions,completed_own_hits=counter.returns))
        if counter.opponent_opportunities>opponent:events.append(dict(kind='opponent',ordinal=counter.opponent_opportunities,success=float(game.ball_velocity[0,0])<0,tick=tick,transition=session.transitions,after_own_hits=counter.returns))
        return out
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            if session.initial_cameras==session.episodes:session._begin_episode()
            raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    finally: del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,truncated=last['truncated'],terminated=last['terminated'],reward=last['reward'],seconds=last['seconds'])
    count_status=native_count_status(annotation,report)
    stem=f"{'candidate' if candidate else 'baseline'}_{index:02d}_{seed}"
    raw_path=directory/(stem+'-raw.npz');scene_path=directory/(stem+'-scenes.npz')
    sp.save_npz(raw_path,raw.matrix());save_scenes(scene_path,scenes)
    choices=session.option_choices if candidate else []
    audit=dict(passed=True,rows=len(records),stateful_replay=True,new_native_cameras=0)
    if candidate:
        controller=MemoryController(memory);X=raw.matrix();changes=starts=holds=fallback=0
        for i,(scene,record,choice) in enumerate(zip(scenes,records,choices)):
            lo,hi=X.indptr[i:i+2];baseline_action,q=original.choose(X.indices[lo:hi],X.data[lo:hi])
            action,decision=controller.choose(scene,baseline_action)
            assert decision==choice and action==record['action'] and np.array_equal(q,np.asarray(record['q']))
            changes+=int(action!=baseline_action);starts+=int(choice['mode']=='earned_start')
            holds+=int(choice['mode']=='earned_hold');fallback+=int(choice['mode']=='baseline_camera')
        assert len(choices)==len(records)
        audit.update(changed_actions=changes,earned_option_starts=starts,earned_held_cameras=holds,
            baseline_camera_decisions=fallback,raw_only_certificate_reuse=False)
    result=dict(seed=seed,complete=report['complete'],config=asdict(config),report=report,annotation=annotation,
        annotation_count_status=count_status,records=records,opportunity_events=events,option_choices=choices,
        feature_file=str(raw_path),structured_feature_file=str(scene_path),transitions=session.transitions,
        initial_cameras=session.initial_cameras,raw_only_episode_reuse=False,full_stateful_audit=audit,
        baseline_model_sha256=EXPECTED,memory_sha256=sha(DEST/'option-memory.npz') if candidate else None,
        wall_seconds=time.monotonic()-began)
    write(directory/(stem+'.json'),result)
    if report['complete']: assert count_status['valid_terminal_counts']
    return result


def comparison(baseline,candidate):
    return dict(baseline=aggregate([r['annotation'] for r in baseline]),candidate=aggregate([r['annotation'] for r in candidate]),
        baseline_more=more(baseline),candidate_more=more(candidate),
        native_cameras=sum(r['transitions'] for r in candidate),initial_cameras=sum(r['initial_cameras'] for r in candidate),
        changed_actions=sum(r['full_stateful_audit'].get('changed_actions',0) for r in candidate),
        earned_option_starts=sum(r['full_stateful_audit'].get('earned_option_starts',0) for r in candidate),
        earned_held_cameras=sum(r['full_stateful_audit'].get('earned_held_cameras',0) for r in candidate))


def evaluate(approval):
    spec=validate_spec(approval);acq=read(DEST/'acquisition-summary.json')
    assert not (DEST/'evaluation-clock.json').exists() and acq['memory_sha256']==sha(DEST/'option-memory.npz')
    began=time.monotonic();deadline=began+180
    write(DEST/'evaluation-clock.json',dict(started_monotonic=began,deadline_monotonic=deadline,started_wall_ns=time.time_ns(),seconds=180))
    bank={e['seed']:e for e in read(BANK)['bank']};results=[];summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0)
    if acq['positive_pairs']==0:
        summary['status']='acquisition_failure_no_positive_pairs'
    else:
        for size in (8,32):
            directory=DEST/f'screen{size}';directory.mkdir()
            target=spec['screen8' if size==8 else 'screen32'];existing={r['seed']:r for r in results}
            jobs=[(str(directory),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in existing]
            new=parallel_jobs(native_task,jobs,deadline,progress=directory/'progress.json');results+=new
            summary['new_transition_cameras']+=sum(r['transitions'] for r in new)
            summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new)
            byseed={r['seed']:r for r in results};complete=all(s in byseed and byseed[s]['complete'] for s in target)
            selected=[byseed[s] for s in target if s in byseed];baseline=[read(bank[r['seed']]['receipt']) for r in selected]
            stats=comparison(baseline,selected);stats['complete']=complete;write(directory/'summary.json',stats);summary[f'screen{size}']=stats
            if not complete: summary['status']=f'screen{size}_bounded_partial';break
            before=(stats['baseline']['own_returns'],stats['baseline_more']['later_returns'],stats['baseline']['first_incoming_returns'],stats['baseline_more']['tail']['2'])
            after=(stats['candidate']['own_returns'],stats['candidate_more']['later_returns'],stats['candidate']['first_incoming_returns'],stats['candidate_more']['tail']['2'])
            if any(a<b for a,b in zip(after,before)) or after[0]<=before[0]:
                summary['status']=f'rejected_screen{size}';break
            if size==32:
                gate=after[1]>before[1] and stats['candidate_more']['later_fraction']>=stats['baseline_more']['later_fraction']
                if not gate: summary['status']='rejected_screen32';break
                fresh=DEST/'fresh';seeds=read(fresh/'schedules.json')['probe']
                jobs=[(str(fresh),seed,i,deadline-1,candidate) for i,seed in enumerate(seeds) for candidate in (False,True)]
                fresh_results=parallel_jobs(native_task,jobs,deadline,progress=fresh/'progress.json')
                summary['new_transition_cameras']+=sum(r['transitions'] for r in fresh_results)
                summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in fresh_results)
                groups={c:{r['seed']:r for r in fresh_results if (r['memory_sha256'] is not None)==c} for c in (False,True)}
                complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values())
                fs=dict(complete=complete,groups={str(c):aggregate([r['annotation'] for r in g.values()]) for c,g in groups.items()})
                if complete:
                    a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds]
                    fs.update(comparison(a,b),paired=paired_metrics(a,b))
                write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_bounded_partial')
    summary.update(wall_seconds=time.monotonic()-began,memory_frozen=sha(DEST/'option-memory.npz')==acq['memory_sha256'],
        source_frozen=base.source_signature()==spec['source'] and all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),
        assets_unchanged=all(sha(p)==h for p,h in spec['pinned_assets'].items()),raw_only_episode_certificates_disabled=True,no_automatic_promotion=True)
    write(DEST/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','acquire','evaluate'])
    parser.add_argument('--approved-spec-sha256');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    elif args.operation=='acquire':acquire(args.approved_spec_sha256)
    else:evaluate(args.approved_spec_sha256)
