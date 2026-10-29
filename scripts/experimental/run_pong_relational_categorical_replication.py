"""Two prospectively fixed categorical acquisitions over frozen visual feedback.

Preparation acquires no observations. Each acquisition starts from literal40069,
uses actual own-hit reward only, and keeps all fresh outcomes out of selection.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from pong_relational_feedback import RelationalPolicy, structural_choices, load_calibration_scales
from pong_relational_persistent_binding import PersistentRelationalPolicy
from run_rally_option_memory import ROOT, MODEL, EXPECTED, read, write, sha
from run_rally_repertoire_collection import select_starts, exclusions
from stateful_reward_fitness import CameraBudget
import run_practical_pong_v2 as base

DEST = ROOT / 'pong-relational-categorical-replication-20261029-r1'
CALIBRATION = ROOT / 'pong-rally-continuous-actor-20261028-r1/calibration-report.json'
BASELINE = 'literal40069'
STAGES = (4, 8, 16)
SURVIVORS = (4, 2, None)
ROOT_SEED = 1943906127
TRAIN_TAGS = (63401, 63402)
FRESH_TAGS = (63417, 63418)
ORDER_SEEDS = (2026102911, 2026102912)
WORKER_MODULE = 'compact_public_pong'


def canonical_seed(seed):
    return int(seed) % base.MODULUS


def make_roster(scales):
    """All sixteen predeclared structures; no privileged prior winner slot."""
    return [dict(id=f'structure_{index:02d}', choices=list(choice),
        coefficients=[0.] * 5, scales=np.asarray(scales, float).tolist(), require_first_hit=False)
        for index, choice in enumerate(c for c in structural_choices() if c[0] in (2, 3))]


def draw_priority(roster, seed):
    control_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    control = roster[int(control_rng.integers(len(roster)))]['id']
    priority = [roster[int(i)]['id'] for i in order_rng.permutation(len(roster))]
    return dict(initial_control=control, priority=priority, seed=seed,
        control_rng_state=control_rng.bit_generator.state, order_rng_state=order_rng.bit_generator.state)


def duration_assignment(seeds):
    if len(seeds) != 64:
        raise ValueError('sixty-four prospectively assigned fresh starts required')
    return [dict(index=i, seed=int(seed), seconds=60 if i < 32 else 20,
        max_episode_ticks=7200 if i < 32 else 2400,
        max_transition_cameras=3000 if i < 32 else 1000,
        stratum='long60' if i < 32 else 'standard20') for i, seed in enumerate(seeds)]


def fixed_schedules(excluded):
    used = set(excluded)
    result = {}
    for replica, tag in enumerate(TRAIN_TAGS):
        selected = select_starts(ROOT_SEED, tag, used, 8)
        seeds = selected.pop('TRAIN')
        result[str(replica)] = dict(train=dict(selected, seeds=seeds))
        used.update(map(canonical_seed, seeds))
    for replica, tag in enumerate(FRESH_TAGS):
        selected = select_starts(ROOT_SEED, tag, used, 32)
        seeds = selected.pop('TRAIN')
        result[str(replica)]['fresh'] = dict(selected,
            purpose='candidate-only frozen endpoint reliability; never acquisition',
            assignments=duration_assignment(seeds))
        used.update(map(canonical_seed, seeds))
    validate_schedules(result, excluded)
    return result


def validate_schedules(schedules, excluded):
    all_seeds = []
    for replica in ('0', '1'):
        train = schedules[replica]['train']['seeds']
        fresh = schedules[replica]['fresh']['assignments']
        if len(train) != 16 or len(fresh) != 64:
            raise ValueError('both complete TRAIN and fresh schedules must be fixed')
        for size in STAGES:
            if sum(base.leftward(s) for s in train[:size]) != size // 2:
                raise ValueError('every nested TRAIN stage must be phase balanced')
        for name, seconds in (('long60', 60), ('standard20', 20)):
            rows = [r for r in fresh if r['stratum'] == name]
            if len(rows) != 32 or any(r['seconds'] != seconds for r in rows):
                raise ValueError('fixed duration cohorts required')
            if sum(base.leftward(r['seed']) for r in rows) != 16:
                raise ValueError('each fresh duration cohort must be balanced')
            if any(r['max_episode_ticks'] != seconds * 120
                    or r['max_transition_cameras'] != seconds * 50 for r in rows):
                raise ValueError('fresh duration and camera grants must agree exactly')
        if [r['index'] for r in fresh] != list(range(64)):
            raise ValueError('fresh schedule ordinal mismatch')
        all_seeds += train + [r['seed'] for r in fresh]
    canonical = list(map(canonical_seed, all_seeds))
    if len(set(canonical)) != 160 or set(canonical) & set(excluded):
        raise ValueError('replica TRAIN/fresh starts must be entirely new and disjoint')


def cache_key(policy, seed):
    return f'{policy}:{int(seed)}'


def completed_score(policy, seeds, cache):
    rows = [cache.get(cache_key(policy, seed)) for seed in seeds]
    if any(row is None or not row.get('fitness_eligible') for row in rows):
        return None
    return sum(int(row['own_hits']) for row in rows)


def rank_complete(population, seeds, cache, fixed_priority):
    scored = [(p, completed_score(p, seeds, cache)) for p in population]
    priority = {policy: index for index, policy in enumerate(fixed_priority)}
    return sorted(((p, score) for p, score in scored if score is not None),
        key=lambda row: (-row[1], priority[row[0]]))


def select_endpoint(priority, seeds, cache):
    """All completed16 structures may contend, including the initial control."""
    baseline = completed_score(BASELINE, seeds, cache)
    if baseline is None:
        raise ValueError('complete matched baseline required')
    selected, score = BASELINE, baseline
    for candidate in priority:
        value = completed_score(candidate, seeds, cache)
        if value is not None and value > score:
            selected, score = candidate, value
    return selected, score


def prepare():
    if DEST.exists():
        raise FileExistsError('preserve all prepared schedules and acquisition attempts')
    if sha(MODEL / 'rbf-model.npz') != EXPECTED:
        raise ValueError('literal40069 baseline mismatch')
    worker = importlib.import_module(WORKER_MODULE)
    scales = load_calibration_scales(CALIBRATION)
    DEST.mkdir()
    excluded = exclusions(DEST)
    schedules = fixed_schedules(excluded)
    write(DEST / 'schedules.json', schedules)
    (DEST / 'models').mkdir()
    roster = make_roster(scales)
    for entry in roster:
        path = DEST / 'models' / (entry['id'] + '.npz')
        original = RelationalPolicy(*entry['choices'], entry['coefficients'], entry['scales'])
        PersistentRelationalPolicy(original, require_first_hit=False).save(path)
        entry.update(model_path=str(path), model_sha256=sha(path))
    replicas = {}
    for replica, seed in enumerate(ORDER_SEEDS):
        folder = DEST / f'replica{replica}'
        folder.mkdir(); (folder / 'fitness').mkdir(); (folder / 'failures').mkdir()
        replicas[str(replica)] = dict(draw_priority(roster, seed),
            folder=str(folder), train_seeds=schedules[str(replica)]['train']['seeds'])
    names = set(worker.FILES) | {Path(__file__).name, WORKER_MODULE + '.py',
        'pong_relational_feedback.py', 'pong_relational_persistent_binding.py',
        'run_rally_repertoire_collection.py', 'stateful_reward_fitness.py'}
    files = [Path(__file__).with_name(name) for name in sorted(names)]
    assets = [MODEL / 'rbf-model.npz', MODEL / 'radial-cap.json', CALIBRATION,
        DEST / 'schedules.json', DEST / 'seed-exclusion-proof.json',
        *[Path(entry['model_path']) for entry in roster]]
    spec = dict(schema='independent-categorical-visual-feedback-acquisition-v1',
        source=base.source_signature(), files={str(p): sha(p) for p in files},
        assets={str(p): sha(p) for p in assets}, baseline_sha256=EXPECTED,
        scales=scales.tolist(), roster=roster, replicas=replicas,
        schedules_path=str(DEST / 'schedules.json'), schedules_sha256=sha(DEST / 'schedules.json'),
        acquisition_seconds_per_replica=240, transition_cap_per_replica=128000,
        acquisition_max_episode_ticks=2400, game_camera_grant=1000,
        initial_cameras_counted_separately=True, maximum_acquisition_games_per_replica=124,
        workers_per_replica=4, replicas_run_sequentially=True,
        stages=list(STAGES), survivors=list(SURVIVORS),
        numeric_coefficients='all five exactly zero; no optimization',
        policy='ungated persistent public visual-ID pair feedback; original baseline on unsupported inputs',
        fitness='complete actual-game scalar own-hit count only; no opportunity labels used for selection',
        initialization='literal40069 incumbent; independent uniform predrawn structural initial control completed on16',
        selection='same nested4→8→16;16→4→2;all complete16 candidates including initial control contend;baseline wins ties;fixed proposal priority resolves candidate ties',
        incomplete='retain and charge every started game; no retry; incomplete declared acquisition yields no learned endpoint',
        cache='per-replica exact candidate file/runtime identity/seed/duration; no prior evaluation or cross-replica reuse',
        dispatch='baseline16,initial-control16,then nested stages using exact cached games',
        fresh='candidate-only32long60s+32standard20s per replica;assignments fixed before acquisition;both endpoints frozen before either validation',
        fresh_worst_case_transition_cameras=256000, fresh_initial_cameras=128,
        reliability='report each replica and duration stratum;approximately95%first/later;>=200later opportunities in each long32;report0–20/20–40/40–60s exposure;no outcome-dependent extension',
        pilot63345_is_separate=True, no_automatic_validation=True,
        inherited='shared frozen trained vision, visual association memory, literal40069 baseline and unlabeled calibration scales;categorical controller relearned,not full system from scratch',
        no_backprop=True, no_privileged_policy_inputs=True, no_expert_action_labels=True,
        no_old_winner_preference=True, no_numeric_sweep=True, no_automatic_extension=True)
    spec['identity'] = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(DEST / 'spec.json', spec)
    result = dict(status='prepared_no_native', spec_sha256=sha(DEST / 'spec.json'),
        schedules_sha256=sha(DEST / 'schedules.json'),
        initial_controls={r: v['initial_control'] for r, v in replicas.items()},
        cohort_sizes=dict(TRAIN=32, fresh_long=64, fresh_standard=64))
    print(result, flush=True)
    return result


def verify_spec(spec, approved):
    if approved != sha(DEST / 'spec.json'):
        raise ValueError('approved specification mismatch')
    if spec['source'] != base.source_signature():
        raise ValueError('runtime source changed')
    if not all(sha(p) == h for p, h in {**spec['files'], **spec['assets']}.items()):
        raise ValueError('pinned implementation or assets changed')
    schedules = read(spec['schedules_path'])
    excluded = read(DEST / 'seed-exclusion-proof.json')['canonical_ids']
    validate_schedules(schedules, excluded)


def dispatch(pool, worker, policies, seeds, cache, budget, spec, replica, deadline, label):
    folder = Path(spec['replicas'][str(replica)]['folder'])
    models = {entry['id']: entry for entry in spec['roster']}
    pending = [(policy, seed) for policy in policies for seed in seeds
        if cache_key(policy, seed) not in cache]
    active = {}; results = []

    def fill():
        while pending and len(active) < spec['workers_per_replica'] and time.monotonic() < deadline - 2:
            grant = spec['game_camera_grant']
            if not budget.reserve(grant):
                break
            policy, seed = pending.pop(0)
            path = None if policy == BASELINE else models[policy]['model_path']
            native_key = worker.fitness_key(path, seed, spec['acquisition_max_episode_ticks'], spec['identity'])
            try:
                future = pool.submit(worker.evaluate, path, seed,
                    spec['acquisition_max_episode_ticks'], str(folder / 'fitness'),
                    deadline - 1, grant, spec['identity'])
            except Exception as error:
                future = Future()
                future.set_exception(error)
            active[future] = dict(policy=policy, seed=int(seed), native_key=native_key,
                model_sha256=None if path is None else models[policy]['model_sha256'])

    fill()
    while active:
        done, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
        for future in done:
            job = active.pop(future)
            try:
                result = future.result()
                if (result['seed'] != job['seed'] or result['model_sha256'] != job['model_sha256']
                        or result['runtime_identity'] != spec['identity']
                        or result['max_episode_ticks'] != spec['acquisition_max_episode_ticks']
                        or result['fitness_key'] != job['native_key']):
                    raise ValueError('worker receipt identity mismatch')
                if (result['transitions'] > spec['game_camera_grant']
                        or result['transition_camera_upper_bound'] > spec['game_camera_grant']
                        or result['initial_camera_upper_bound'] > 1):
                    raise ValueError('worker exceeded declared grant')
                if result['fitness_eligible'] and not result['complete']:
                    raise ValueError('incomplete game cannot contribute fitness')
            except Exception as error:
                result = dict(seed=job['seed'], complete=False, fitness_eligible=False,
                    transitions=0, initial_cameras=0,
                    transition_camera_upper_bound=spec['game_camera_grant'],
                    initial_camera_upper_bound=1, error=repr(error),
                    ambiguous_or_failed_grant_charged=True)
                write(folder / 'failures' / (job['native_key'] + '.json'), dict(job=job, result=result))
            budget.finish(spec['game_camera_grant'], result)
            results.append(dict(policy=job['policy'], **result))
            if result['fitness_eligible']:
                cache[cache_key(job['policy'], job['seed'])] = result
        write(folder / 'progress.json', dict(phase=label, cost=budget.state(),
            completed_jobs=len(results), pending_jobs=len(pending), active=list(active.values())))
        fill()
    # Each stage is attempted once. Failed/unfinished jobs never become cache hits.
    return results


def run(replica, approved):
    spec = read(DEST / 'spec.json'); verify_spec(spec, approved)
    folder = Path(spec['replicas'][str(replica)]['folder'])
    if (folder / 'work-clock.json').exists() or (DEST / 'endpoints-frozen.json').exists():
        raise FileExistsError('preserve the original acquisition clock and endpoints')
    if replica == 1 and not (DEST / 'replica0/summary.json').exists():
        raise ValueError('replicas must run sequentially, with the first pool closed')
    if replica == 0 and (DEST / 'replica1/work-clock.json').exists():
        raise ValueError('replica1 already started')
    worker = importlib.import_module(WORKER_MODULE)
    began = time.monotonic(); deadline = began + spec['acquisition_seconds_per_replica']
    write(folder / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        started_monotonic=began, deadline_monotonic=deadline,
        seconds=spec['acquisition_seconds_per_replica'], spec_sha256=approved))
    budget = CameraBudget(spec['transition_cap_per_replica'])
    config = spec['replicas'][str(replica)]; seeds = config['train_seeds']
    cache = {}; stages = []; receipts = []
    protocol_complete = True
    with ProcessPoolExecutor(max_workers=spec['workers_per_replica'],
            mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        for policies, label in (([BASELINE], 'baseline16'),
                                ([config['initial_control']], 'initial-control16')):
            receipts += dispatch(pool, worker, policies, seeds, cache, budget, spec, replica, deadline, label)
            if any(completed_score(policy, seeds, cache) is None for policy in policies):
                protocol_complete = False
                break
        population = config['priority']
        if protocol_complete:
            for size, survivors in zip(spec['stages'], spec['survivors']):
                receipts += dispatch(pool, worker, population, seeds[:size], cache,
                    budget, spec, replica, deadline, f'stage{size}')
                ranked = rank_complete(population, seeds[:size], cache, config['priority'])
                stage = dict(size=size, expected=len(population), completed=len(ranked),
                    baseline_own_hits=completed_score(BASELINE, seeds[:size], cache),
                    scored=[dict(policy=policy, own_hits=score) for policy, score in ranked])
                stages.append(stage); write(folder / f'stage{size}.json', stage)
                print(dict(replica=replica, **stage, cost=budget.state(),
                    elapsed_seconds=time.monotonic() - began), flush=True)
                if len(ranked) != len(population):
                    protocol_complete = False
                    break
                if survivors is not None:
                    population = [policy for policy, _ in ranked[:survivors]]
    selected, selected_score = (select_endpoint(config['priority'], seeds, cache)
        if protocol_complete else (BASELINE, completed_score(BASELINE, seeds, cache)))
    models = {entry['id']: entry for entry in spec['roster']}
    endpoint = dict(replica=replica, selected=selected,
        model_path=None if selected == BASELINE else models[selected]['model_path'],
        model_sha256=None if selected == BASELINE else models[selected]['model_sha256'],
        own_hits=selected_score, baseline_own_hits=completed_score(BASELINE, seeds, cache),
        initial_control=config['initial_control'],
        initial_control_own_hits=completed_score(config['initial_control'], seeds, cache),
        complete16_candidates=[dict(policy=policy, own_hits=completed_score(policy, seeds, cache))
            for policy in config['priority'] if completed_score(policy, seeds, cache) is not None],
        acquisition_complete=protocol_complete, learned_endpoint=selected != BASELINE,
        spec_sha256=approved, frozen_before_any_final_fresh=True)
    write(folder / 'endpoint.json', endpoint)
    verify_spec(spec, approved)
    summary = dict(status='acquisition_complete' if protocol_complete else 'incomplete_no_learned_endpoint',
        endpoint=endpoint, stages=stages, cost=budget.state(), jobs=receipts,
        cache={key: row['receipt_file'] for key, row in cache.items()},
        wall_seconds=time.monotonic() - began,
        post_deadline_cleanup_seconds=max(0., time.monotonic() - deadline),
        source_frozen=True, assets_frozen=True, spec_sha256=approved,
        no_fresh_evaluation=True, no_numeric_optimization=True,
        inherited_frozen_vision_and_baseline=True)
    write(folder / 'summary.json', summary)
    print(dict(replica=replica, status=summary['status'], endpoint=endpoint,
        cost=summary['cost'], wall_seconds=summary['wall_seconds']), flush=True)
    return summary


def freeze_endpoints(approved):
    """Explicit barrier for a separate candidate-only validation coordinator."""
    spec = read(DEST / 'spec.json'); verify_spec(spec, approved)
    path = DEST / 'endpoints-frozen.json'
    if path.exists():
        raise FileExistsError('both frozen endpoints are immutable')
    endpoints = {}
    for replica in ('0', '1'):
        folder = DEST / f'replica{replica}'
        summary = read(folder / 'summary.json'); endpoint = read(folder / 'endpoint.json')
        if (summary['status'] != 'acquisition_complete' or not endpoint['acquisition_complete']
                or summary['spec_sha256'] != approved or summary['endpoint'] != endpoint):
            raise ValueError('both declared acquisitions must complete before validation')
        if endpoint['model_path'] is not None and sha(endpoint['model_path']) != endpoint['model_sha256']:
            raise ValueError('selected endpoint changed')
        endpoints[replica] = dict(endpoint,
            endpoint_receipt_sha256=sha(folder / 'endpoint.json'),
            acquisition_summary_sha256=sha(folder / 'summary.json'))
    result = dict(endpoints=endpoints, schedules_sha256=spec['schedules_sha256'],
        spec_sha256=approved, frozen_wall_ns=time.time_ns(),
        validation_started=False, candidate_only=True,
        no_automatic_validation=True, both_endpoints_frozen=True)
    write(path, result)
    print(dict(status='both_endpoints_frozen_no_validation', sha256=sha(path)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'freeze'])
    parser.add_argument('--replica', type=int, choices=[0, 1])
    parser.add_argument('--approved-spec-sha256')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        if args.replica is None or args.approved_spec_sha256 is None:
            parser.error('run requires replica and approved spec hash')
        run(args.replica, args.approved_spec_sha256)
    else:
        if args.approved_spec_sha256 is None:
            parser.error('freeze requires approved spec hash')
        freeze_endpoints(args.approved_spec_sha256)
