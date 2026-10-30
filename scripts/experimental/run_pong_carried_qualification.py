"""Fresh bounded carried-learning replication with matched tail-only controls.

Preparation is separate from execution. The live policy sees every actual game
in chronological order. Reset/frozen controls replay only prospectively matched
tail starts in their own streams and never supply feedback to a live learner.
The frozen control branches the actual first replica's acquired end-A state.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

from compact_public_pong import FILES as FROZEN_FILES
from pong_carried_structure_learner import CarriedStructureLearner, state_hash
from pong_fixed_rbf import load_readout
from pong_relational_feedback import load_calibration_scales
from return_metrics import aggregate
from run_pong_carried_learning import evaluate_game, verify_spec
from run_pong_relational_categorical_replication import make_roster, CALIBRATION
from run_rally_option_memory import MODEL, EXPECTED, read, write, sha, seed_rngs
from run_rally_repertoire_collection import exclusions, select_starts
import run_practical_pong_v2 as base

CONTRACT = 'fresh-carried-learning-replication-matched-tail-controls-v1'
PHASES = ('A_acquire', 'B_reverse', 'A_return')
TAIL = 32


def wall_budget(maximum_cameras):
    """Prospective worst-case/84 cameras/sec plus20%, rounded to full minutes."""
    return 60 * math.ceil(maximum_cameras / 84. * 1.2 / 60.)


def phase_metrics(rows, assignments, games_per_phase):
    result = {}
    for phase in PHASES:
        scheduled = [row for row in assignments if row['phase'] == phase]
        if not scheduled:
            continue
        completed = [row for row in rows if row['assignment']['phase'] == phase and row['complete']]
        tail = [row for row in completed
            if row['assignment']['phase_index'] >= games_per_phase - TAIL]
        metric = aggregate([row['annotation'] for row in tail])
        first_n, first_hits = metric['first_incoming_opportunities'], metric['first_incoming_returns']
        later_n = metric['own_opportunities'] - first_n
        later_hits = metric['own_returns'] - first_hits
        later_fraction = later_hits / later_n if later_n else None
        metric.update(later_opportunities=later_n, later_returns=later_hits,
            later_return_fraction=later_fraction, scheduled_tail_games=TAIL,
            completed_tail_games=len(tail), all_live_selection_modes_included=True)
        metric['passes'] = bool(len(tail) == TAIL and first_n > 0 and
            first_hits / first_n >= .95 and later_n >= 100 and later_fraction >= .95)
        result[phase] = dict(all=aggregate([row['annotation'] for row in completed]),
            last32=metric, chosen_arms=[row['arm'] for row in completed],
            own_hits_by_game=[row['report']['hits'] for row in completed],
            selection_counts={mode: sum(row['selection_mode'] == mode for row in completed)
                for mode in sorted({row['selection_mode'] for row in completed})})
    return result


def validate_schedules(schedules, streams, games_per_phase, excluded):
    used = set()
    for name, config in streams.items():
        assignments = schedules[name]
        if [row['index'] for row in assignments] != list(range(len(assignments))):
            raise ValueError('stream assignment indices must be sequential')
        if config['role'] != 'live':
            continue
        if len(assignments) != 3 * games_per_phase:
            raise ValueError('incomplete live phase schedule')
        for phase in PHASES:
            rows = [row for row in assignments if row['phase'] == phase]
            if [row['phase_index'] for row in rows] != list(range(games_per_phase)):
                raise ValueError('live phase ordinals differ')
            if sum(base.leftward(row['seed']) for row in rows) != games_per_phase // 2:
                raise ValueError('live serve phase imbalance')
        ids = {row['seed'] % base.MODULUS for row in assignments}
        if len(ids) != len(assignments) or ids & (used | set(excluded)):
            raise ValueError('live starts repeat or overlap historical exclusions')
        used.update(ids)
    for name, config in streams.items():
        if config['role'] == 'live':
            continue
        for row in schedules[name]:
            live = schedules[row['matched_live_stream']][row['matched_live_index']]
            if row['phase_index'] < games_per_phase - TAIL:
                raise ValueError('control prefix must not be replayed')
            if config['role'] == 'frozen' and row['phase'] == 'A_acquire':
                raise ValueError('frozen control must branch its actual acquired prefix')
            for field in ('seed', 'phase', 'phase_index', 'reverse',
                    'max_episode_ticks', 'max_transition_cameras'):
                if row[field] != live[field]:
                    raise ValueError('paired control differs from its live start')
    return len(used)


def frozen_prefix(destination, parent_name, spec, approved):
    """Verify the closed end-A artifacts before copying their acquired arm."""
    folder = Path(destination) / parent_name
    ready_path = folder / 'checkpoint-end-A-ready.json'
    ready = read(ready_path)
    checkpoint = folder / 'checkpoint-end-A.json'
    receipt_path = folder / f"game-{spec['games_per_phase'] - 1:03d}.json"
    if (ready['stream'] != parent_name or ready['runtime_identity'] != spec['identity']
            or ready['spec_sha256'] != approved or ready['checkpoint_sha256'] != sha(checkpoint)
            or ready['receipt_sha256'] != sha(receipt_path)):
        raise ValueError('frozen prefix ready barrier or source identity differs')
    acquired, receipt = read(checkpoint), read(receipt_path)
    if (acquired['games'] != spec['games_per_phase'] or acquired['active_arm'] is not None
            or not receipt['complete'] or receipt['runtime_identity'] != spec['identity']
            or receipt['assignment']['phase'] != 'A_acquire'
            or receipt['assignment']['phase_index'] != spec['games_per_phase'] - 1
            or receipt['learner_after_sha256'] != state_hash(acquired)):
        raise ValueError('frozen source must be the actual final acquired A game')
    learner = CarriedStructureLearner.from_state(acquired).frozen_branch()
    learner.incumbent = int(receipt['arm'])
    return learner, dict(parent_checkpoint=str(checkpoint), parent_sha256=sha(checkpoint),
        acquired_state=acquired, acquired_state_sha256=state_hash(acquired),
        final_acquisition_receipt=str(receipt_path), final_acquisition_receipt_sha256=sha(receipt_path),
        ready_barrier_file=str(ready_path), ready_barrier_sha256=sha(ready_path),
        frozen_arm=learner.incumbent, frozen_arm_rule='actual lastcompletedA structure')


def control_attribution(results):
    """Fixed20 percentage-point attribution requirements on paired tail starts."""
    required = ('replica0-live', 'replica0-reset', 'replica0-frozen')
    if any(name not in results or not results[name]['complete'] for name in required):
        return dict(passes=False, complete=False)
    live, reset, frozen = [results[name]['metrics'] for name in required]
    def rate(metrics, phase):
        return metrics[phase]['last32']['return_fraction']
    observed = [rate(live, 'A_acquire'), rate(reset, 'A_acquire'),
        rate(frozen, 'A_return'), rate(frozen, 'B_reverse'), rate(live, 'B_reverse')]
    if any(value is None or not math.isfinite(value) for value in observed):
        return dict(passes=False, complete=True, insufficient_opportunities=True)
    values = dict(reset_acquisition_lift=rate(live, 'A_acquire') - rate(reset, 'A_acquire'),
        frozen_B_drop=rate(frozen, 'A_return') - rate(frozen, 'B_reverse'),
        B_adaptation_lift=rate(live, 'B_reverse') - rate(frozen, 'B_reverse'))
    frozen_competent = bool(frozen['A_return']['last32']['passes'])
    return dict(complete=True, passes=frozen_competent and all(value >= .20 for value in values.values()),
        frozen_A_return_competent=frozen_competent,
        required_lift=.20, metric='all own-return opportunities in matched fixedlast32starts', **values)


def prepare(destination, *, algorithm, replicas=2, games_per_phase=64):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve all prospectively prepared attempts')
    if algorithm not in ('capped_beta', 'discounted_ucb') or replicas not in (2, 3):
        raise ValueError('fixed selector and two or three independent replicas required')
    if games_per_phase not in (64, 80):
        raise ValueError('choose64 or80 before native work')
    if sha(MODEL / 'rbf-model.npz') != EXPECTED:
        raise ValueError('inherited baseline mismatch')
    destination.mkdir()
    excluded = exclusions(destination)
    used = set(excluded)
    schedules, streams = {}, {}
    for replica in range(replicas):
        name = f'replica{replica}-live'
        assignments = []
        for phase_index, phase in enumerate(PHASES):
            drawn = select_starts(1943906127, 63601 + replica * 10 + phase_index,
                used, games_per_phase // 2)
            seeds = drawn['TRAIN']
            used.update(seed % base.MODULUS for seed in seeds)
            first_index = len(assignments)
            assignments.extend(dict(index=first_index + i, phase=phase,
                phase_index=i, seed=int(seed), reverse=phase == 'B_reverse',
                max_episode_ticks=2400, max_transition_cameras=1000)
                for i, seed in enumerate(seeds))
        schedules[name] = assignments
        streams[name] = dict(role='live', kind=algorithm, replica=replica,
            random_seed=2026103110 + replica, maximum_cameras=len(assignments) * 1000,
            seconds=wall_budget(len(assignments) * 1000))
    for role in ('reset', 'frozen'):
        name = f'replica0-{role}'
        matched = [row for row in schedules['replica0-live']
            if row['phase_index'] >= games_per_phase - TAIL
            and (role != 'frozen' or row['phase'] != 'A_acquire')]
        schedules[name] = [dict(row, index=i, matched_live_index=row['index'],
            matched_live_stream='replica0-live', control_prefix_omitted=True)
            for i, row in enumerate(matched)]
        streams[name] = dict(role=role, kind='reset_beta' if role == 'reset' else 'frozen',
            replica=0, random_seed=2026103199, maximum_cameras=len(matched) * 1000,
            seconds=wall_budget(len(matched) * 1000),
            parent_stream='replica0-live' if role == 'frozen' else None)
    unique_live_starts = validate_schedules(schedules, streams, games_per_phase, excluded)
    write(destination / 'schedules.json', schedules)
    names = set(FROZEN_FILES) | {'pong_carried_structure_learner.py',
        'run_pong_carried_learning.py', 'run_pong_relational_categorical_replication.py',
        Path(__file__).name}
    files = {str(Path(__file__).with_name(name)): sha(Path(__file__).with_name(name))
        for name in sorted(names)}
    spec = dict(contract=CONTRACT, source=base.source_signature(), files=files,
        assets={str(path): sha(path) for path in (MODEL / 'rbf-model.npz',
            MODEL / 'radial-cap.json', CALIBRATION)},
        schedules_sha256=sha(destination / 'schedules.json'),
        exclusions_sha256=sha(destination / 'seed-exclusion-proof.json'),
        roster=make_roster(load_calibration_scales(CALIBRATION)), streams=streams,
        algorithm=algorithm, replicas=replicas, games_per_phase=games_per_phase,
        tail_games=TAIL, workers=4, seconds_per_game=20,
        unique_live_starts=unique_live_starts,
        maximum_transition_cameras=sum(row['maximum_cameras'] for row in streams.values()),
        wall_budget='maximum cameras/84 per second times1.2; round up to next60seconds; fixed before first native; never extend',
        evidence_cap=8., discount=.95, exploration=.15,
        learner_inputs=['own_hits', 'reward', 'terminated', 'truncated', 'elapsed_seconds'],
        intervention='hidden external legal action0/2 exchange; requested command history stays intact',
        success='every live replica every phase: complete fixedlast32games,>=95%first,>=95%later,>=100later opportunities; includeall exploration',
        control_success='all-return fractions: liveA-resetA>=20pp; frozenAreturn-frozenB>=20pp; liveB-frozenB>=20pp; frozenAreturn meets same95%first/later+100later; controlscomplete and exactmatchedstarts',
        controls='onlyreplica0; exactmatchedlast32starts/phase; resetA/B/A, frozenB/A fromactualendAcheckpoint; allcontrolfeedbackisolated',
        frozen_arm='actual structure used by the last completed acquisitionA game, pinned by that receipt and its state hash; do not recompute optimism or reroll',
        control_prefix_omission='reset evidence clears eachgame; frozen parameters fixed and sensory/physical game histories reset eachgame; skipped prefixes cannot affect their policy',
        reset_selector='reset_beta uniformcold arm pergame; whole-gamechoice means within-game estimator neverchangesactions; RNG alone carries exogenous draws',
        controls_do_not_qualify_other_replicas=True, no_counterfactual_live_start_repeats=True,
        inherited='identical frozen vision/baseline/unlabeled scales and16generic relational controller structures; structural online learning only',
        matched_control_replays_are_separate_conditions=True,
        qualification_does_not_imply_full_M1B=True, no_backprop=True,
        no_privileged_policy_inputs=True, no_expert_labels=True, no_automatic_extension=True)
    spec['identity'] = state_hash(spec)
    write(destination / 'spec.json', spec)
    return dict(status='prepared_no_native', directory=str(destination),
        spec_sha256=sha(destination / 'spec.json'), streams=streams,
        maximum_transition_cameras=spec['maximum_transition_cameras'],
        scheduled_games=sum(map(len, schedules.values())))


def run_stream(destination, approved, name):
    destination = Path(destination)
    spec = verify_spec(destination, approved)
    config = spec['streams'][name]
    directory = destination / name
    if directory.exists():
        raise FileExistsError('never retry a chronological stream')
    directory.mkdir()
    began = time.monotonic()
    deadline = began + config['seconds']
    write(directory / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        seconds=config['seconds'], spec_sha256=approved, stream=name, role=config['role']))
    if config['role'] == 'frozen':
        learner, acquisition = frozen_prefix(destination, config['parent_stream'], spec, approved)
        write(directory / 'acquired-prefix.json', acquisition)
    else:
        learner = CarriedStructureLearner(config['kind'], len(spec['roster']),
            config['random_seed'], evidence_cap=spec['evidence_cap'],
            discount=spec['discount'], exploration=spec['exploration'])
    shadow = CarriedStructureLearner.from_state(json.loads(json.dumps(learner.state())))
    write(directory / 'checkpoint-initial.json', learner.state())
    seed_rngs()
    base.initialize()
    baseline = load_readout(MODEL / 'rbf-model.npz')
    assignments = read(destination / 'schedules.json')[name]
    rows = []
    for assignment in assignments:
        if time.monotonic() >= deadline:
            break
        row = evaluate_game(baseline, learner, shadow, spec['roster'], assignment,
            directory, deadline, spec['identity'])
        rows.append(row)
        write(directory / 'checkpoint-latest.json', learner.state())
        if (config['role'] == 'live' and row['complete']
                and assignment['phase'] == 'A_acquire'
                and assignment['phase_index'] == spec['games_per_phase'] - 1):
            write(directory / 'checkpoint-end-A.json', learner.state())
            write(directory / 'checkpoint-end-A-ready.json', dict(stream=name,
                runtime_identity=spec['identity'], spec_sha256=approved,
                checkpoint_sha256=sha(directory / 'checkpoint-end-A.json'),
                receipt_sha256=sha(directory / f"game-{assignment['index']:03d}.json")))
        write(directory / 'progress.json', dict(complete_games=sum(r['complete'] for r in rows),
            phase=assignment['phase'], phase_index=assignment['phase_index'],
            transitions=sum(r['transitions'] for r in rows),
            learner_sha256=state_hash(learner.state()), arm=row['arm'], hits=row['report']['hits']))
        if not row['complete']:
            break
        learner = CarriedStructureLearner.from_state(read(directory / 'checkpoint-latest.json'))
        assert state_hash(learner.state()) == state_hash(shadow.state())
    complete = len(rows) == len(assignments) and all(row['complete'] for row in rows)
    metrics = phase_metrics(rows, assignments, spec['games_per_phase'])
    summary = dict(contract=CONTRACT, stream=name, role=config['role'], complete=complete,
        scheduled_games=len(assignments), completed_games=sum(row['complete'] for row in rows),
        metrics=metrics, final_state=learner.state(), final_learner_sha256=state_hash(learner.state()),
        transitions=sum(row['transitions'] for row in rows),
        initial_cameras=sum(row['initial_cameras'] for row in rows),
        transition_camera_upper_bound=sum(row['transition_camera_upper_bound'] for row in rows),
        initial_camera_upper_bound=sum(row['initial_camera_upper_bound'] for row in rows),
        all_completed_streaming_audits_passed=all(row['streaming_audit']['passed'] for row in rows),
        passes_live_criteria=bool(config['role'] == 'live' and complete
            and all(metrics[phase]['last32']['passes'] for phase in PHASES)),
        wall_seconds=time.monotonic() - began,
        source_and_assets_frozen=verify_spec(destination, approved) == spec)
    write(directory / 'summary.json', summary)
    return summary


def run(destination, approved):
    destination = Path(destination)
    spec = verify_spec(destination, approved)
    if (destination / 'coordinator-clock.json').exists():
        raise FileExistsError('preserve original dispatch; no automatic resume')
    write(destination / 'coordinator-clock.json', dict(started_wall_ns=time.time_ns(),
        spec_sha256=approved, maximum_transition_cameras=spec['maximum_transition_cameras']))
    ready = [name for name, row in spec['streams'].items() if row['role'] != 'frozen']
    frozen = [name for name, row in spec['streams'].items() if row['role'] == 'frozen']
    results, errors, dispatched = {}, {}, []
    with ProcessPoolExecutor(max_workers=spec['workers'], mp_context=mp.get_context('spawn'),
            initializer=base.initialize) as pool:
        active = {}
        while ready or frozen or active:
            for name in list(frozen):
                parent = destination / spec['streams'][name]['parent_stream']
                if (parent / 'checkpoint-end-A-ready.json').exists():
                    ready.append(name)
                    frozen.remove(name)
                elif spec['streams'][name]['parent_stream'] in results or spec['streams'][name]['parent_stream'] in errors:
                    errors[name] = dict(error='parent did not acquire a complete end-A state', native_started=False)
                    frozen.remove(name)
            while ready and len(active) < spec['workers']:
                name = ready.pop(0)
                active[pool.submit(run_stream, str(destination), approved, name)] = name
                dispatched.append(name)
                write(destination / 'dispatch.json', dict(streams=dispatched))
            if not active:
                if frozen:
                    raise RuntimeError('unresolvable frozen-control dependency')
                break
            done, _ = wait(active, timeout=2., return_when=FIRST_COMPLETED)
            for future in done:
                name = active.pop(future)
                try:
                    results[name] = future.result()
                except Exception as error:
                    errors[name] = dict(error=repr(error), native_cost_may_be_unknown=True,
                        maximum_transition_camera_charge=spec['streams'][name]['maximum_cameras'])
                    write(destination / f'{name}-error.json', errors[name])
            write(destination / 'coordinator-progress.json', dict(dispatched=dispatched,
                active=list(active.values()), pending=ready + frozen,
                completed=list(results), errors=errors))
    complete = not errors and len(results) == len(spec['streams']) and all(row['complete'] for row in results.values())
    live_passed = complete and all(row['passes_live_criteria'] for name, row in results.items()
        if spec['streams'][name]['role'] == 'live')
    attribution = control_attribution(results)
    all_audits = len(results) == len(spec['streams']) and all(
        row['all_completed_streaming_audits_passed'] and row['source_and_assets_frozen']
        for row in results.values())
    source_error = None
    try:
        source_frozen = verify_spec(destination, approved) == spec
    except Exception as error:
        source_frozen, source_error = False, repr(error)
    passed = bool(live_passed and attribution['passes'] and all_audits and source_frozen)
    summary = dict(contract=CONTRACT, complete=complete, qualification_passed=passed,
        passes_live_criteria=bool(live_passed),
        all_live_tail_criteria_passed=bool(live_passed), control_attribution=attribution,
        all_streaming_and_worker_source_audits_passed=all_audits,
        source_and_assets_frozen=source_frozen, source_audit_error=source_error,
        results=results, errors=errors, controls_apply_to_replica0_only=True,
        transitions=sum(row['transitions'] for row in results.values()),
        initial_cameras=sum(row['initial_cameras'] for row in results.values()),
        maximum_transition_cameras=spec['maximum_transition_cameras'],
        no_automatic_extension=True, full_M1B_completion=False)
    write(destination / 'qualification-summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--directory', required=True)
    parser.add_argument('--approved-spec')
    parser.add_argument('--algorithm', choices=('capped_beta', 'discounted_ucb'))
    parser.add_argument('--replicas', type=int, choices=(2, 3), default=2)
    parser.add_argument('--games-per-phase', type=int, choices=(64, 80), default=64)
    args = parser.parse_args()
    output = prepare(args.directory, algorithm=args.algorithm, replicas=args.replicas,
        games_per_phase=args.games_per_phase) if args.command == 'prepare' else run(args.directory, args.approved_spec)
    print(json.dumps(output, sort_keys=True), flush=True)
