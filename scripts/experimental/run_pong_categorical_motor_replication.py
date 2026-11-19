"""Prospectively fixed native Pong transfer of the categorical motor learner."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from fly_connectome.control.session import MODEL_HASH, model_bytes
from fly_connectome.vision import VisualStateEncoder
from pong_categorical_motor_fir import CategoricalMotorFIR
from pong_visual_motor_fir import DEFAULTS
from run_pong_carried_learning import policy_from_entry
from run_pong_relational_categorical_replication import make_roster
from run_rally_option_memory import MODEL, EXPECTED, read, sha, write, seed_rngs
from run_rally_repertoire_collection import exclusions, select_starts
import run_pong_categorical_motor_learning as pilot
import run_pong_visual_motor_learning as runner
import run_practical_pong_v2 as base

CONTRACT = 'categorical-visual-motor-native-neutral-fallback-transfer-v1'
ROOT_SEED = 428540879
LEARNER_SEEDS = {'0': 2026111901, '1': 2026111902}
PHASES = {'A_acquire': (32, [0, 1, 2]), 'B_cycle': (40, [1, 2, 0]), 'A_return': (32, [0, 1, 2])}
FINAL_GAMES = 24
MIN_LATER_OPPORTUNITIES = 95
SECONDS = 2400
MAX_CAMERAS = 248000
ROOT = Path(__file__).resolve().parents[2]


class NeutralReadout:
    def __init__(self):
        self.base = SimpleNamespace(weights=np.zeros((3, VisualStateEncoder(64, 64, .02).dimension)))

    def frozen_parameters(self):
        return ('constant-neutral-request', 1)


class NeutralEvaluator(runner.NumericEvaluator):
    def _baseline_choice(self):
        return 1, np.zeros(3)


def prior_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def dependencies():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')}


def prepare(destination):
    model_bytes()
    if destination.exists():
        raise FileExistsError('preserve previous specifications and evidence')
    destination.mkdir(parents=True)
    used = exclusions(destination)
    # Include committed receipt metadata and prospective draws of the recovered prior pilot.
    proof_path = destination / 'seed-exclusion-proof.json'
    proof = read(proof_path)
    committed = {}
    def collect(value):
        if type(value) is int and 0 < value < 2**32:
            used.add(value % base.MODULUS)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
    for path in sorted((ROOT / 'docs').rglob('*.json')):
        collect(read(path))
        committed[str(path.relative_to(ROOT))] = sha(path)
    for tag in (63611, 63612, 63613, 63631, 63632, 63633):
        rng = np.random.default_rng(np.random.SeedSequence([1943906127, tag]))
        used.update(int(seed) % base.MODULUS for seed in rng.integers(1, 2**32, size=4096, dtype=np.uint64))
    used.update({ROOT_SEED, 1939032277, *(s % base.MODULUS for s in LEARNER_SEEDS.values())})
    proof.update(committed_metadata_sha256=committed, canonical_ids=sorted(used), ids=len(used),
        prior_artifact_tree_present=bool(proof['metadata_sha256']),
        earlier_pilot_candidate_exclusions=dict(root=1943906127, tags=[63611, 63612, 63613, 63631, 63632, 63633],
            draws_per_tag=4096),
        historical_exclusion_scope='Available committed evidence, local schedules/receipts, and reserved earlier pilot draws; complete earlier archive unavailable.')
    write(proof_path, proof)
    excluded = used.copy()
    schedules = {}
    for replica in LEARNER_SEEDS:
        index = 0
        schedules[replica] = {}
        for offset, (phase, (count, mapping)) in enumerate(PHASES.items()):
            tag = 63701 + 10 * int(replica) + offset
            draw = select_starts(ROOT_SEED, tag, used, count // 2)
            used.update(seed % base.MODULUS for seed in draw['TRAIN'])
            schedules[replica][phase] = [dict(index=index + i, phase=phase, phase_index=i, seed=seed,
                intervention={'action_map': mapping}) for i, seed in enumerate(draw['TRAIN'])]
            index += count
    canonical = [a['seed'] % base.MODULUS for replica in schedules.values() for phase in replica.values() for a in phase]
    assert len(canonical) == len(set(canonical)) == 208 and not excluded.intersection(canonical)
    write(destination / 'schedules.json', schedules)
    files = [Path(__file__), destination / 'schedules.json', proof_path]
    files.extend(Path(__file__).with_name(name) for name in sorted(set(runner.compact.FILES) | {
        'run_pong_categorical_motor_learning.py', 'run_pong_visual_motor_learning.py',
        'pong_categorical_motor_fir.py', 'pong_visual_motor_fir.py', 'run_pong_carried_learning.py',
        'run_pong_relational_categorical_replication.py', 'run_rally_repertoire_collection.py'}))
    spec = dict(contract=CONTRACT, parent_commit=prior_commit(), source=base.source_signature(),
        files={str(path): sha(path) for path in files}, dependencies=dependencies(), frozen_vision_sha256=MODEL_HASH,
        source_asset_unavailable=dict(path=str(MODEL / 'rbf-model.npz'), expected_sha256=EXPECTED),
        fallback='constant request index 1; no inherited readout or policy-feature scaling',
        inherited=make_roster(np.ones(4))[4], settings=DEFAULTS, learner_seeds=LEARNER_SEEDS,
        numerical_coefficients=96, seconds=SECONDS, workers=3, core_budget=4, opponent_speed=.65,
        max_episode_ticks=2400, max_episode_physics_seconds=20, phase_games={p: n for p, (n, _) in PHASES.items()},
        live_games=208, frozen_games=40, total_games=248, maximum_transition_cameras=MAX_CAMERAS,
        maximum_initial_cameras=248, matched_frozen='replica 0 B, true end-A checkpoint, coefficient and covariance updates disabled',
        phase_gate=dict(final_games=FINAL_GAMES, minimum_first_opportunities=FINAL_GAMES,
            minimum_first_return_rate=.95, minimum_later_opportunities=MIN_LATER_OPPORTUNITIES,
            minimum_later_return_rate=.95),
        frozen_gate=dict(phase='B_cycle', replica='0', games=40, minimum_first_rate_margin=.20,
            minimum_all_opportunity_rate_margin=.20),
        recovery='full phase and fixed pre-tail totals; simulation time, cameras, and both kinds of misses; first successful return latency; count within-game model-dependent decisions before first opportunity',
        no_backprop=True, no_privileged_inputs=True, no_fitting_of_inherited_parameters=True,
        no_extension=True, full_public_replay=True, exact_prior_replication=False)
    write(destination / 'spec.json', spec)
    return sha(destination / 'spec.json')


def verify(destination, digest):
    assert sha(destination / 'spec.json') == digest
    spec = read(destination / 'spec.json')
    assert spec['source'] == base.source_signature() and spec['dependencies'] == dependencies()
    assert spec['frozen_vision_sha256'] == MODEL_HASH
    model_bytes()
    assert all(sha(path) == expected for path, expected in spec['files'].items())
    return spec


def play_stream(directory, spec, assignments, deadline, seed=None, acquired=None):
    pilot.configure_runtime()
    runner.CONTRACT = CONTRACT
    base.initialize()
    seed_rngs()
    directory.mkdir(parents=True, exist_ok=True)
    baseline = NeutralReadout()
    policy = policy_from_entry(spec['inherited'])
    learner = (CategoricalMotorFIR(seed, settings=spec['settings']) if acquired is None
        else CategoricalMotorFIR.from_state(acquired).frozen_branch())
    write(directory / 'checkpoint-start.json', learner.state())
    shadow = CategoricalMotorFIR.from_state(learner.state())
    initial_model = (learner.theta.copy(), learner.covariance.copy(), learner.updates)
    for assignment in assignments:
        if time.monotonic() >= deadline:
            break
        started = directory / 'active-game.json'
        write(started, dict(assignment=assignment, wall_ns=time.time_ns(), transition_grant=1000, initial_grant=1))
        row = runner.evaluate_game(baseline, policy, learner, shadow, assignment, directory, deadline,
            evaluator_class=NeutralEvaluator, use_policy_scaling=False)
        if acquired is not None:
            assert np.array_equal(learner.theta, initial_model[0])
            assert np.array_equal(learner.covariance, initial_model[1]) and learner.updates == initial_model[2]
        write(directory / 'checkpoint-latest.json', learner.state())
        if assignment['phase'] == 'A_acquire' and assignment['phase_index'] == PHASES['A_acquire'][0] - 1 and row['complete']:
            write(directory / 'checkpoint-end-A.json', learner.state())
        started.unlink()
        if not row['complete']:
            break
        learner = CategoricalMotorFIR.from_state(read(directory / 'checkpoint-latest.json'))


def score(rows):
    result = runner.summarize(rows)
    aggregate = result['aggregate']
    first_total = aggregate['first_incoming_opportunities']
    first_returns = aggregate['first_incoming_returns']
    later_total = aggregate['own_opportunities'] - first_total
    later_returns = aggregate['own_returns'] - first_returns
    result.update(first_misses=first_total - first_returns, later_misses=later_total - later_returns,
        later_opportunities=later_total, later_returns=later_returns,
        later_return_fraction=later_returns / later_total if later_total else None,
        changes_before_first_opportunity=sum(row.get('full_public_replay', {}).get(
            'within_game_changes_before_first_opportunity', 0) for row in rows))
    return result


def trial(row):
    first = row['annotation']['first_incoming']
    audit = row.get('full_public_replay', {})
    return dict(index=row['assignment']['phase_index'], seed=row['assignment']['seed'],
        complete=row['complete'], first_returned=first['success'] if first else None,
        first_opportunity_tick=first['tick'] if first else None,
        returns=row['annotation']['own_returns'], opportunities=row['annotation']['own_opportunities'],
        time_seconds=row['annotation']['seconds'], transition_cameras=row['transitions'],
        within_game_changes=audit.get('within_game_update_changed_unperturbed_actions'),
        changes_before_first_opportunity=audit.get('within_game_changes_before_first_opportunity'),
        first_within_game_change=audit.get('first_within_game_change'))


def phase_report(rows, expected):
    tail = [r for r in rows if r['assignment']['phase_index'] >= expected - FINAL_GAMES]
    pre_tail = [r for r in rows if r['assignment']['phase_index'] < expected - FINAL_GAMES]
    tail_score = score(tail)
    aggregate = tail_score['aggregate']
    first_total = aggregate['first_incoming_opportunities']
    first_returns = aggregate['first_incoming_returns']
    later_total = tail_score['later_opportunities']
    passed = (tail_score['completed_games'] == FINAL_GAMES and first_total == FINAL_GAMES
        and first_returns * 100 >= first_total * 95 and later_total >= MIN_LATER_OPPORTUNITIES
        and tail_score['later_returns'] * 100 >= later_total * 95)
    elapsed = 0.
    preceding_opportunities = 0
    first_success = None
    for row in rows:
        first = row['annotation']['first_incoming']
        if first is not None and first['success']:
            first_success = dict(phase_game=1 + row['assignment']['phase_index'],
                elapsed_physics_seconds=elapsed + first['tick'] / 120,
                previous_first_misses=preceding_opportunities)
            break
        elapsed += row['annotation']['seconds']
        preceding_opportunities += int(first is not None)
    return dict(full=score(rows), pre_tail=score(pre_tail), final=tail_score, gate_passed=passed,
        first_success=first_success, games=[trial(row) for row in rows])


def receipts(directory):
    return [read(path) for path in sorted(directory.glob('game-*.json'))]


def continuity(rows, start):
    previous = runner.state_hash(start)
    for row in rows:
        if previous != row['learner_before_sha256'] or previous != runner.state_hash(row['learner_before']):
            return False
        previous = runner.state_hash(row['learner_after'])
        if previous != row['learner_after_sha256']:
            return False
    return True


def compare(live, frozen):
    a, b = score(live), score(frozen)
    pairs, geometry = [], True
    def first_geometry(row):
        first = row['annotation']['first_incoming']
        return {k: v for k, v in first.items() if k != 'success'} if first else None
    for one, other in zip(live, frozen):
        geometry &= (one['assignment'] == other['assignment'] and first_geometry(one) == first_geometry(other))
        pairs.append(dict(live=trial(one), frozen=trial(other)))
    aa, bb = a['aggregate'], b['aggregate']
    def difference(key):
        return (aa[key] - bb[key]) if aa[key] is not None and bb[key] is not None else None
    all_margin = difference('return_fraction')
    first_margin = difference('first_incoming_return_fraction')
    return dict(gate_passed=(a['completed_games'] == b['completed_games'] == PHASES['B_cycle'][0]
            and geometry and all_margin is not None and all_margin >= .20
            and first_margin is not None and first_margin >= .20),
        all_margin=all_margin, first_margin=first_margin, matched_geometry=geometry,
        frozen=score(frozen), pairs=pairs)


def run(destination, digest):
    spec = verify(destination, digest)
    began = time.monotonic()
    deadline = began + spec['seconds']
    # Atomic exclusive creation guards the chronological allocation against repeats.
    with (destination / 'work-clock.json').open('x') as clock:
        json.dump(dict(started_wall_ns=time.time_ns(), began_monotonic=began,
            deadline_monotonic=deadline, spec_sha256=digest), clock)
    schedules = read(destination / 'schedules.json')
    directories = {replica: destination / f'live-{replica}' for replica in LEARNER_SEEDS}
    frozen_directory = destination / 'frozen-0'
    errors = []
    frozen_launched = False
    with ProcessPoolExecutor(max_workers=spec['workers'], mp_context=mp.get_context('spawn')) as pool:
        active = {pool.submit(play_stream, directories[replica], spec,
            [a for phase in PHASES for a in schedules[replica][phase]], deadline, seed): replica
            for replica, seed in LEARNER_SEEDS.items()}
        end_a = directories['0'] / 'checkpoint-end-A.json'
        while active:
            if end_a.exists() and not frozen_launched and time.monotonic() < deadline:
                acquired = read(end_a)
                if acquired['updates'] >= spec['settings']['initial_dither_updates']:
                    active[pool.submit(play_stream, frozen_directory, spec, schedules['0']['B_cycle'],
                        deadline, acquired=acquired)] = 'frozen-0'
                else:
                    errors.append(dict(stream='frozen-0', error='end-A checkpoint has not reached the continuing probe clock'))
                frozen_launched = True
            completed, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
            for future in completed:
                stream = active.pop(future)
                try:
                    future.result()
                except Exception as error:
                    errors.append(dict(stream=stream, error=repr(error)))
    live = {r: receipts(directory) for r, directory in directories.items()}
    frozen = receipts(frozen_directory)
    controls = compare([row for row in live['0'] if row['assignment']['phase'] == 'B_cycle'], frozen)
    reports = {replica: {phase: phase_report([r for r in rows if r['assignment']['phase'] == phase], expected)
        for phase, (expected, _) in PHASES.items()} for replica, rows in live.items()}
    inventory = {**directories, 'frozen-0': frozen_directory}
    stream_rows = {**live, 'frozen-0': frozen}
    all_rows = [*live['0'], *live['1'], *frozen]
    live_chain = {key: (directory / 'checkpoint-start.json').exists()
        and continuity(live[key], read(directory / 'checkpoint-start.json')) for key, directory in directories.items()}
    acquired_file = directories['0'] / 'checkpoint-end-A.json'
    frozen_start_file = frozen_directory / 'checkpoint-start.json'
    frozen_chain = False
    if acquired_file.exists() and frozen_start_file.exists():
        acquired, frozen_start = read(acquired_file), read(frozen_start_file)
        actual_end_a = next((row for row in live['0'] if row['assignment']['index'] == 31), None)
        frozen_chain = (actual_end_a is not None and runner.state_hash(acquired) == actual_end_a['learner_after_sha256']
            and frozen_start == dict(acquired, updates_enabled=False) and continuity(frozen, frozen_start)
            and all(row['new_numeric_updates'] == 0 and not row['coefficients_changed']
                and row['learner_after']['covariance'] == acquired['covariance'] for row in frozen))
    all_replays = len(all_rows) == spec['total_games'] and all(
        row.get('full_public_replay', {}).get('passed') and sha(row['tape_file']) == row['tape_sha256']
        and sha(row['public_file']) == row['public_sha256'] for row in all_rows)
    complete = len(all_rows) == spec['total_games'] and all(row['complete'] for row in all_rows) and not errors
    total_transitions = sum(row['transition_camera_upper_bound'] for row in all_rows)
    initial_cameras = sum(row['initial_camera_upper_bound'] for row in all_rows)
    # Outstanding grants are charged when a worker exits without a saved game receipt.
    unsaved = []
    for stream, directory in inventory.items():
        path = directory / 'active-game.json'
        if path.exists():
            active_game = read(path)
            if not any(row['assignment'] == active_game['assignment'] for row in stream_rows[stream]):
                unsaved.append(dict(stream=stream, **active_game))
    charged_transitions = total_transitions + sum(r['transition_grant'] for r in unsaved)
    charged_initial = initial_cameras + sum(r['initial_grant'] for r in unsaved)
    assert charged_transitions <= spec['maximum_transition_cameras'] and charged_initial <= spec['maximum_initial_cameras']
    unchanged = verify(destination, digest) == spec
    gates = all(phase['gate_passed'] for replica in reports.values() for phase in replica.values())
    result = dict(contract=CONTRACT, complete=complete, transfer_passed=bool(complete and gates and
        controls['gate_passed'] and all(live_chain.values()) and frozen_chain and all_replays and unchanged),
        exact_prior_replication=False, live=reports, comparison=controls, errors=errors,
        all_public_replays=all_replays, learner_continuity=live_chain, frozen_checkpoint_audit=frozen_chain,
        unchanged_sources_assets_spec=unchanged, wall_seconds=time.monotonic() - began,
        transitions=sum(row['transitions'] for row in all_rows), initial_cameras=sum(row['initial_cameras'] for row in all_rows),
        charged_transitions=charged_transitions, charged_initial=charged_initial, unsaved_grants=unsaved,
        spec_sha256=digest)
    write(destination / 'summary.json', result)
    return {key: result[key] for key in ('complete', 'transfer_passed', 'transitions', 'initial_cameras', 'errors')}


def smoke(destination):
    pilot.configure_runtime()
    runner.CONTRACT = CONTRACT
    base.initialize()
    seed_rngs()
    policy = policy_from_entry(make_roster(np.ones(4))[4])
    learner = CategoricalMotorFIR(2026111900)
    shadow = CategoricalMotorFIR.from_state(learner.state())
    row = runner.evaluate_game(NeutralReadout(), policy, learner, shadow, dict(index=0,
        phase='preflight', phase_index=0, seed=1939032277, intervention={'action_map': [0, 1, 2]}),
        destination, time.monotonic() + 60, evaluator_class=NeutralEvaluator, use_policy_scaling=False,
        max_episode_ticks=120)
    assert row['complete'] and row['full_public_replay']['passed']
    return dict(transitions=row['transitions'], initial_cameras=row['initial_cameras'], wall_seconds=row['wall_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'smoke'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--spec-sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = dict(spec_sha256=prepare(args.directory.resolve()))
    elif args.command == 'smoke':
        result = smoke(args.directory.resolve())
    else:
        result = run(args.directory.resolve(), args.spec_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
