"""Prospective paired native confirmation of public motor gain recalibration."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from fly_connectome.control.session import MODEL_HASH, model_bytes, source_signature
from pong_categorical_motor_fir import CategoricalMotorController
from pong_motor_recalibration import BoundedCategoricalMotor, MotorGainController, MotorGainLearner
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
import run_pong_inverse_cycle as native
from run_pong_surprise_recovery import POLICY, exclusions, read, reached_deadline, runtime_dependencies, state_hash


ROOT = Path(__file__).resolve().parents[2]
START_SEED = 198427313
PROBE_SEEDS = (197271091, 530698177)
MODULUS = 2147483647
PHASES = (('A', (0, 1, 2)), ('B', (1, 2, 0)), ('C', (2, 0, 1)), ('restored', (0, 1, 2)))
GAMES = 32
TAIL = 20
MIN_LATER = 78
SECONDS = 2400
FILES = ('run_pong_motor_recalibration.py', 'run_pong_inverse_cycle.py',
         'run_pong_surprise_recovery.py', 'pong_motor_recalibration.py',
         'pong_categorical_surprise_recovery.py',
         'pong_categorical_motor_fir.py', 'pong_visual_motor_fir.py',
         'pong_relational_feedback.py', 'pong_relational_persistent_binding.py',
         'public_entity_pong_native.py', 'return_metrics.py')


def prepare(directory):
    model_bytes()
    used, metadata = exclusions(directory.parent)
    used.update((*PROBE_SEEDS, START_SEED))
    schedules = []
    for replica, probe_seed in enumerate(PROBE_SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([START_SEED, replica]))
        games = []
        for phase, mapping in PHASES:
            for index in range(GAMES):
                while True:
                    seed = int(rng.integers(1, MODULUS))
                    if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == index % 2:
                        break
                used.add(seed)
                games.append(dict(index=len(games), phase=phase, phase_index=index,
                                  seed=seed, mapping=list(mapping)))
        schedules.append(dict(replica=replica, probe_seed=probe_seed, games=games))
    directory.mkdir(parents=True, exist_ok=False)
    native.save(directory / 'schedules.json', schedules)
    spec = dict(contract='native-paired-gain-recalibration-qualification-v1',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source=source_signature(), dependencies=runtime_dependencies(), vision_asset_sha256=MODEL_HASH,
        files={name: native.sha(Path(__file__).with_name(name)) for name in FILES},
        schedule_sha256=native.sha(directory / 'schedules.json'), previously_excluded_metadata=metadata,
        seed_exclusion_scope='available schedules and exclusion proofs', start_seed=START_SEED,
        probe_seeds=PROBE_SEEDS, phases=PHASES, games_per_phase=GAMES, normal_episode_seconds=20,
        neutral_unsupported_fallback=True, inherited_policy=POLICY, hidden_maps=True,
        frozen='true candidate replica 0 end-A response, disabled numerical/gain updates and new calibrations, ordinary continuing probes, no queued recalibration burst',
        stable='bounded unconstrained 96-response model; same candidate end-A state if still full-model, matched separate normal A otherwise',
        budget=dict(seconds=SECONDS, workers=3, planned_games=320, conditional_stable_normal_games=32,
                    max_games=352, max_transitions=352000, max_initial=352),
        acceptance=dict(final_games=TAIL, required_first_opportunities=TAIL,
                        minimum_first_and_later_rate=.95, minimum_later_opportunities=MIN_LATER,
                        replica0_full_B_all_opportunity_lift_vs_frozen=.20),
        diagnostics='each phase full, first8 and final20; elapsed physics time and cameras through the first two successive games of at least five returns, including misses through that recovery',
        extension=False)
    native.save(directory / 'spec.json', spec)
    return native.sha(directory / 'spec.json')


def verify(directory, digest):
    assert native.sha(directory / 'spec.json') == digest
    spec = read(directory / 'spec.json')
    model_bytes()
    assert spec['vision_asset_sha256'] == MODEL_HASH
    assert spec['source'] == source_signature() and spec['dependencies'] == runtime_dependencies()
    assert spec['schedule_sha256'] == native.sha(directory / 'schedules.json')
    assert all(native.sha(Path(__file__).with_name(name)) == expected for name, expected in spec['files'].items())
    return spec


def play_stream(directory, name, learner, controller, games, deadline, end_a=None):
    signal.signal(signal.SIGALRM, reached_deadline)
    torch.set_num_threads(1)
    native.CategoricalMotorController = controller
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    starting_state = learner.state()
    native.save(directory / f'{name}-start.json', starting_state)
    active = directory / f'{name}-active.json'
    rows = []
    for game in games:
        if time.monotonic() >= deadline:
            return
        before = state_hash(learner.state())
        start = time.monotonic()
        probes = learner.dither_count
        recalibrations = getattr(learner, 'calibrations', 0)
        native.save(active, dict(stream=name, index=game['index']))
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - start))
        try:
            outcome = native.play(learner, policy, game['seed'], game['mapping'])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        after = learner.state()
        assert outcome['initial'] == 1 and outcome['transitions'] <= 1000
        if not starting_state['updates_enabled']:
            assert all(after[key] == starting_state[key] for key in
                       ('theta', 'covariance', 'kernel', 'gains', 'gain_covariance', 'updates', 'calibrations'))
        row = dict(**game, outcome=outcome, before=before, after=state_hash(after),
            probe_frames=learner.dither_count - probes,
            recalibrations=getattr(learner, 'calibrations', 0) - recalibrations,
            numerical_form='three_gain' if after.get('kernel') else 'full_96', wall_seconds=time.monotonic() - start)
        rows.append(row)
        native.save(directory / f'{name}-games.json', rows)
        native.save(directory / f'{name}-latest.json', after)
        active.unlink()
        if game['phase_index'] == GAMES - 1:
            if end_a is not None and game['phase'] == 'A':
                native.save(end_a, after)
            print(name, game['phase'], 'complete', flush=True)


def candidate(directory, name, spec, deadline, end_a=None):
    learner = MotorGainLearner(spec['probe_seed'])
    play_stream(directory, name, learner, MotorGainController, spec['games'], deadline, end_a)


def controls(directory, spec, acquired, deadline):
    assert acquired['updates'] >= acquired['settings']['initial_dither_updates']
    games_b = [row for row in spec['games'] if row['phase'] == 'B']
    frozen = MotorGainLearner.from_state(dict(acquired, updates_enabled=False, probe_sequence=[]))
    play_stream(directory, 'frozen', frozen, MotorGainController, games_b, deadline)
    stable = BoundedCategoricalMotor(spec['probe_seed'])
    if acquired['kernel']:
        stable_games = [row for row in spec['games'] if row['phase'] in ('A', 'B')]
    else:
        stable = BoundedCategoricalMotor.from_state({k: acquired[k] for k in stable.state()})
        stable_games = games_b
    play_stream(directory, 'stable', stable, CategoricalMotorController, stable_games, deadline)


def tally(rows):
    games = [r['outcome'] for r in rows]
    hits, opportunities = sum(x['returns'] for x in games), sum(x['opportunities'] for x in games)
    first_hits = sum(bool(x['events'] and x['events'][0]['hit']) for x in games)
    first_ops = sum(bool(x['events']) for x in games)
    return dict(games=len(rows), hits=hits, opportunities=opportunities, misses=opportunities - hits,
        first_hits=first_hits, first_opportunities=first_ops, first_misses=first_ops - first_hits,
        later_hits=hits - first_hits, later_opportunities=opportunities - first_ops,
        later_misses=(opportunities - first_ops) - (hits - first_hits),
        physics_seconds=sum(x['seconds'] for x in games), cameras=sum(x['transitions'] for x in games),
        initial_cameras=sum(x['initial'] for x in games), probe_frames=sum(r['probe_frames'] for r in rows),
        recalibrations=sum(r['recalibrations'] for r in rows))


def phase_report(rows):
    tail = tally([r for r in rows if r['phase_index'] >= GAMES - TAIL])
    passed = (tail['games'] == TAIL and tail['first_opportunities'] == TAIL
        and 20 * tail['first_hits'] >= 19 * TAIL and tail['later_opportunities'] >= MIN_LATER
        and 20 * tail['later_hits'] >= 19 * tail['later_opportunities'])
    attained = None
    for i, (previous, current) in enumerate(zip(rows, rows[1:]), start=2):
        if min(previous['outcome']['returns'], current['outcome']['returns']) >= 5:
            attained = tally(rows[:i])
            break
    return dict(full=tally(rows), first8=tally(rows[:8]), final20=tail, tail_passed=passed,
        recovery_to_two_successive_five_return_games=attained, recovery_censored=attained is None)


def summarize(directory, schedules, errors, elapsed, unchanged):
    rows = {}
    checks = {}
    acquired_file = directory / 'candidate-r0-end-A.json'
    acquired = read(acquired_file) if acquired_file.exists() else None
    for name in ('candidate-r0', 'candidate-r1', 'frozen', 'stable'):
        games_path = directory / f'{name}-games.json'
        start_path = directory / f'{name}-start.json'
        latest_path = directory / f'{name}-latest.json'
        rows[name] = read(games_path) if games_path.exists() else []
        start = read(start_path) if start_path.exists() else None
        previous = state_hash(start) if start else None
        replica = int(name[-1]) if name.startswith('candidate') else 0
        phases = {'A', 'B', 'C', 'restored'} if name.startswith('candidate') else {'B'}
        if name == 'stable' and acquired and acquired['kernel']:
            phases.add('A')
        expected = [game for game in schedules[replica]['games'] if game['phase'] in phases]
        exact = len(rows[name]) == len(expected)
        for row, game in zip(rows[name], expected):
            exact &= all(row[k] == v for k, v in game.items()) and row['before'] == previous
            previous = row['after']
        exact &= latest_path.exists() and state_hash(read(latest_path)) == previous
        if name == 'frozen':
            exact &= bool(acquired and start == dict(acquired, updates_enabled=False, probe_sequence=[]))
        if name == 'stable' and acquired and not acquired['kernel'] and start:
            exact &= start == {key: acquired[key] for key in start}
        checks[name] = exact
    end_a = next((r for r in rows['candidate-r0'] if r['index'] == GAMES - 1), None)
    checks['acquired_checkpoint'] = bool(acquired and end_a and state_hash(acquired) == end_a['after'])
    phase_results = {name: {phase: phase_report([r for r in results if r['phase'] == phase])
        for phase, _ in PHASES if any(r['phase'] == phase for r in results)} for name, results in rows.items()}
    b = [tally([r for r in rows[name] if r['phase'] == 'B']) for name in ('candidate-r0', 'frozen', 'stable')]
    lift = Fraction(b[0]['hits'], b[0]['opportunities']) - Fraction(b[1]['hits'], b[1]['opportunities']) \
        if b[0]['opportunities'] and b[1]['opportunities'] else None
    complete = all(checks.values()) and not errors and unchanged
    tails = all(len(phase_results[f'candidate-r{i}']) == len(PHASES) and all(
        result['tail_passed'] for result in phase_results[f'candidate-r{i}'].values()) for i in range(2))
    all_rows = [r for group in rows.values() for r in group]
    costs = tally(all_rows)
    outstanding = [read(path) for path in directory.glob('*-active.json')]
    unsaved = [entry for entry in outstanding if not any(r['index'] == entry['index'] for r in rows[entry['stream']])]
    charged = dict(cameras=costs['cameras'] + 1000 * len(unsaved),
                   initial_cameras=costs['initial_cameras'] + len(unsaved))
    assert charged['cameras'] <= 352000 and charged['initial_cameras'] <= 352
    return dict(complete=complete, passed=bool(complete and tails and lift is not None and lift >= Fraction(1, 5)),
        all_tail_gates=tails, b_frozen_lift=float(lift) if lift is not None else None,
        b_frozen_lift_passed=bool(lift is not None and lift >= Fraction(1, 5)),
        stable_repeat_normal_a=bool(acquired and acquired['kernel']), checks=checks, phase_results=phase_results,
        costs=costs, charged=charged, outstanding=outstanding, errors=errors, wall_seconds=elapsed)


def run(directory, digest):
    spec = verify(directory, digest)
    start = time.monotonic()
    deadline = start + spec['budget']['seconds']
    with (directory / 'work-clock.json').open('x') as clock:
        json.dump(dict(spec_sha256=digest, started_wall_ns=time.time_ns(), deadline_monotonic=deadline), clock)
    schedules = read(directory / 'schedules.json')
    end_a = directory / 'candidate-r0-end-A.json'
    controls_launched = False
    errors = []
    with ProcessPoolExecutor(max_workers=spec['budget']['workers'], mp_context=mp.get_context('spawn')) as pool:
        active = {pool.submit(candidate, directory, f'candidate-r{i}', item, deadline, end_a if i == 0 else None):
                  f'candidate-r{i}' for i, item in enumerate(schedules)}
        while active:
            if not controls_launched and end_a.exists() and time.monotonic() < deadline:
                active[pool.submit(controls, directory, schedules[0], read(end_a), deadline)] = 'controls'
                controls_launched = True
            done, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
            for future in done:
                name = active.pop(future)
                try:
                    future.result()
                except Exception as error:
                    errors.append(dict(stream=name, error=repr(error)))
    result = summarize(directory, schedules, errors, time.monotonic() - start, verify(directory, digest) == spec)
    native.save(directory / 'summary.json', dict(spec_sha256=digest, **result))
    return {key: result[key] for key in ('complete', 'passed', 'all_tail_gates', 'b_frozen_lift', 'costs', 'errors')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--spec-sha256')
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = dict(spec_sha256=prepare(directory)) if args.command == 'prepare' else run(directory, args.spec_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
