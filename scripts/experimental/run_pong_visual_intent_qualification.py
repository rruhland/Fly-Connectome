"""Prospective paired native qualification of measured visual command effects."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
import json
import multiprocessing as mp
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fly_connectome.control.session import MODEL_HASH, model_bytes, source_signature
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
import run_pong_inverse_cycle as native
from run_pong_surprise_recovery import POLICY, exclusions, read, reached_deadline, runtime_dependencies, score, state_hash
from run_pong_visual_button_scout import FILES as SCOUT_FILES, play_game


START_SEED = 469287301
PROBE_SEEDS = (507063749, 681202159)
MODULUS = 2147483647
PHASES = (('normal', (0, 1, 2)), ('cycle', (1, 2, 0)), ('inverse', (2, 0, 1)), ('restored', (0, 1, 2)))
PHASE_LENGTH = 32
TAIL_LENGTH = 20
LATER_MINIMUM = 78
GAMES = 288
SECONDS = 2400
FILES = (*SCOUT_FILES, 'run_pong_visual_intent_qualification.py', 'pong_categorical_surprise_recovery.py')


def prepare(directory, scout_directory):
    model_bytes()
    used, metadata = exclusions(directory.parent)
    used.update((*PROBE_SEEDS, START_SEED))
    schedules = []
    for stream, probe_seed in enumerate(PROBE_SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([START_SEED, stream]))
        rows = []
        for phase, mapping in PHASES:
            for position in range(PHASE_LENGTH):
                while True:
                    seed = int(rng.integers(1, MODULUS))
                    if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == position % 2:
                        break
                used.add(seed)
                rows.append(dict(index=len(rows), phase=phase, phase_index=position, seed=seed, mapping=mapping))
        schedules.append(dict(probe_seed=probe_seed, games=rows))
    directory.mkdir(parents=True, exist_ok=False)
    native.save(directory / 'schedules.json', schedules)
    spec = dict(contract='paired-native-visual-intent-qualification-v1',
        sources=source_signature(), dependencies=runtime_dependencies(), vision_asset_sha256=MODEL_HASH,
        files={name: native.sha(Path(__file__).with_name(name)) for name in FILES},
        schedule_sha256=native.sha(directory / 'schedules.json'), excluded_prior_metadata=metadata,
        scout_directory=str(scout_directory), scout_evidence={str(path.relative_to(scout_directory)): native.sha(path)
            for path in (scout_directory / 'spec.json', *sorted((scout_directory / 'live').glob('game-*.json')))},
        start_seed=START_SEED, probe_seeds=PROBE_SEEDS, phase_length=PHASE_LENGTH, phases=PHASES,
        inherited_visual_intent=POLICY, execution_only_remappings=True,
        fixed_tail=dict(games=TAIL_LENGTH, minimum_first_exposure=TAIL_LENGTH,
                        minimum_later_exposure=LATER_MINIMUM, minimum_first_and_later_rate=.95),
        matched_frozen=dict(replica=0, phase='cycle', games=32, minimum_all_opportunity_lift=.20,
                            source='true end-normal checkpoint; numerical updates and calibration probes disabled'),
        recovery='first8 misses and earliest two successive completed games with five or more returns, time and observations charged through the second game',
        budget=dict(games=GAMES, seconds=SECONDS, workers=3, transition_cameras=288000,
                    initial_cameras=288), extension=False)
    native.save(directory / 'spec.json', spec)
    return native.sha(directory / 'spec.json')


def verify(directory, digest):
    path = directory / 'spec.json'
    assert native.sha(path) == digest
    spec = read(path)
    model_bytes()
    assert spec['vision_asset_sha256'] == MODEL_HASH and spec['sources'] == source_signature()
    assert spec['dependencies'] == runtime_dependencies()
    assert spec['schedule_sha256'] == native.sha(directory / 'schedules.json')
    assert all(native.sha(Path(__file__).with_name(name)) == sha for name, sha in spec['files'].items())
    return spec


def worker(directory, arm, games, learner, deadline, acquired_file=None):
    torch.set_num_threads(1)
    native.CategoricalMotorController = VisualIntentController
    signal.signal(signal.SIGALRM, reached_deadline)
    directory.mkdir(exist_ok=False)
    native.save(directory / 'start.json', learner.state())
    for assignment in games:
        row = play_game(directory, assignment, learner, arm, deadline)
        if row is None:
            return
        learner = VisualButtonEffects.from_state(row['state'])
        if assignment['phase_index'] == PHASE_LENGTH - 1:
            if acquired_file is not None and assignment['phase'] == 'normal':
                native.save(acquired_file, learner.state())
            print(arm, assignment['phase'], 'complete', flush=True)


def measure(rows):
    data = score(rows)
    del data['surprise_episodes']
    data['recalibrations'] = sum(r['recalibrations'] for r in rows)
    data['contradictions'] = sum(r['contradictions'] for r in rows)
    return data


def phase_statistics(rows):
    tail = measure([r for r in rows if r['phase_index'] >= PHASE_LENGTH - TAIL_LENGTH])
    gate = (tail['games'] == tail['first_opportunities'] == TAIL_LENGTH
        and 20 * tail['first_returns'] >= 19 * TAIL_LENGTH
        and tail['later_opportunities'] >= LATER_MINIMUM
        and 20 * tail['later_returns'] >= 19 * tail['later_opportunities'])
    recovery = next((measure(rows[:i]) for i, (a, b) in enumerate(zip(rows, rows[1:]), start=2)
                     if min(a['outcome']['returns'], b['outcome']['returns']) >= 5), None)
    return dict(full=measure(rows), first8=measure(rows[:8]), final20=tail,
                tail_gate=gate, recovery=recovery, recovery_censored=recovery is None)


def summarize(directory, schedules, errors, elapsed, unchanged):
    reports, checks, all_rows, unsaved = {}, {}, [], []
    acquired_path = directory / 'acquired.json'
    acquired = read(acquired_path) if acquired_path.exists() else None
    for name in ('live-0', 'live-1', 'frozen'):
        root = directory / name
        start_path = root / 'start.json'
        starting = read(start_path) if start_path.exists() else None
        rows = [read(path) for path in sorted((root / name).glob('game-*.json'),
                                             key=lambda path: int(path.stem.split('-')[-1]))]
        all_rows.extend(rows)
        phases = [p for p, _ in PHASES] if name != 'frozen' else ['cycle']
        expected = [a for a in schedules[int(name[-1]) if name != 'frozen' else 0]['games'] if a['phase'] in phases]
        chain = len(rows) == len(expected)
        previous = state_hash(starting) if starting else None
        for row, assignment in zip(rows, expected):
            after = state_hash(row['state'])
            chain &= all(row[k] == v for k, v in assignment.items()) and row['before'] == previous and row['after'] == after
            previous = after
            if name == 'live-0' and row['index'] == PHASE_LENGTH - 1:
                chain &= row['state'] == acquired
        if name == 'frozen':
            chain &= acquired is not None and starting == dict(acquired, updates_enabled=False)
            chain &= all(r['state'] == starting for r in rows)
        checks[name] = chain
        active_file = root / 'active.json'
        if active_file.exists():
            active = read(active_file)
            if not any(r['index'] == active['assignment']['index'] for r in rows):
                unsaved.append(active)
        reports[name] = {p: phase_statistics([r for r in rows if r['phase'] == p]) for p in phases}
    chosen, frozen = (reports[name]['cycle']['full'] for name in ('live-0', 'frozen'))
    margin = None
    if chosen['total_opportunities'] and frozen['total_opportunities']:
        margin = (Fraction(chosen['total_returns'], chosen['total_opportunities'])
                  - Fraction(frozen['total_returns'], frozen['total_opportunities']))
    complete = len(all_rows) == GAMES and all(checks.values()) and not errors and unchanged
    tails = all(r['tail_gate'] for name in ('live-0', 'live-1') for r in reports[name].values())
    margin_gate = margin is not None and margin >= Fraction(1, 5)
    costs = measure(all_rows)
    charged_transitions = costs['transitions'] + sum(a['camera_grant'] for a in unsaved)
    charged_initial = costs['initial_cameras'] + len(unsaved)
    assert charged_transitions <= 288000 and charged_initial <= 288
    return dict(complete=complete, passed=complete and tails and margin_gate, all_tail_gates=tails,
        matched_cycle_lift=float(margin) if margin is not None else None, matched_cycle_lift_passed=margin_gate,
        reports=reports, receipt_and_checkpoint_checks=checks, costs=costs, wall_seconds=elapsed, errors=errors,
        unchanged_source_assets_spec=unchanged, charged_transitions=charged_transitions, charged_initial=charged_initial)


def run(directory, digest):
    spec = verify(directory, digest)
    began = time.monotonic()
    deadline = began + spec['budget']['seconds']
    with (directory / 'work-clock.json').open('x') as clock:
        json.dump(dict(spec_sha256=digest, started_wall_ns=time.time_ns(), deadline_monotonic=deadline), clock)
    schedules = read(directory / 'schedules.json')
    acquired = directory / 'acquired.json'
    frozen_started = False
    errors = []
    with ProcessPoolExecutor(max_workers=spec['budget']['workers'], mp_context=mp.get_context('spawn')) as pool:
        jobs = {}
        for i, stream in enumerate(schedules):
            name = f'live-{i}'
            jobs[pool.submit(worker, directory / name, name, stream['games'],
                 VisualButtonEffects(stream['probe_seed']), deadline, acquired if i == 0 else None)] = name
        while jobs:
            if not frozen_started and acquired.exists() and time.monotonic() < deadline:
                groups = [r for r in schedules[0]['games'] if r['phase'] == 'cycle']
                control = VisualButtonEffects.from_state(read(acquired)).frozen_branch()
                jobs[pool.submit(worker, directory / 'frozen', 'frozen', groups, control, deadline)] = 'frozen'
                frozen_started = True
            done, _ = wait(jobs, timeout=.5, return_when=FIRST_COMPLETED)
            for future in done:
                name = jobs.pop(future)
                try:
                    future.result()
                except Exception as error:
                    errors.append(dict(arm=name, error=repr(error)))
    summary = summarize(directory, schedules, errors, time.monotonic() - began, verify(directory, digest) == spec)
    native.save(directory / 'summary.json', dict(spec_sha256=digest, **summary))
    return {k: summary[k] for k in ('complete', 'passed', 'matched_cycle_lift', 'costs', 'errors')}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('command', choices=('prepare', 'run'))
    cli.add_argument('directory', type=Path)
    cli.add_argument('--scout-directory', type=Path)
    cli.add_argument('--spec-sha256')
    args = cli.parse_args()
    if args.command == 'prepare':
        print(json.dumps(dict(spec_sha256=prepare(args.directory.resolve(), args.scout_directory.resolve()))))
    else:
        print(json.dumps(run(args.directory.resolve(), args.spec_sha256)))
