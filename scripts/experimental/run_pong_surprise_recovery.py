"""Fixed paired native development comparison of public residual-triggered recovery."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from fly_connectome.control.session import MODEL_HASH, model_bytes, source_signature
from pong_categorical_motor_fir import CategoricalMotorFIR
from pong_categorical_surprise_recovery import SurpriseRecoveryFIR
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from run_pong_inverse_cycle import play, save, sha


ROOT = Path(__file__).resolve().parents[2]
MODULUS = 2_147_483_647
START_SEED = 90722311
PROBE_SEEDS = (210459738, 751228426)
PHASES = (
    ('normal', 8, (0, 1, 2)),
    ('cycle', 16, (1, 2, 0)),
    ('inverse', 16, (2, 0, 1)),
    ('restored', 16, (0, 1, 2)),
)
ARMS = {'original': CategoricalMotorFIR, 'adaptive': SurpriseRecoveryFIR}
FILES = (
    'run_pong_surprise_recovery.py', 'run_pong_inverse_cycle.py',
    'pong_categorical_surprise_recovery.py', 'pong_categorical_motor_fir.py',
    'pong_visual_motor_fir.py', 'pong_relational_feedback.py',
    'pong_relational_persistent_binding.py', 'public_entity_pong_native.py', 'return_metrics.py',
)
POLICY = dict(choices=[3, 0, 1, 0, -1], coefficients=[0.] * 5,
              scales=[1.] * 4, require_first_hit=False, unsupported_fallback=1)
SECONDS = 1800
WORKERS = 2


def read(path):
    return json.loads(path.read_text())


def runtime_dependencies():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')}


def exclusions(experiment_root):
    metadata = {}
    used = set(PROBE_SEEDS) | {START_SEED, 1939032277}

    def collect(value):
        if type(value) is int and 0 < value < 2 ** 32:
            used.add(value % MODULUS)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    patterns = ('schedules.json', 'schedule.json', 'plan.json', 'seed-exclusion-proof.json')
    for pattern in patterns:
        for path in sorted(experiment_root.rglob(pattern)):
            collect(read(path))
            metadata[str(path)] = sha(path)
    return used, metadata


def prepare(directory):
    model_bytes()
    used, metadata = exclusions(directory.parent)
    schedules = []
    for replica, probe_seed in enumerate(PROBE_SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([START_SEED, replica]))
        rows = []
        for phase, games, mapping in PHASES:
            for local_index in range(games):
                while True:
                    seed = int(rng.integers(1, MODULUS))
                    direction = ((seed * 48271 + 1) % MODULUS) % 2
                    if seed not in used and direction == local_index % 2:
                        break
                used.add(seed)
                rows.append(dict(index=len(rows), phase=phase, phase_index=local_index,
                                 seed=seed, mapping=list(mapping)))
        schedules.append(dict(replica=replica, probe_seed=probe_seed, games=rows))
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'schedules.json', schedules)
    spec = dict(contract='paired-native-visual-surprise-recovery-development-v1',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        package_source=source_signature(), dependencies=runtime_dependencies(),
        experimental_source={name: sha(Path(__file__).with_name(name)) for name in FILES},
        vision_asset_sha256=MODEL_HASH, schedule_sha256=sha(directory / 'schedules.json'),
        prior_seed_metadata=metadata, seed_exclusion_scope='available prior schedules and exclusion proofs',
        start_seed=START_SEED, probe_seeds=PROBE_SEEDS, phases=PHASES, inherited_policy=POLICY,
        same_asset_and_initial_probe_seed_within_pair=True, hidden_execution_maps=True,
        play='native public entity path with neutral request fallback on unsupported scenes',
        budget=dict(seconds=SECONDS, workers=WORKERS, games=224, transitions=224000, initial_cameras=224),
        gates=dict(early='strictly fewer own misses over the first eight games after all three switches',
            sustained='at least 95% first and later separately in each phase final eight pooled across streams after each switch, with all 16 first opportunities and nonzero later exposure',
            acquisition='adaptive initial normal eight have no more aggregate own misses than original'),
        recovery='first two consecutive completed games with at least five returns each, cost through second game; never recovered is reported as censored',
        development_only=True, extension=False)
    path = directory / 'spec.json'
    save(path, spec)
    return sha(path)


def verify(directory, digest):
    path = directory / 'spec.json'
    assert sha(path) == digest
    spec = read(path)
    model_bytes()
    assert spec['vision_asset_sha256'] == MODEL_HASH
    assert spec['schedule_sha256'] == sha(directory / 'schedules.json')
    assert spec['package_source'] == source_signature()
    assert spec['dependencies'] == runtime_dependencies()
    assert all(sha(Path(__file__).with_name(name)) == expected
               for name, expected in spec['experimental_source'].items())
    return spec


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reached_deadline(_number, _frame):
    raise TimeoutError('native development wall allocation reached')


def play_pair(directory, stream, deadline):
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, reached_deadline)
    replica = stream['replica']
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    learners = {arm: cls(stream['probe_seed']) for arm, cls in ARMS.items()}
    for arm, learner in learners.items():
        save(directory / f'{arm}-r{replica}-start.json', learner.state())
    rows = []
    active = directory / f'r{replica}-active.json'
    for assignment in stream['games']:
        # Alternate order so neither arm consistently runs first within the pair.
        arms = tuple(learners)
        if assignment['index'] % 2:
            arms = arms[::-1]
        for arm in arms:
            if time.monotonic() >= deadline:
                return
            learner = learners[arm]
            save(active, dict(replica=replica, arm=arm, assignment=assignment,
                              transition_grant=1000, initial_grant=1))
            before = state_hash(learner.state())
            probes = learner.dither_count
            surprises = getattr(learner, 'surprise_episodes', 0)
            started = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - started))
            try:
                result = play(learner, policy, assignment['seed'], assignment['mapping'])
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            assert result['transitions'] <= 1000 and result['initial'] == 1
            after = learner.state()
            row = dict(**assignment, arm=arm, replica=replica, outcome=result,
                elapsed_wall_seconds=time.monotonic() - started, before=before, after=state_hash(after),
                probe_frames=learner.dither_count - probes,
                surprise_episodes=getattr(learner, 'surprise_episodes', 0) - surprises)
            rows.append(row)
            save(directory / f'r{replica}-games.json', rows)
            save(directory / f'{arm}-r{replica}-latest.json', after)
            active.unlink()
        phase_index = assignment['phase_index']
        if phase_index + 1 == dict((phase, games) for phase, games, _ in PHASES)[assignment['phase']]:
            print(f"replica {replica} completed {assignment['phase']}", flush=True)


def score(rows):
    outcomes = [row['outcome'] for row in rows]
    total_returns = sum(x['returns'] for x in outcomes)
    total_opportunities = sum(x['opportunities'] for x in outcomes)
    first_returns = sum(bool(x['events'] and x['events'][0]['hit']) for x in outcomes)
    first_opportunities = sum(bool(x['events']) for x in outcomes)
    later_returns = total_returns - first_returns
    later_opportunities = total_opportunities - first_opportunities
    return dict(games=len(rows), first_returns=first_returns, first_opportunities=first_opportunities,
        first_rate=first_returns / first_opportunities if first_opportunities else None,
        later_returns=later_returns, later_opportunities=later_opportunities,
        later_rate=later_returns / later_opportunities if later_opportunities else None,
        first_misses=first_opportunities - first_returns, later_misses=later_opportunities - later_returns,
        misses=total_opportunities - total_returns, total_returns=total_returns,
        total_opportunities=total_opportunities, physics_seconds=sum(x['seconds'] for x in outcomes),
        transitions=sum(x['transitions'] for x in outcomes), initial_cameras=sum(x['initial'] for x in outcomes),
        probe_frames=sum(x['probe_frames'] for x in rows), surprise_episodes=sum(x['surprise_episodes'] for x in rows))


def phase_report(rows, expected_games):
    recovery = None
    for index, (one, two) in enumerate(zip(rows, rows[1:])):
        if min(one['outcome']['returns'], two['outcome']['returns']) >= 5:
            recovery = score(rows[:index + 2])
            break
    return dict(full=score(rows), first4=score(rows[:4]), first8=score(rows[:8]),
        final8=score([row for row in rows if row['phase_index'] >= expected_games - 8]),
        two_successive_five_return_games=recovery,
        recovery_censored=recovery is None)


def report(directory, spec, errors, elapsed, unchanged):
    schedules = read(directory / 'schedules.json')
    rows = []
    paired = True
    continuity = True
    for stream in schedules:
        replica = stream['replica']
        path = directory / f'r{replica}-games.json'
        games = read(path) if path.exists() else []
        rows.extend(games)
        for arm in ARMS:
            arm_rows = [row for row in games if row['arm'] == arm]
            assignments = [{k: r[k] for k in ('index', 'seed', 'phase', 'phase_index', 'mapping')}
                           for r in arm_rows]
            paired &= assignments == stream['games']
            start = directory / f'{arm}-r{replica}-start.json'
            before = state_hash(read(start)) if start.exists() else None
            for row in arm_rows:
                continuity &= row['before'] == before
                before = row['after']
            latest = directory / f'{arm}-r{replica}-latest.json'
            continuity &= latest.exists() and state_hash(read(latest)) == before

    by_replica = {}
    pooled = {}
    for arm in ARMS:
        by_replica[arm] = {str(stream['replica']): {phase: phase_report([r for r in rows
            if r['arm'] == arm and r['replica'] == stream['replica'] and r['phase'] == phase], games)
            for phase, games, _ in PHASES} for stream in schedules}
        pooled[arm] = {}
        for phase, games, _ in PHASES:
            full = [r for r in rows if r['arm'] == arm and r['phase'] == phase]
            pooled[arm][phase] = dict(full=score(full), first4=score([r for r in full if r['phase_index'] < 4]),
                first8=score([r for r in full if r['phase_index'] < 8]),
                final8=score([r for r in full if r['phase_index'] >= games - 8]))
    changed_phases = [phase for phase, _, _ in PHASES[1:]]
    early_misses = {arm: sum(pooled[arm][phase]['first8']['misses'] for phase in changed_phases) for arm in ARMS}
    late_gates = {}
    for phase in changed_phases:
        tail = pooled['adaptive'][phase]['final8']
        late_gates[phase] = (tail['first_opportunities'] == 16 and tail['later_opportunities'] > 0
            and Fraction(tail['first_returns'], tail['first_opportunities']) >= Fraction(19, 20)
            and Fraction(tail['later_returns'], tail['later_opportunities']) >= Fraction(19, 20))
    initial_gate = pooled['adaptive']['normal']['full']['misses'] <= pooled['original']['normal']['full']['misses']
    complete = len(rows) == spec['budget']['games'] and paired and continuity and unchanged and not errors
    costs = score(rows)
    outstanding = [read(path) for path in directory.glob('r*-active.json')]
    unsaved = [active for active in outstanding if not any(
        r['arm'] == active['arm'] and r['replica'] == active['replica']
        and r['index'] == active['assignment']['index'] for r in rows)]
    charged_transitions = costs['transitions'] + sum(active['transition_grant'] for active in unsaved)
    charged_initial_cameras = costs['initial_cameras'] + sum(active['initial_grant'] for active in unsaved)
    assert charged_transitions <= spec['budget']['transitions']
    assert charged_initial_cameras <= spec['budget']['initial_cameras']
    return dict(complete=complete, matched_starts=paired, checkpoint_continuity=continuity,
        unchanged_source_and_assets=unchanged, early_misses=early_misses, late_gates=late_gates,
        cold_normal_gate=initial_gate, development_gate_passed=bool(complete and all(late_gates.values())
            and initial_gate and early_misses['adaptive'] < early_misses['original']),
        by_replica=by_replica, pooled=pooled, errors=errors, wall_seconds=elapsed,
        games=len(rows), transitions=costs['transitions'], initial_cameras=costs['initial_cameras'],
        charged_transitions=charged_transitions, charged_initial_cameras=charged_initial_cameras,
        outstanding=outstanding)


def run(directory, digest):
    spec = verify(directory, digest)
    start = time.monotonic()
    deadline = start + spec['budget']['seconds']
    with (directory / 'work-clock.json').open('x') as stream:
        json.dump(dict(spec_sha256=digest, started_wall_ns=time.time_ns(), deadline_monotonic=deadline), stream)
    errors = []
    schedules = read(directory / 'schedules.json')
    with ProcessPoolExecutor(max_workers=spec['budget']['workers'], mp_context=mp.get_context('spawn')) as pool:
        futures = {stream['replica']: pool.submit(play_pair, directory, stream, deadline) for stream in schedules}
        for replica, future in futures.items():
            try:
                future.result()
            except Exception as error:
                errors.append(dict(replica=replica, error=repr(error)))
    unchanged = verify(directory, digest) == spec
    summary = dict(spec_sha256=digest, **report(directory, spec, errors, time.monotonic() - start, unchanged))
    save(directory / 'summary.json', summary)
    return {k: summary[k] for k in ('complete', 'development_gate_passed', 'games', 'transitions', 'errors')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--spec-sha256')
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = dict(spec_sha256=prepare(directory)) if args.command == 'prepare' else run(directory, args.spec_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
