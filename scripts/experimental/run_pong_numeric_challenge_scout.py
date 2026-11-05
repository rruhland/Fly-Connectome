"""Twelve prospectively paired games to choose a numerical-learning challenge.

Execution changes stay wholly outside the frozen public visual controller. This
diagnostic does no learning and is not qualification or a parameter search.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import compact_public_pong as compact
from pong_relational_feedback import load_calibration_scales, RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from return_metrics import aggregate
from run_pong_relational_categorical_replication import CALIBRATION, make_roster
from run_rally_option_memory import MODEL, EXPECTED, sha, read, write
from run_rally_repertoire_collection import exclusions, select_starts
import run_practical_pong_v2 as base

CONTRACT = 'numeric-learning-challenge-scout-v1'
SETTINGS = {'original': {'delay_commands': 0, 'drive_bias': 0.},
            'delay8': {'delay_commands': 8, 'drive_bias': 0.},
            'drift060': {'delay_commands': 0, 'drive_bias': .60}}
DEST = Path('/workspace/m1b-experiments/pong-numeric-challenge-20261105-r1')


def install_intervention(environment, settings):
    """Policy sees requested commands and ordinary public camera feedback only."""
    queue = deque([1] * settings['delay_commands'])
    original = environment.step
    physical_actions = []
    def execute(action):
        queue.append(action)
        physical = queue.popleft()
        physical_actions.append(physical)
        return original(physical)
    environment.step = execute
    if settings['drive_bias']:
        physical_step = environment.game.step
        def biased(drive, **kwargs):
            return physical_step(drive + settings['drive_bias'], **kwargs)
        environment.game.step = biased
    return physical_actions


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve previous specifications and native work')
    destination.mkdir()
    schedule = select_starts(1943906127, 63601, exclusions(destination), 2)
    entry = make_roster(load_calibration_scales(CALIBRATION))[4]
    policy = PersistentRelationalPolicy(RelationalPolicy(*entry['choices'],
        entry['coefficients'], entry['scales']), require_first_hit=False)
    policy.save(destination / 'inherited-structure04.npz')
    write(destination / 'schedules.json', schedule)
    names = set(compact.FILES) | {Path(__file__).name,
        'run_pong_relational_categorical_replication.py'}
    paths = [Path(__file__).with_name(name) for name in sorted(names)]
    paths += [MODEL / 'rbf-model.npz', MODEL / 'radial-cap.json', CALIBRATION,
              destination / 'inherited-structure04.npz', destination / 'schedules.json',
              destination / 'seed-exclusion-proof.json']
    spec = dict(contract=CONTRACT, settings=SETTINGS, seeds=schedule['TRAIN'],
        source=base.source_signature(), files={str(p): sha(p) for p in paths},
        baseline_sha256=EXPECTED, workers=4, seconds=240, games=12,
        transition_cap_per_game=1000, transition_cap_total=12000,
        max_episode_ticks=2400, opponent_speed=.65, no_learning=True,
        no_parameter_grid=True, no_extension=True, matched_exogenous_starts=True,
        purpose='development challenge selection; never untouched qualification')
    write(destination / 'spec.json', spec)
    return dict(directory=str(destination), spec_sha256=sha(destination / 'spec.json'),
                seeds=spec['seeds'], settings=SETTINGS, native_started=False)


def verify(destination, approved):
    spec = read(destination / 'spec.json')
    assert sha(destination / 'spec.json') == approved
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    return spec


def task(destination, spec, kind, index, seed, deadline):
    original_class = compact.CompactEvaluator
    class ChallengedEvaluator(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.physical_actions = install_intervention(self.env, spec['settings'][kind])
    compact.CompactEvaluator = ChallengedEvaluator
    try:
        result = compact.evaluate(str(Path(destination) / 'inherited-structure04.npz'),
            seed, spec['max_episode_ticks'], Path(destination) / kind,
            deadline, spec['transition_cap_per_game'], CONTRACT + ':' + kind)
    finally:
        compact.CompactEvaluator = original_class
    result.update(kind=kind, index=index, settings=spec['settings'][kind])
    write(Path(destination) / kind / f'game-{index:02d}.json', result)
    return result


def run(destination, approved):
    destination = Path(destination)
    spec = verify(destination, approved)
    if (destination / 'work-clock.json').exists():
        raise FileExistsError('a started allocation cannot be retried or extended')
    began = time.monotonic()
    deadline = began + spec['seconds']
    write(destination / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        started_monotonic=began, deadline_monotonic=deadline, approved_spec=approved))
    jobs = [(kind, i, seed) for i, seed in enumerate(spec['seeds']) for kind in SETTINGS]
    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=spec['workers'],
            mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        futures = [pool.submit(task, str(destination), spec, *job, deadline) for job in jobs]
        for job, future in zip(jobs, futures):
            try:
                rows.append(future.result())
            except Exception as error:
                errors.append(dict(job=job, error=repr(error)))
            write(destination / 'progress.json', dict(saved_games=len(rows),
                complete_games=sum(r['complete'] for r in rows), errors=errors,
                transitions=sum(r['transitions'] for r in rows)))
    summary = dict(contract=CONTRACT, complete=len(rows) == 12
        and all(r['complete'] for r in rows) and not errors,
        groups={kind: aggregate([r['annotation'] for r in rows
            if r['kind'] == kind and r['complete']]) for kind in SETTINGS},
        completed_games={kind: sum(r['kind'] == kind and r['complete'] for r in rows)
            for kind in SETTINGS}, errors=errors,
        transitions=sum(r['transitions'] for r in rows),
        initial_cameras=sum(r['initial_cameras'] for r in rows),
        transition_camera_upper_bound=sum(r['transition_camera_upper_bound'] for r in rows),
        source_assets_spec_frozen=verify(destination, approved) == spec,
        all_streaming_audits_passed=all(r['streaming_audit']['passed'] for r in rows),
        wall_seconds=time.monotonic() - began, no_learning=True)
    assert summary['transition_camera_upper_bound'] <= spec['transition_cap_total']
    write(destination / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--directory', type=Path, default=DEST)
    parser.add_argument('--approved-spec')
    args = parser.parse_args()
    result = prepare(args.directory) if args.command == 'prepare' else run(args.directory, args.approved_spec)
    print(json.dumps(result, sort_keys=True), flush=True)
