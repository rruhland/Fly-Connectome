"""One bounded gain/intercept challenge after weak delay-specific learning gap.

All four chronological streams branch the same actually acquired A checkpoint.
Only execution wrappers differ; no setting or stream identity enters a learner.
The pinned FIR/controller and previous pilot sources remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from pong_fixed_rbf import load_readout
from pong_visual_motor_fir import VisualMotorFIR
from pong_numeric_execution_changes import install_intervention
from run_pong_carried_learning import policy_from_entry
from run_rally_option_memory import MODEL, sha, read, write, seed_rngs
from run_rally_repertoire_collection import select_starts, exclusions
import run_pong_visual_motor_learning as runner
import run_practical_pong_v2 as base

CONTRACT = 'public-fir-gain-intercept-adjustment-scout-v1'
PILOT = Path('/workspace/m1b-experiments/pong-visual-motor-learning-20261105-r2')
PILOT_SHA = 'ce37f57612ea9a462501ede46a566fb35f9d1bf1197f1b2da301e2a26d8a6e75'
DEST = Path('/workspace/m1b-experiments/pong-numeric-adjustment-scout-20261105-r1')
CHANGES = dict(gain2=dict(delay_commands=0, drive_bias=0., drive_gain=2.),
               affine140_040=dict(delay_commands=0, drive_bias=.4, drive_gain=1.4))


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve prior prospective specifications')
    pilot = runner.verify(PILOT, PILOT_SHA)
    report = read(PILOT / 'summary.json')
    assert report['complete'] and not report['errors']
    acquired = read(PILOT / 'checkpoint-end-A.json')
    assert acquired['updates'] >= acquired['settings']['initial_dither_updates']
    destination.mkdir()
    used = exclusions(destination)
    schedules = {}
    for tag, (change, intervention) in enumerate(CHANGES.items(), start=63621):
        drawn = select_starts(1943906127, tag, used, 4)
        used.update(seed % base.MODULUS for seed in drawn['TRAIN'])
        schedules[change] = [dict(index=i, phase=change, phase_index=i, seed=seed,
            intervention=intervention) for i, seed in enumerate(drawn['TRAIN'])]
    write(destination / 'schedules.json', schedules)
    write(destination / 'acquired-A.json', acquired)
    paths = [Path(__file__), Path(__file__).with_name('pong_numeric_execution_changes.py'),
        Path(runner.__file__), PILOT / 'checkpoint-end-A.json', PILOT / 'summary.json',
        destination / 'schedules.json', destination / 'acquired-A.json',
        destination / 'seed-exclusion-proof.json']
    files = dict(pilot['files'])
    files.update({str(p): sha(p) for p in paths})
    spec = dict(contract=CONTRACT, source=base.source_signature(), files=files,
        inherited=pilot['inherited'], settings=pilot['settings'], changes=CHANGES,
        pilot_directory=str(PILOT), pilot_spec_sha256=PILOT_SHA,
        acquired_A_sha256=sha(destination / 'acquired-A.json'),
        streams=[dict(change=change, kind=kind) for change in CHANGES for kind in ('live', 'frozen')],
        games_per_stream=8, total_games=32, maximum_transition_cameras=32000,
        seconds=600, workers=4, matched_starts=True,
        learned_model_unchanged=True, live_frozen_same_initial_model_and_rng=True,
        frozen='coefficients/covariance disabled; ordinary history/observations/1percent dither continue',
        intervention_inputs_excluded=True, no_backprop=True, no_privileged_inputs=True,
        no_expert_labels=True, no_parameter_grid=True, no_automatic_extension=True,
        purpose='two mechanistically distinct numerical changes; development attribution, not qualification')
    write(destination / 'spec.json', spec)
    return dict(directory=str(destination), spec_sha256=sha(destination / 'spec.json'),
        changes=CHANGES, games=32, maximum_transition_cameras=32000, native_started=False)


def verify(destination, approved):
    spec = read(destination / 'spec.json')
    assert sha(destination / 'spec.json') == approved
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    return spec


def stream(destination, spec, assignment, deadline):
    base.initialize()
    seed_rngs()
    directory = Path(destination)
    name = assignment['change'] + '-' + assignment['kind']
    folder = directory / name
    folder.mkdir()
    learner = VisualMotorFIR.from_state(read(directory / 'acquired-A.json'))
    if assignment['kind'] == 'frozen':
        learner = learner.frozen_branch()
    shadow = VisualMotorFIR.from_state(learner.state())
    original_theta = learner.theta.copy()
    original_covariance = learner.covariance.copy()
    baseline = load_readout(MODEL / 'rbf-model.npz')
    policy = policy_from_entry(spec['inherited'])
    schedules = read(directory / 'schedules.json')[assignment['change']]
    # This new process changes only the execution-boundary function used by the
    # unchanged evaluator. It never modifies an old source file or live pilot.
    runner.install_intervention = install_intervention
    rows = []
    for game in schedules:
        if time.monotonic() >= deadline or (directory / 'stop-on-error.json').exists():
            break
        try:
            row = runner.evaluate_game(baseline, policy, learner, shadow, game, folder, deadline)
        except Exception as error:
            write(directory / 'stop-on-error.json', dict(stream=name, game=game,
                error=repr(error), unknown_unsaved_native_cost=True))
            raise
        rows.append(row)
        write(folder / 'checkpoint-latest.json', learner.state())
        write(folder / 'progress.json', runner.summarize(rows))
        if assignment['kind'] == 'frozen':
            assert np.array_equal(learner.theta, original_theta)
            assert np.array_equal(learner.covariance, original_covariance)
        if not row['complete']:
            if row.get('error'):
                write(directory / 'stop-on-error.json', dict(stream=name, game=game,
                    error=row['error'], unknown_unsaved_native_cost=False))
            break
        learner = VisualMotorFIR.from_state(read(folder / 'checkpoint-latest.json'))
    result = runner.summarize(rows)
    result.update(complete=len(rows) == len(schedules) and all(r['complete'] for r in rows),
        final_learner_sha256=runner.state_hash(learner.state()))
    write(folder / 'summary.json', result)
    return result


def run(destination, approved):
    destination = Path(destination)
    spec = verify(destination, approved)
    if (destination / 'work-clock.json').exists():
        raise FileExistsError('never retry or extend a started allocation')
    began = time.monotonic()
    deadline = began + spec['seconds']
    write(destination / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        began_monotonic=began, deadline_monotonic=deadline, approved_spec=approved))
    errors = []
    with ProcessPoolExecutor(max_workers=spec['workers'],
            mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        futures = [(assignment, pool.submit(stream, str(destination), spec, assignment, deadline))
            for assignment in spec['streams']]
        for assignment, future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(dict(stream=assignment, error=repr(error), unknown_unsaved_native_cost=True))
    # Receipts survive worker failure; always charge all known saved work.
    results = {}
    all_rows = []
    for assignment in spec['streams']:
        name = assignment['change'] + '-' + assignment['kind']
        rows = [read(p) for p in sorted((destination / name).glob('game-*.json'))]
        all_rows.extend(rows)
        result = runner.summarize(rows)
        result['complete'] = len(rows) == 8 and all(r['complete'] for r in rows)
        results[name] = result
    summary = dict(contract=CONTRACT, complete=all(r['complete'] for r in results.values()) and not errors,
        streams=results, errors=errors, transitions=sum(r['transitions'] for r in all_rows),
        initial_cameras=sum(r['initial_cameras'] for r in all_rows),
        known_transition_camera_upper_bound=sum(r['transition_camera_upper_bound'] for r in all_rows),
        known_initial_camera_upper_bound=sum(r['initial_camera_upper_bound'] for r in all_rows),
        unknown_unsaved_failed_native_cost=bool(errors),
        maximum_total_transition_allocation=spec['maximum_transition_cameras'],
        source_assets_spec_frozen=verify(destination, approved) == spec,
        all_full_public_replays_passed=all(r.get('full_public_replay', {}).get('passed') for r in all_rows),
        wall_seconds=time.monotonic() - began, pilot_only=True, no_completion_claim=True)
    assert summary['known_transition_camera_upper_bound'] <= spec['maximum_transition_cameras']
    write(destination / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--directory', type=Path, default=DEST)
    parser.add_argument('--approved-spec')
    args = parser.parse_args()
    output = prepare(args.directory) if args.command == 'prepare' else run(args.directory, args.approved_spec)
    print(json.dumps(output, sort_keys=True), flush=True)
