"""Bounded continuous response learning through an unannounced action cycle.

The three request-to-physical-action associations live only in execution. The
learner fits 96 continuous response taps from public visual displacements and
never receives a mapping, mapping identity or catalogue of alternatives.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

from pong_categorical_motor_fir import CategoricalMotorFIR, CategoricalMotorController, PROBE_HOLD_FRAMES
from pong_fixed_rbf import load_readout
from run_pong_carried_learning import policy_from_entry
from run_rally_option_memory import MODEL, sha, read, write, seed_rngs
from run_rally_repertoire_collection import exclusions, select_starts
import run_pong_visual_motor_learning as runner
import run_practical_pong_v2 as base

CONTRACT = 'continuous-three-request-visual-fir-cycle-learning-pilot-v1'
DEST = Path('/workspace/m1b-experiments/pong-categorical-motor-learning-20261105-r1')
PILOT = Path('/workspace/m1b-experiments/pong-visual-motor-learning-20261105-r2')


def install_execution(environment, settings):
    if set(settings) != {'action_map'} or sorted(settings['action_map']) != [0, 1, 2]:
        raise ValueError('a physical three-action permutation is required')
    mapping = tuple(settings['action_map'])
    original = environment.step
    physical_actions = []
    def execute(action):
        physical = mapping[action]
        physical_actions.append(physical)
        return original(physical)
    environment.step = execute
    return physical_actions


def configure_runtime():
    runner.VisualMotorFIR = CategoricalMotorFIR
    runner.VisualMotorController = CategoricalMotorController
    runner.install_intervention = install_execution
    runner.CONTRACT = CONTRACT


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve prospective specifications')
    inherited = read(PILOT / 'spec.json')
    destination.mkdir()
    used = exclusions(destination)
    schedules = {}
    index = 0
    for tag, phase, count, mapping in ((63631, 'A_acquire', 8, [0, 1, 2]),
            (63632, 'B_cycle', 16, [1, 2, 0]), (63633, 'A_return', 8, [0, 1, 2])):
        drawn = select_starts(1943906127, tag, used, count // 2)
        used.update(seed % base.MODULUS for seed in drawn['TRAIN'])
        schedules[phase] = [dict(index=index + i, phase=phase, phase_index=i,
            seed=seed, intervention=dict(action_map=mapping)) for i, seed in enumerate(drawn['TRAIN'])]
        index += count
    write(destination / 'schedules.json', schedules)
    files = dict(inherited['files'])
    files.update({str(p): sha(p) for p in (Path(__file__),
        Path(__file__).with_name('pong_categorical_motor_fir.py'),
        destination / 'schedules.json', destination / 'seed-exclusion-proof.json')})
    spec = dict(contract=CONTRACT, source=base.source_signature(), files=files,
        inherited=inherited['inherited'], settings=inherited['settings'],
        random_seed=2026110509, seconds=600, workers=2,
        live_games=32, frozen_acquired_A_games=16, total_games=48,
        maximum_transition_cameras=48000, numerical_coefficients=96,
        feature_prior='3 request categories times32lags, no intercept; unissued history allzero',
        controller_prior='inherited anonymous visual relation,16step learned-model forecast,3constant request candidates',
        exploration='random4request holds; initial10percent / continuing1percent probe-start probability; unsupported scene cancels hold',
        probe_hold_frames=PROBE_HOLD_FRAMES, no_knowledge_or_rng_reset=True,
        action_prior='existing tiny request-index1 command penalty retained regardless of physical remapping',
        intervention='execution-only cycle request0→physical1,1→2,2→0; zero added latency',
        unchanged_control_gate='both current controlled/target coordinates actually observed; inherited persistent servo otherwise',
        controls='frozen acquired A response coefficients/covariance, ordinary runtime/history/probes continue on matched Bstarts',
        known_limit='separate harder synthetic added6frame-delay plus cycle deadlocks under inherited fallback/response gate; not claimed supported',
        no_backprop=True, no_privileged_inputs=True, no_expert_labels=True,
        no_mapping_catalogue=True, no_parameter_grid=True, no_automatic_extension=True,
        full_public_replay=True, purpose='development acquisition/adaptation pilot, not qualification')
    write(destination / 'spec.json', spec)
    return dict(directory=str(destination), spec_sha256=sha(destination / 'spec.json'),
        total_games=48, maximum_transition_cameras=48000, native_started=False)


def verify(destination, approved):
    spec = read(destination / 'spec.json')
    assert sha(destination / 'spec.json') == approved
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    return spec


def frozen_worker(destination, spec, state, assignments, deadline):
    configure_runtime()
    return runner.frozen_stream(destination, spec, state, assignments, deadline)


def run(destination, approved):
    destination = Path(destination)
    spec = verify(destination, approved)
    if (destination / 'work-clock.json').exists():
        raise FileExistsError('never retry or extend a started chronological allocation')
    began = time.monotonic()
    deadline = began + spec['seconds']
    write(destination / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        began_monotonic=began, deadline_monotonic=deadline, approved_spec=approved))
    configure_runtime()
    base.initialize()
    seed_rngs()
    learner = CategoricalMotorFIR(spec['random_seed'], settings=spec['settings'])
    shadow = CategoricalMotorFIR.from_state(learner.state())
    baseline = load_readout(MODEL / 'rbf-model.npz')
    policy = policy_from_entry(spec['inherited'])
    schedules = read(destination / 'schedules.json')
    rows, frozen_rows, errors = [], [], []
    def play(assignment):
        nonlocal learner
        try:
            row = runner.evaluate_game(baseline, policy, learner, shadow, assignment,
                destination / 'live', deadline)
        except Exception as error:
            errors.append(dict(stream='live', assignment=assignment, error=repr(error),
                unknown_unsaved_native_cost=True))
            return False
        rows.append(row)
        write(destination / 'checkpoint-latest.json', learner.state())
        write(destination / 'progress.json', dict(phase=assignment['phase'],
            phase_index=assignment['phase_index'], **runner.summarize(rows)))
        if not row['complete']:
            return False
        learner = CategoricalMotorFIR.from_state(read(destination / 'checkpoint-latest.json'))
        return True
    for assignment in schedules['A_acquire']:
        if time.monotonic() >= deadline or not play(assignment):
            break
    write(destination / 'checkpoint-end-A.json', learner.state())
    ready = (len(rows) == 8 and all(r['complete'] for r in rows)
        and learner.updates >= spec['settings']['initial_dither_updates'])
    if len(rows) == 8 and not ready:
        errors.append(dict(stream='controls', error='incomplete A or insufficient updates to match probe clock'))
    if ready:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                initializer=base.initialize) as pool:
            frozen = pool.submit(frozen_worker, str(destination), spec, learner.state(),
                schedules['B_cycle'], deadline)
            for assignment in schedules['B_cycle'] + schedules['A_return']:
                if time.monotonic() >= deadline or not play(assignment):
                    break
            try:
                frozen_rows = frozen.result()
            except Exception as error:
                errors.append(dict(stream='frozen', error=repr(error), unknown_unsaved_native_cost=True))
                frozen_rows = [read(p) for p in sorted((destination / 'frozen-acquired-A').glob('game-*.json'))]
    # Any saved receipt preceding a replay/return failure must still be charged.
    saved = {r['assignment']['index']: r for r in rows}
    for p in sorted((destination / 'live').glob('game-*.json')):
        row = read(p)
        saved[row['assignment']['index']] = row
    rows = [saved[i] for i in sorted(saved)]
    all_rows = rows + frozen_rows
    result = dict(contract=CONTRACT, complete=len(rows) == 32 and len(frozen_rows) == 16
        and all(r['complete'] for r in all_rows) and not errors,
        live={phase: runner.summarize([r for r in rows if r['assignment']['phase'] == phase])
            for phase in schedules}, frozen_acquired_A_B=runner.summarize(frozen_rows), errors=errors,
        transitions=sum(r['transitions'] for r in all_rows),
        initial_cameras=sum(r['initial_cameras'] for r in all_rows),
        known_transition_camera_upper_bound=sum(r['transition_camera_upper_bound'] for r in all_rows),
        known_initial_camera_upper_bound=sum(r['initial_camera_upper_bound'] for r in all_rows),
        unknown_unsaved_failed_native_cost=any(e.get('unknown_unsaved_native_cost') for e in errors),
        maximum_total_transition_allocation=spec['maximum_transition_cameras'],
        frozen_control_probe_probability_matched=ready,
        all_full_public_replays_passed=all(r.get('full_public_replay', {}).get('passed') for r in all_rows),
        source_assets_spec_frozen=verify(destination, approved) == spec,
        wall_seconds=time.monotonic() - began, pilot_only=True, no_completion_claim=True)
    assert result['known_transition_camera_upper_bound'] <= spec['maximum_transition_cameras']
    write(destination / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--directory', type=Path, default=DEST)
    parser.add_argument('--approved-spec')
    args = parser.parse_args()
    output = prepare(args.directory) if args.command == 'prepare' else run(args.directory, args.approved_spec)
    print(json.dumps(output, sort_keys=True), flush=True)
