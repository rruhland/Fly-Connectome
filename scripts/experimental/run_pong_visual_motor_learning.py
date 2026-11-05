"""Bounded chronological within-game numeric motor-response acquisition.

The learner sees only public visual entities, requested command history, actual
elapsed time and public outcomes. All intervention and return-counter state is
kept outside its interface. Archived public inputs support exact offline replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
import compact_public_pong as compact
from exact_visual_transport_copy import install_exact_transport_copy
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_visual_motor_fir import VisualMotorFIR, VisualMotorController, DEFAULTS
from public_entity_pong_native import snapshot, save_public_entities, load_public_entities
from return_metrics import ReturnCounter, aggregate, native_count_status
from run_pong_numeric_challenge_scout import install_intervention
from run_pong_carried_learning import policy_from_entry
from run_pong_relational_categorical_replication import make_roster, CALIBRATION
from pong_relational_feedback import load_calibration_scales
from run_rally_option_memory import MODEL, EXPECTED, sha, read, write, seed_rngs
from run_rally_repertoire_collection import select_starts, exclusions
import run_practical_pong_v2 as base

CONTRACT = 'within-game-public-visual-fir-motor-learning-pilot-v1'
DEST = Path('/workspace/m1b-experiments/pong-visual-motor-learning-20261105-r2')


def state_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def public_feedback(row):
    return dict(own_hits=int(row['hits']), reward=float(row['reward']),
        terminated=bool(row['terminated']), truncated=bool(row['truncated']),
        elapsed_seconds=float(row['elapsed_seconds']))


def digest_row(digest, baseline, action, decision, feedback, update):
    payload = dict(baseline=int(baseline), action=int(action), decision=decision,
                   feedback=feedback, update=update)
    digest.update(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


class NumericEvaluator(FreshRBFEvaluator):
    def __init__(self, baseline, policy, learner, shadow, config):
        self.inherited_policy = policy
        self.learner, self.shadow_learner = learner, shadow
        self.controller = VisualMotorController(policy, learner)
        self.shadow_controller = VisualMotorController(policy, shadow)
        self.public_inputs = []
        self.records = []
        self.baseline_actions = []
        self.digest = hashlib.sha256()
        self.pending = None
        super().__init__(baseline, config)
        self.initial_learned_hash = self.learned_hash()

    def learned_hash(self):
        return fingerprint((super().learned_hash(), CONTRACT,
            self.inherited_policy.frozen_parameters(), self.learner.settings))

    def _begin_episode(self):
        super()._begin_episode()
        self.public_inputs.append(snapshot(self.latest_encoded['entities']))

    def _choose_policy(self):
        baseline, q = self.rbf_readout.choose(*policy_features(self.latest_encoded, self.scaler))
        entities = snapshot(self.latest_encoded['entities'])
        action, decision = self.controller.choose(entities, int(baseline))
        other, other_decision = self.shadow_controller.choose(snapshot(entities), int(baseline))
        if action != other or decision != other_decision:
            raise AssertionError('independently restored numeric controllers disagree')
        self.pending = (int(baseline), int(action), decision)
        self.baseline_actions.append(int(baseline))
        return int(action), q

    def step(self, **kwargs):
        row = super().step(**kwargs)
        baseline, action, decision = self.pending
        assert action == row['action']
        entities = snapshot(self.latest_encoded['entities'])
        feedback = public_feedback(row)
        update = self.controller.observe(**feedback, public_entities=entities)
        other = self.shadow_controller.observe(**feedback, public_entities=snapshot(entities))
        if (update != other or not np.array_equal(self.learner.theta, self.shadow_learner.theta)
                or not np.array_equal(self.learner.covariance, self.shadow_learner.covariance)):
            raise AssertionError('independent causal numeric updates disagree')
        digest_row(self.digest, baseline, action, decision, feedback, update)
        self.public_inputs.append(entities)
        self.records.append(row)
        self.pending = None
        return row


def replay(receipt, policy):
    """No native environment is constructed: replay exact archived public inputs."""
    learner = VisualMotorFIR.from_state(receipt['learner_before'])
    controller = VisualMotorController(policy, learner)
    game_start_model = learner.frozen_branch()
    game_start_controller = VisualMotorController(policy, game_start_model)
    within_game_changed = 0
    inputs = load_public_entities(receipt['public_file'])
    with np.load(receipt['tape_file'], allow_pickle=False) as data:
        assert len(inputs) == len(data['action']) + 1
        digest = hashlib.sha256()
        for i, action in enumerate(data['action']):
            chosen, decision = controller.choose(inputs[i], int(data['baseline_action'][i]))
            assert chosen == int(action), ('archived numeric action mismatch', i)
            _, start_decision = game_start_controller.choose(inputs[i], int(data['baseline_action'][i]))
            def before_dither(value):
                return min((1, 0, 2), key=lambda a: value['costs'][a]) if value['ready'] else value['inherited_action']
            within_game_changed += int(learner.updates > receipt['learner_before']['updates']
                and before_dither(decision) != before_dither(start_decision))
            feedback = dict(own_hits=int(data['hits'][i]), reward=float(data['reward'][i]),
                terminated=bool(data['terminated'][i]), truncated=bool(data['truncated'][i]),
                elapsed_seconds=float(data['elapsed_seconds'][i]))
            update = controller.observe(**feedback, public_entities=inputs[i + 1])
            # This comparison receives the same ACTUAL command history, never
            # a second imagined trajectory or another game query. Its choices
            # are reporting only and do not enter the live controller.
            game_start_controller.pending_action = chosen
            game_start_controller.inherited.pending_action = chosen
            game_start_controller.observe(**feedback, public_entities=inputs[i + 1])
            digest_row(digest, int(data['baseline_action'][i]), chosen, decision, feedback, update)
    assert digest.hexdigest() == receipt['public_decision_update_digest']
    assert state_hash(learner.state()) == receipt['learner_after_sha256']
    return dict(passed=True, rows=receipt['transitions'], native_cameras=0,
        public_inputs='all exact public visual entities, inherited baseline commands and actual public feedback',
        exact_actions_decisions_updates_and_final_state=True,
        within_game_update_changed_unperturbed_actions=within_game_changed,
        within_game_comparison='game-start coefficients and gate, same actual public observations/request history; descriptive action effect, not performance')


def evaluate_game(baseline, policy, learner, shadow, assignment, directory, deadline):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    before = learner.state()
    session = NumericEvaluator(baseline, policy, learner, shadow,
        PongEvaluationConfig(seed=int(assignment['seed']), episodes=1,
            max_episode_ticks=2400, opponent_speed=.65,
            policy_scaling=read(MODEL / 'radial-cap.json')))
    install_exact_transport_copy(session)
    install_intervention(session.env, assignment['intervention'])
    counter, events = ReturnCounter(), []
    game = session.env.game
    original = game.step
    def instrument(drive, **kwargs):
        opportunities, returns = counter.opportunities, counter.returns
        tick = int(game.rally_steps[0]) + 1
        result = counter.step(game, original, drive, **kwargs)
        if counter.opportunities != opportunities:
            events.append(dict(tick=tick, hit=counter.returns > returns, first=opportunities == 0))
        return result
    game.step = instrument
    attempted = 0
    initial_attempted = False
    error = None
    reason = 'complete'
    frozen = False
    try:
        if time.monotonic() >= deadline:
            reason = 'deadline_before_initial'
        else:
            initial_attempted = True
            session._begin_episode()
            while not session.complete:
                if attempted >= 1000:
                    reason = 'camera_grant'
                    break
                if time.monotonic() >= deadline:
                    reason = 'wall_deadline'
                    break
                attempted += 1
                session.step(check_frozen=False)
        session.verify_frozen()
        frozen = True
    except Exception:
        reason, error = 'native_failure', traceback.format_exc()
    finally:
        del game.step
    report = session.summary()
    last = report['rallies'][-1] if report['rallies'] else dict(truncated=False,
        terminated=False, reward=session.episode_return, seconds=session.episode_seconds)
    annotation = counter.summary(game, **{k: last[k]
        for k in ('truncated', 'terminated', 'reward', 'seconds')})
    complete = bool(session.complete and frozen and error is None)
    if complete:
        assert len(session.records) == session.transitions
        assert len(session.public_inputs) == session.transitions + 1
        assert native_count_status(annotation, report)['valid_terminal_counts']
        assert state_hash(learner.state()) == state_hash(shadow.state())
    index = assignment['index']
    tape = directory / f'game-{index:03d}-events.npz'
    public = directory / f'game-{index:03d}-public.npz'
    np.savez_compressed(tape, action=np.asarray([r['action'] for r in session.records], np.int8),
        baseline_action=np.asarray(session.baseline_actions[:len(session.records)], np.int8),
        hits=np.asarray([r['hits'] for r in session.records], np.int8),
        reward=np.asarray([r['reward'] for r in session.records], np.float64),
        elapsed_seconds=np.asarray([r['elapsed_seconds'] for r in session.records], np.float64),
        physics_ticks=np.asarray([r['physics_ticks'] for r in session.records], np.int8),
        terminated=np.asarray([r['terminated'] for r in session.records], bool),
        truncated=np.asarray([r['truncated'] for r in session.records], bool),
        opportunity_tick=np.asarray([e['tick'] for e in events], np.int32),
        opportunity_hit=np.asarray([e['hit'] for e in events], bool),
        opportunity_first=np.asarray([e['first'] for e in events], bool))
    save_public_entities(public, session.public_inputs)
    result = dict(contract=CONTRACT, assignment=assignment, complete=complete,
        stop_reason=reason, error=error, report=report, annotation=annotation,
        learner_before=before, learner_after=learner.state(),
        learner_before_sha256=state_hash(before), learner_after_sha256=state_hash(learner.state()),
        transitions=session.transitions, initial_cameras=session.initial_cameras,
        transition_camera_upper_bound=max(attempted, session.transitions),
        initial_camera_upper_bound=max(int(initial_attempted), session.initial_cameras),
        frozen_inherited_parameters_checked=frozen,
        tape_file=str(tape), tape_sha256=sha(tape), public_file=str(public), public_sha256=sha(public),
        public_decision_update_digest=session.digest.hexdigest(),
        numeric_actions=session.controller.numeric_actions,
        numeric_changed_from_inherited=session.controller.changed_from_inherited,
        new_numeric_updates=learner.updates - before['updates'],
        coefficients_changed=learner.theta.tolist() != before['theta'],
        model_diagnostics=dict(updates=learner.updates,
            preupdate_squared_error=learner.preupdate_squared_error - before['preupdate_squared_error'],
            zero_model_squared_error=learner.zero_model_squared_error - before['zero_model_squared_error']),
        wall_seconds=time.monotonic() - began)
    if complete:
        result['full_public_replay'] = replay(result, policy)
    write(directory / f'game-{index:03d}.json', result)
    return result


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve all previous specifications and evidence')
    if sha(MODEL / 'rbf-model.npz') != EXPECTED:
        raise ValueError('inherited baseline hash differs')
    destination.mkdir()
    used = exclusions(destination)
    schedules = {}
    index = 0
    for tag, phase, n, delay in ((63611, 'A_acquire', 8, 0),
            (63612, 'B_delay', 16, 8), (63613, 'A_return', 8, 0)):
        drawn = select_starts(1943906127, tag, used, n // 2)
        seeds = drawn['TRAIN']
        used.update(seed % base.MODULUS for seed in seeds)
        schedules[phase] = [dict(index=index + i, phase=phase, phase_index=i,
            seed=seed, intervention=dict(delay_commands=delay, drive_bias=0.))
            for i, seed in enumerate(seeds)]
        index += n
    write(destination / 'schedules.json', schedules)
    inherited = make_roster(load_calibration_scales(CALIBRATION))[4]
    policy_from_entry(inherited).save(destination / 'inherited-structure04.npz')
    names = set(compact.FILES) | {Path(__file__).name, 'pong_visual_motor_fir.py',
        'run_pong_numeric_challenge_scout.py', 'run_pong_carried_learning.py',
        'pong_carried_structure_learner.py', 'run_pong_relational_categorical_replication.py'}
    paths = [Path(__file__).with_name(n) for n in sorted(names)]
    paths += [MODEL / 'rbf-model.npz', MODEL / 'radial-cap.json', CALIBRATION,
        destination / 'inherited-structure04.npz', destination / 'schedules.json',
        destination / 'seed-exclusion-proof.json']
    spec = dict(contract=CONTRACT, source=base.source_signature(),
        files={str(p): sha(p) for p in paths}, inherited=inherited,
        settings=DEFAULTS, random_seed=2026110501, seconds=600, workers=4,
        live_games=32, frozen_acquired_A_games=16, inherited_only_B_games=16,
        total_games=64, maximum_transition_cameras=64000,
        schedules_sha256=sha(destination / 'schedules.json'),
        inputs=['public visual entities', 'requested command history',
            'actual public elapsed time', 'own-hit/terminal feedback for inherited history only'],
        learned='33 continuous FIR/intercept coefficients from past observed visual displacement; no reward optimization',
        forgetting_clock='actual elapsed time of newly admitted observed transitions; unsupported gaps do not age model',
        control='3 constant candidates through learned public model, .32s horizon; inherited relation fallback until data-supported gain',
        intervention='execution-only FIFO delay8, initial neutral queue; no phase/delay input to learner',
        controls='frozen acquired A coefficients with ordinary runtime/RNG history on matched B starts; inherited zero-coefficient servo on same B starts',
        no_backprop=True, no_privileged_inputs=True, no_expert_labels=True,
        no_parameter_grid=True, no_automatic_extension=True,
        artifacts='every exact public input, command, actual feedback, exact offline replay, per-game checkpoints',
        within_game_attribution='offline game-start coefficient/gate comparison on actual public observations and requested history; unperturbed decisions only',
        scope='development pilot; independently replicated fresh qualification required')
    write(destination / 'spec.json', spec)
    return dict(directory=str(destination), spec_sha256=sha(destination / 'spec.json'),
        games=spec['total_games'], maximum_transition_cameras=spec['maximum_transition_cameras'], native_started=False)


def verify(destination, approved):
    spec = read(destination / 'spec.json')
    assert sha(destination / 'spec.json') == approved
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    return spec


def summarize(rows):
    complete = [r for r in rows if r['complete']]
    return dict(saved_games=len(rows), completed_games=len(complete),
        aggregate=aggregate([r['annotation'] for r in complete]),
        transitions=sum(r['transitions'] for r in rows),
        initial_cameras=sum(r['initial_cameras'] for r in rows),
        transition_camera_upper_bound=sum(r.get('transition_camera_upper_bound', r['transitions']) for r in rows),
        initial_camera_upper_bound=sum(r.get('initial_camera_upper_bound', r['initial_cameras']) for r in rows),
        not_started_games=sum(bool(r.get('not_started')) for r in rows),
        numeric_actions=sum(r.get('numeric_actions', 0) for r in rows),
        numeric_changed_from_inherited=sum(r.get('numeric_changed_from_inherited', 0) for r in rows),
        numeric_updates=sum(r.get('new_numeric_updates', 0) for r in rows),
        within_game_update_changed_unperturbed_actions=sum(r.get('full_public_replay', {}).get(
            'within_game_update_changed_unperturbed_actions', 0) for r in rows),
        own_hits=[r.get('report', {}).get('hits') for r in rows])


def frozen_stream(destination, spec, state, assignments, deadline):
    base.initialize()
    learner = VisualMotorFIR.from_state(state).frozen_branch()
    shadow = VisualMotorFIR.from_state(learner.state())
    baseline = load_readout(MODEL / 'rbf-model.npz')
    policy = policy_from_entry(spec['inherited'])
    rows = []
    initial_theta = learner.theta.copy()
    initial_covariance = learner.covariance.copy()
    for assignment in assignments:
        if time.monotonic() >= deadline:
            break
        row = evaluate_game(baseline, policy, learner, shadow, assignment,
            Path(destination) / 'frozen-acquired-A', deadline)
        rows.append(row)
        assert np.array_equal(learner.theta, initial_theta)
        assert np.array_equal(learner.covariance, initial_covariance)
        if not row['complete']:
            break
        learner = VisualMotorFIR.from_state(learner.state())
    result = summarize(rows)
    write(Path(destination) / 'frozen-acquired-A-summary.json', result)
    return rows


def inherited_task(destination, assignment, deadline):
    original_class = compact.CompactEvaluator
    class DelayedEvaluator(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_intervention(self.env, assignment['intervention'])
    compact.CompactEvaluator = DelayedEvaluator
    try:
        row = compact.evaluate(str(Path(destination) / 'inherited-structure04.npz'),
            assignment['seed'], 2400, Path(destination) / 'inherited-only-B',
            deadline, 1000, CONTRACT + ':inherited-only-B')
    finally:
        compact.CompactEvaluator = original_class
    row['assignment'] = assignment
    write(Path(destination) / 'inherited-only-B' / f"game-{assignment['index']:03d}.json", row)
    return row


def run(destination, approved):
    destination = Path(destination)
    spec = verify(destination, approved)
    if (destination / 'work-clock.json').exists():
        raise FileExistsError('never retry or extend a started allocation')
    began = time.monotonic()
    deadline = began + spec['seconds']
    write(destination / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        began_monotonic=began, deadline_monotonic=deadline, approved_spec=approved))
    base.initialize()
    seed_rngs()
    learner = VisualMotorFIR(spec['random_seed'], settings=spec['settings'])
    shadow = VisualMotorFIR.from_state(learner.state())
    baseline = load_readout(MODEL / 'rbf-model.npz')
    policy = policy_from_entry(spec['inherited'])
    schedules = read(destination / 'schedules.json')
    rows = []
    for assignment in schedules['A_acquire']:
        if time.monotonic() >= deadline:
            break
        row = evaluate_game(baseline, policy, learner, shadow, assignment, destination / 'live', deadline)
        rows.append(row)
        write(destination / 'checkpoint-latest.json', learner.state())
        if not row['complete']:
            break
        learner = VisualMotorFIR.from_state(read(destination / 'checkpoint-latest.json'))
    write(destination / 'checkpoint-end-A.json', learner.state())
    frozen_rows, inherited_rows, errors = [], [], []
    controls_ready = (len(rows) == 8 and all(r['complete'] for r in rows)
        and learner.updates >= spec['settings']['initial_dither_updates'])
    if len(rows) == 8 and all(r['complete'] for r in rows) and not controls_ready:
        errors.append(dict(stream='controls', error='insufficient acquired A updates to match continuing dither probability'))
    if controls_ready:
        with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'),
                initializer=base.initialize) as pool:
            frozen_future = pool.submit(frozen_stream, str(destination), spec,
                learner.state(), schedules['B_delay'], deadline)
            inherited_futures = [pool.submit(inherited_task, str(destination), a, deadline)
                for a in schedules['B_delay']]
            for assignment in schedules['B_delay'] + schedules['A_return']:
                if time.monotonic() >= deadline:
                    break
                row = evaluate_game(baseline, policy, learner, shadow, assignment,
                    destination / 'live', deadline)
                rows.append(row)
                write(destination / 'checkpoint-latest.json', learner.state())
                write(destination / 'progress.json', dict(phase=assignment['phase'],
                    phase_index=assignment['phase_index'], **summarize(rows)))
                if not row['complete']:
                    break
                learner = VisualMotorFIR.from_state(read(destination / 'checkpoint-latest.json'))
            for name, assignment, future in [('frozen', None, frozen_future)] + [
                    ('inherited', assignment, f) for assignment, f in zip(schedules['B_delay'], inherited_futures)]:
                try:
                    result = future.result()
                    if name == 'frozen':
                        frozen_rows = result
                    else:
                        inherited_rows.append(result)
                except Exception as error:
                    errors.append(dict(stream=name, assignment=assignment, error=repr(error),
                        unknown_unsaved_native_cost=True))
            # A worker may save evidence and then fail during replay, summary,
            # or its final return. Recover all known saved games; explicitly
            # retain the unknown-cost flag for any unsaved work before failure.
            if any(e['stream'] == 'frozen' for e in errors):
                frozen_rows = [read(p) for p in sorted((destination / 'frozen-acquired-A').glob('game-*.json'))]
            if any(e['stream'] == 'inherited' for e in errors):
                inherited_rows = []
                for assignment in schedules['B_delay']:
                    named = destination / 'inherited-only-B' / f"game-{assignment['index']:03d}.json"
                    key = compact.fitness_key(str(destination / 'inherited-structure04.npz'),
                        assignment['seed'], 2400, CONTRACT + ':inherited-only-B')
                    fallback = destination / 'inherited-only-B' / (key + '.json')
                    found = named if named.exists() else fallback
                    if found.exists():
                        saved = read(found)
                        saved['assignment'] = assignment
                        inherited_rows.append(saved)
    groups = {phase: summarize([r for r in rows if r['assignment']['phase'] == phase])
        for phase in schedules}
    result = dict(contract=CONTRACT, complete=len(rows) == 32 and len(frozen_rows) == 16
        and len(inherited_rows) == 16 and all(r['complete'] for r in rows + frozen_rows + inherited_rows)
        and not errors, live=groups, frozen_acquired_A_B=summarize(frozen_rows),
        inherited_only_B=summarize(inherited_rows), errors=errors,
        transitions=sum(r['transitions'] for r in rows + frozen_rows + inherited_rows),
        initial_cameras=sum(r['initial_cameras'] for r in rows + frozen_rows + inherited_rows),
        known_transition_camera_upper_bound=sum(r.get('transition_camera_upper_bound', r['transitions'])
            for r in rows + frozen_rows + inherited_rows),
        known_initial_camera_upper_bound=sum(r.get('initial_camera_upper_bound', r['initial_cameras'])
            for r in rows + frozen_rows + inherited_rows),
        unknown_unsaved_failed_native_cost=any(e.get('unknown_unsaved_native_cost', False) for e in errors),
        maximum_total_transition_allocation=spec['maximum_transition_cameras'],
        frozen_A_control_dither_probability_matched=controls_ready,
        source_assets_spec_frozen=verify(destination, approved) == spec,
        all_numeric_full_public_replays_passed=all(r.get('full_public_replay', {}).get('passed', False)
            for r in rows + frozen_rows), wall_seconds=time.monotonic() - began,
        no_completion_claim=True, pilot_only=True)
    assert result['transitions'] <= spec['maximum_transition_cameras']
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
