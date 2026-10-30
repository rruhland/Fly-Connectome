"""Prospectively bounded chronological acquisition and hidden A/B/A adaptation.

Only separate learner streams run in parallel. Within a stream every chosen
structure is followed by its own actual feedback before the next choice. No
counterfactual starts, cached fitness or frozen validation outcome is selected.
Preparation and native execution are separate commands.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import struct
import time
import traceback

import numpy as np

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from compact_public_pong import controller_state, FILES as FROZEN_FILES
from exact_visual_transport_copy import install_exact_transport_copy
from pong_carried_structure_learner import CarriedStructureLearner, state_hash
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_relational_feedback import RelationalPolicy, load_calibration_scales
from pong_relational_persistent_binding import PersistentRelationalPolicy, PersistentRelationalController
from public_entity_pong_native import snapshot, save_public_entities
from return_metrics import ReturnCounter, aggregate, native_count_status
from run_pong_relational_categorical_replication import make_roster, CALIBRATION
from run_rally_option_memory import MODEL, EXPECTED, sha, read, write, seed_rngs
from run_rally_repertoire_collection import select_starts, exclusions
import run_practical_pong_v2 as base

CONTRACT = 'chronological-public-reward-structure-learning-v1'
ROOT = Path('/workspace/m1b-experiments')
DEFAULT_DEST = ROOT / 'pong-carried-learning-20261030-r1'
STREAMS = ('capped_beta', 'discounted_ucb')


def feedback(row):
    """The complete, deliberately narrow learner feedback boundary."""
    return dict(own_hits=int(row['hits']), reward=float(row['reward']),
        terminated=bool(row['terminated']), truncated=bool(row['truncated']),
        elapsed_seconds=float(row['elapsed_seconds']))


def apply_actuator_intervention(environment, reverse):
    """Wrap only physical execution; policy command and its history stay intact."""
    original = environment.step
    def execute(action):
        return original(2 - action if reverse else action)
    environment.step = execute
    return original


def policy_from_entry(entry):
    return PersistentRelationalPolicy(RelationalPolicy(*entry['choices'],
        entry['coefficients'], entry['scales']), require_first_hit=False)


class CarriedEvaluator(FreshRBFEvaluator):
    def __init__(self, baseline, learner, shadow, policy, shadow_policy, config):
        self.learner, self.shadow_learner = learner, shadow
        self.structural_policy, self.shadow_policy = policy, shadow_policy
        self.controller = PersistentRelationalController(policy)
        self.shadow_controller = PersistentRelationalController(shadow_policy)
        self.digest = hashlib.sha256()
        self.ring = deque(maxlen=64)
        self.pending = None
        self.checked_rows = 0
        self.learning_events = []
        super().__init__(baseline, config)
        self.initial_learned_hash = self.learned_hash()

    def learned_hash(self):
        # Intentionally mutable learner evidence is checked against an
        # independent shadow. Vision, inherited baseline and all structural
        # execution parameters remain subject to the original frozen audit.
        return fingerprint((super().learned_hash(), CONTRACT,
            self.structural_policy.frozen_parameters(), self.shadow_policy.frozen_parameters()))

    def _choose_policy(self):
        baseline, q = self.rbf_readout.choose(*policy_features(self.latest_encoded, self.scaler))
        entities = self.latest_encoded['entities']
        owned = snapshot(entities)
        before = controller_state(self.controller)
        action, decision = self.controller.choose(entities, baseline)
        other, shadow_decision = self.shadow_controller.choose(owned, baseline)
        if action != other or decision != shadow_decision:
            raise AssertionError('independently constructed structural controllers disagree')
        entry = dict(ordinal=self.transitions, public_entities=owned,
            state_before=before, baseline_action=int(baseline),
            baseline_q=q.tolist(), action=int(action), decision=decision)
        self.ring.append(entry)
        self.pending = entry
        payload = json.dumps(entry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.digest.update(struct.pack('<I', len(payload)))
        self.digest.update(payload)
        return int(action), q

    def step(self, **kwargs):
        row = super().step(**kwargs)
        if self.pending is None or row['action'] != self.pending['action']:
            raise AssertionError('actual command differs from audited decision')
        self.controller.observe(row['hits'], row['terminated'])
        self.shadow_controller.observe(row['hits'], row['terminated'])
        if controller_state(self.controller) != controller_state(self.shadow_controller):
            raise AssertionError('command histories disagree')
        public = feedback(row)
        changed = row['hits'] or row['terminated'] or row['truncated']
        before = state_hash(self.learner.state()) if changed else None
        update = self.learner.observe(**public)
        other = self.shadow_learner.observe(**public)
        if update != other:
            raise AssertionError('online learner updates disagree')
        if changed:
            after = state_hash(self.learner.state())
            if after != state_hash(self.shadow_learner.state()):
                raise AssertionError('serialized reward histories disagree')
            self.learning_events.append(dict(transition=self.transitions - 1,
                before=before, after=after, feedback=public, update=update,
                success=list(self.learner.success), failure=list(self.learner.failure)))
        event = dict(action=int(row['action']), **public, physics_ticks=int(row['physics_ticks']))
        self.pending['record'] = event
        self.pending = None
        self.checked_rows += 1
        self.digest.update(json.dumps(event, sort_keys=True, separators=(',', ':')).encode())
        return row


def evaluate_game(baseline, learner, shadow, roster, assignment, directory, deadline, identity):
    """Exactly one new actual game and its chronologically applied updates."""
    began = time.monotonic()
    before_state = learner.state()
    arm, mode = learner.begin_game()
    other_arm, other_mode = shadow.begin_game()
    if (arm, mode) != (other_arm, other_mode):
        raise AssertionError('restored selector chose a different arm')
    policy, shadow_policy = policy_from_entry(roster[arm]), policy_from_entry(roster[arm])
    assert not np.shares_memory(policy.original.scales, shadow_policy.original.scales)
    session = CarriedEvaluator(baseline, learner, shadow, policy, shadow_policy,
        PongEvaluationConfig(seed=assignment['seed'], episodes=1,
            max_episode_ticks=assignment['max_episode_ticks'], opponent_speed=.65,
            policy_scaling=read(MODEL / 'radial-cap.json')))
    transport = install_exact_transport_copy(session)
    apply_actuator_intervention(session.env, assignment['reverse'])
    counter, events, rows = ReturnCounter(), [], []
    game = session.env.game
    original = game.step
    tail = None
    miss_pending = False
    def instrument(drive, **kwargs):
        nonlocal miss_pending
        opportunities, returns = counter.opportunities, counter.returns
        tick = int(game.rally_steps[0]) + 1
        outcome = counter.step(game, original, drive, **kwargs)
        if counter.opportunities > opportunities:
            hit = counter.returns > returns
            events.append(dict(tick=tick, hit=bool(hit), first=opportunities == 0))
            miss_pending |= not hit
        return outcome
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
                if session.transitions >= assignment['max_transition_cameras']:
                    reason = 'camera_grant'
                    break
                if time.monotonic() >= deadline:
                    reason = 'wall_deadline'
                    break
                attempted += 1
                rows.append(session.step(check_frozen=False))
                if miss_pending and tail is None:
                    tail = copy.deepcopy(list(session.ring))
        session.verify_frozen()
        frozen = True
    except Exception:
        reason, error = 'native_failure', traceback.format_exc()
        if tail is None:
            tail = copy.deepcopy(list(session.ring))
    finally:
        del game.step
    report = session.summary()
    last = report['rallies'][-1] if report['rallies'] else dict(truncated=False,
        terminated=False, reward=session.episode_return, seconds=session.episode_seconds)
    annotation = counter.summary(game, **{key: last[key]
        for key in ('truncated', 'terminated', 'reward', 'seconds')})
    complete = bool(session.complete and frozen and error is None)
    if complete:
        assert session.checked_rows == session.transitions == len(rows)
        assert native_count_status(annotation, report)['valid_terminal_counts']
        assert state_hash(learner.state()) == state_hash(shadow.state())
        assert sum(event['hit'] for event in events) == report['hits']
    index = assignment['index']
    tape = directory / f'game-{index:03d}-events.npz'
    np.savez_compressed(tape, action=np.asarray([r['action'] for r in rows], np.int8),
        physical_action=np.asarray([2 - r['action'] if assignment['reverse'] else r['action'] for r in rows], np.int8),
        hits=np.asarray([r['hits'] for r in rows], np.int8),
        reward=np.asarray([r['reward'] for r in rows], np.float32),
        physics_ticks=np.asarray([r['physics_ticks'] for r in rows], np.int8),
        elapsed_seconds=np.asarray([r['elapsed_seconds'] for r in rows]),
        terminated=np.asarray([r['terminated'] for r in rows], bool),
        truncated=np.asarray([r['truncated'] for r in rows], bool),
        opportunity_tick=np.asarray([e['tick'] for e in events], np.int32),
        opportunity_hit=np.asarray([e['hit'] for e in events], bool),
        opportunity_first=np.asarray([e['first'] for e in events], bool))
    tail_artifact = None
    if tail:
        public = directory / f'game-{index:03d}-miss-public.npz'
        details = directory / f'game-{index:03d}-miss.json'
        save_public_entities(public, [entry['public_entities'] for entry in tail])
        write(details, [{k: v for k, v in entry.items() if k != 'public_entities'} for entry in tail])
        tail_artifact = dict(public_file=str(public), public_sha256=sha(public),
            decisions_file=str(details), decisions_sha256=sha(details), rows=len(tail))
    result = dict(contract=CONTRACT, runtime_identity=identity, assignment=assignment,
        arm=arm, arm_id=roster[arm]['id'], selection_mode=mode,
        learner_before=before_state, learner_after=learner.state(),
        learner_before_sha256=state_hash(before_state), learner_after_sha256=state_hash(learner.state()),
        learning_events=session.learning_events, complete=complete, stop_reason=reason,
        error=error, report=report, annotation=annotation, events=events,
        transitions=session.transitions, initial_cameras=session.initial_cameras,
        transition_camera_upper_bound=max(attempted, session.transitions),
        initial_camera_upper_bound=max(int(initial_attempted), session.initial_cameras),
        frozen_execution_parameters_checked=frozen, transport_copy=transport,
        tape_file=str(tape), tape_sha256=sha(tape), miss_tail=tail_artifact,
        streaming_audit=dict(passed=complete, checked_rows=session.checked_rows,
            command_and_public_input_digest=session.digest.hexdigest(),
            independently_restored_learner=True, independently_constructed_controllers=True,
            full_archived_public_replay=False), wall_seconds=time.monotonic() - began)
    write(directory / f'game-{index:03d}.json', result)
    return result


def summarize(rows, per_phase):
    result = {}
    for phase in ('A_acquire', 'B_reverse', 'A_return'):
        selected = [row for row in rows if row['assignment']['phase'] == phase and row['complete']]
        result[phase] = dict(all=aggregate([row['annotation'] for row in selected]),
            last16=aggregate([row['annotation'] for row in selected if row['assignment']['phase_index'] >= per_phase - 16]),
            selection_counts={mode: sum(row['selection_mode'] == mode for row in selected)
                for mode in sorted({row['selection_mode'] for row in selected})},
            chosen_arms=[row['arm'] for row in selected],
            own_hits_by_game=[row['report']['hits'] for row in selected])
    return result


def prepare(destination, games_per_phase=48, seconds=20, stream_seconds=600):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve the original prospective specification')
    if games_per_phase % 2 or games_per_phase < 16 or seconds not in (10, 20):
        raise ValueError('balanced fixed cohorts and 10/20-second games required')
    if sha(MODEL / 'rbf-model.npz') != EXPECTED:
        raise ValueError('inherited baseline mismatch')
    destination.mkdir()
    used = exclusions(destination)
    roster = make_roster(load_calibration_scales(CALIBRATION))
    schedules = {}
    for stream_index, kind in enumerate((*STREAMS, 'reset_beta', 'frozen')):
        assignments = []
        phases = ('B_reverse', 'A_return') if kind == 'frozen' else ('A_acquire', 'B_reverse', 'A_return')
        for phase_index, phase in enumerate(phases):
            drawn = select_starts(1943906127, 63501 + stream_index * 10 + phase_index,
                used, games_per_phase // 2)
            seeds = drawn.pop('TRAIN')
            used.update(seed % base.MODULUS for seed in seeds)
            for i, seed in enumerate(seeds):
                assignments.append(dict(index=len(assignments), phase=phase,
                    phase_index=i, seed=int(seed), reverse=phase == 'B_reverse',
                    max_episode_ticks=seconds * 120, max_transition_cameras=seconds * 50))
        schedules[kind] = assignments
    write(destination / 'schedules.json', schedules)
    names = set(FROZEN_FILES) | {Path(__file__).name, 'pong_carried_structure_learner.py',
        'run_pong_relational_categorical_replication.py'}
    files = {str(Path(__file__).with_name(name)): sha(Path(__file__).with_name(name)) for name in sorted(names)}
    spec = dict(contract=CONTRACT, source=base.source_signature(), files=files,
        assets={str(p): sha(p) for p in (MODEL / 'rbf-model.npz', MODEL / 'radial-cap.json', CALIBRATION)},
        schedules_sha256=sha(destination / 'schedules.json'),
        exclusions_sha256=sha(destination / 'seed-exclusion-proof.json'), roster=roster,
        pilot_streams=list(STREAMS), prepared_controls=['reset_beta', 'frozen'],
        games_per_phase=games_per_phase, seconds_per_game=seconds, seconds_per_stream=stream_seconds,
        workers=2, random_seeds=dict(capped_beta=2026103001, discounted_ucb=2026103002,
            reset_beta=2026103003, frozen=2026103004),
        transition_cap_per_stream=3 * games_per_phase * seconds * 50,
        whole_game_commitment=True, own_hit_updates_immediate=True,
        evidence_cap=8., discount=.95, exploration=.15,
        surprise='two consecutive exploiting games ending with public negative terminal reward; seeded full16 resweep, no evidence cleared',
        boundary_credit='own hits positive; terminated and reward<0 negative; positive termination and cap censor unfinished return',
        learner_inputs=['own_hits', 'reward', 'terminated', 'truncated', 'elapsed_seconds'],
        intervention='external legal action0/2 exchange; hidden from selector/controller; command history stays requested command',
        frozen_control='load capped_beta end-A checkpoint; freeze acquired incumbent/evidence; new B/A games only; never auto-run',
        reset_control='clear reward evidence at every game; keep only exogenous rotating RNG; never auto-run',
        prepared_controls_comparison='disjoint predetermined starts; unpaired descriptive controls only; final matched qualification requires a new prospective schedule',
        diagnostic='48games per phase; last16 is pilot diagnostic only; all live exposure including exploration counted',
        no_backprop=True, no_privileged_policy_inputs=True, no_expert_labels=True,
        no_counterfactual_or_repeated_starts=True, no_automatic_extension=True,
        inherited='frozen vision/baseline/unlabeled scales and 16 generic relational structural controllers',
        retained_artifacts='small event tapes, learner checkpoints, streaming digests and bounded miss tails')
    spec['identity'] = state_hash(spec)
    write(destination / 'spec.json', spec)
    return dict(status='prepared_no_native', directory=str(destination),
        spec_sha256=sha(destination / 'spec.json'), pilot_games=6 * games_per_phase,
        maximum_pilot_transition_cameras=2 * spec['transition_cap_per_stream'])


def verify_spec(destination, approved):
    destination = Path(destination)
    spec = read(destination / 'spec.json')
    if sha(destination / 'spec.json') != approved:
        raise ValueError('reviewed specification mismatch')
    if (spec['source'] != base.source_signature()
            or any(sha(p) != h for p, h in {**spec['files'], **spec['assets']}.items())
            or sha(destination / 'schedules.json') != spec['schedules_sha256']
            or sha(destination / 'seed-exclusion-proof.json') != spec['exclusions_sha256']):
        raise ValueError('pinned runtime or schedule changed')
    return spec


def run_stream(destination, approved, kind):
    destination = Path(destination)
    spec = verify_spec(destination, approved)
    directory = destination / kind
    if directory.exists():
        raise FileExistsError('never retry or extend a started chronological stream')
    directory.mkdir()
    began = time.monotonic()
    deadline = began + spec['seconds_per_stream']
    write(directory / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        seconds=spec['seconds_per_stream'], spec_sha256=approved))
    if kind == 'frozen':
        learned = read(destination / 'capped_beta/checkpoint-end-A.json')
        learner = CarriedStructureLearner.from_state(learned).frozen_branch()
    else:
        learner = CarriedStructureLearner(kind, len(spec['roster']), spec['random_seeds'][kind],
            evidence_cap=spec['evidence_cap'], discount=spec['discount'], exploration=spec['exploration'])
    shadow = CarriedStructureLearner.from_state(json.loads(json.dumps(learner.state())))
    initial_state = learner.state()
    write(directory / 'checkpoint-initial.json', initial_state)
    seed_rngs()
    base.initialize()
    baseline = load_readout(MODEL / 'rbf-model.npz')
    assignments = read(destination / 'schedules.json')[kind]
    rows = []
    for assignment in assignments:
        if time.monotonic() >= deadline:
            break
        row = evaluate_game(baseline, learner, shadow, spec['roster'], assignment,
            directory, deadline, spec['identity'])
        rows.append(row)
        write(directory / 'checkpoint-latest.json', learner.state())
        if (row['complete'] and assignment['phase'] == 'A_acquire'
                and assignment['phase_index'] == spec['games_per_phase'] - 1):
            write(directory / 'checkpoint-end-A.json', learner.state())
        write(directory / 'progress.json', dict(complete_games=sum(r['complete'] for r in rows),
            phase=assignment['phase'], phase_index=assignment['phase_index'],
            transitions=sum(r['transitions'] for r in rows),
            learner_sha256=state_hash(learner.state()), arm=row['arm'], hits=row['report']['hits']))
        if not row['complete']:
            break
        # Resume from JSON after EVERY game. Live and shadow use distinct
        # objects, making serialization part of actual carried operation.
        learner = CarriedStructureLearner.from_state(read(directory / 'checkpoint-latest.json'))
        assert state_hash(learner.state()) == state_hash(shadow.state())
    complete = len(rows) == len(assignments) and all(row['complete'] for row in rows)
    summary = dict(contract=CONTRACT, kind=kind, complete=complete,
        scheduled_games=len(assignments), completed_games=sum(row['complete'] for row in rows),
        metrics=summarize(rows, spec['games_per_phase']),
        initial_learner_sha256=state_hash(initial_state), final_learner_sha256=state_hash(learner.state()),
        final_state=learner.state(), transitions=sum(row['transitions'] for row in rows),
        initial_cameras=sum(row['initial_cameras'] for row in rows),
        transition_camera_upper_bound=sum(row['transition_camera_upper_bound'] for row in rows),
        initial_camera_upper_bound=sum(row['initial_camera_upper_bound'] for row in rows),
        all_completed_streaming_audits_passed=all(row['streaming_audit']['passed'] for row in rows),
        wall_seconds=time.monotonic() - began, pilot_only=True,
        source_and_assets_frozen=verify_spec(destination, approved) == spec)
    write(directory / 'summary.json', summary)
    return summary


def run_pilot(destination, approved):
    spec = verify_spec(destination, approved)
    with ProcessPoolExecutor(max_workers=spec['workers'], mp_context=mp.get_context('spawn'),
            initializer=base.initialize) as pool:
        futures = {kind: pool.submit(run_stream, str(destination), approved, kind)
            for kind in spec['pilot_streams']}
        results = {kind: future.result() for kind, future in futures.items()}
    summary = dict(complete=all(r['complete'] for r in results.values()),
        streams={kind: {key: result[key] for key in ('complete', 'completed_games', 'metrics',
            'transitions', 'initial_cameras', 'wall_seconds')} for kind, result in results.items()},
        transition_cameras=sum(r['transitions'] for r in results.values()),
        initial_cameras=sum(r['initial_cameras'] for r in results.values()),
        controls_run=False, success_declared=False, no_automatic_extension=True)
    write(Path(destination) / 'pilot-summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'pilot', 'stream'))
    parser.add_argument('--directory', default=str(DEFAULT_DEST))
    parser.add_argument('--approved-spec')
    parser.add_argument('--kind', choices=(*STREAMS, 'reset_beta', 'frozen'))
    args = parser.parse_args()
    if args.command == 'prepare':
        output = prepare(args.directory)
    elif args.command == 'pilot':
        output = run_pilot(args.directory, args.approved_spec)
    else:
        output = run_stream(args.directory, args.approved_spec, args.kind)
    print(json.dumps(output, sort_keys=True), flush=True)
