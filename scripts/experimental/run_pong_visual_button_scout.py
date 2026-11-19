"""Fixed native scout for empirical translation of inherited movement intent."""

import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fly_connectome.control.session import MODEL_HASH, model_bytes, source_signature
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
import run_pong_inverse_cycle as native
from run_pong_surprise_recovery import POLICY, exclusions, reached_deadline, score, state_hash

PHASES = (
    ('normal', 6, (0, 1, 2)),
    ('cycle', 10, (1, 2, 0)),
    ('inverse', 10, (2, 0, 1)),
    ('restored', 6, (0, 1, 2)),
)
START_SEED = 560687829
PROBE_SEED = 91041985
WALL_BUDGET = 600
MODULUS = 2147483647
FROZEN_MATCHES = 6
FILES = ('run_pong_visual_button_scout.py', 'pong_visual_button_effects.py',
         'run_pong_inverse_cycle.py', 'pong_relational_feedback.py',
         'pong_relational_persistent_binding.py', 'public_entity_pong_native.py',
         'return_metrics.py', 'run_pong_surprise_recovery.py')


def manifest(directory):
    used, metadata = exclusions(directory.parent)
    used.update((PROBE_SEED, START_SEED))
    rng = np.random.default_rng(START_SEED)
    assignments = []
    for phase, count, mapping in PHASES:
        for local_index in range(count):
            while True:
                seed = int(rng.integers(1, MODULUS))
                if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == local_index % 2:
                    break
            used.add(seed)
            assignments.append(dict(index=len(assignments), phase_index=local_index, seed=seed,
                                    phase=phase, mapping=mapping))
    directory.mkdir(parents=True, exist_ok=False)
    native.save(directory / 'schedules.json', assignments)
    spec = dict(contract='native-public-inverse-button-effects-scout-v1', budget_seconds=WALL_BUDGET,
                budget_games=44, budget_transition_cameras=44000, budget_initial_cameras=44,
                probe_seed=PROBE_SEED, frozen_vision_sha256=MODEL_HASH, package_source=source_signature(),
                files={name: native.sha(Path(__file__).with_name(name)) for name in FILES},
                schedules_sha256=native.sha(directory / 'schedules.json'), prior_seed_metadata=metadata,
                policy=POLICY, frozen_comparison='separate end-normal frozen branches for first six starts of each cycle',
                fixed_reporting='all games, first four, final six, first/later returns, actual time and exploratory holds',
                no_catalogue=True, no_backprop=True, no_hidden_learner_inputs=True, exploratory_only=True)
    native.save(directory / 'spec.json', spec)
    return spec, assignments, native.sha(directory / 'spec.json')


def play_game(directory, assignment, learner, name, deadline):
    if time.monotonic() >= deadline:
        return None
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    before = learner.state()
    native.save(directory / 'active.json', dict(assignment=assignment, arm=name, camera_grant=1000))
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - started))
    try:
        outcome = native.play(learner, policy, assignment['seed'], assignment['mapping'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    after = learner.state()
    assert outcome['transitions'] <= 1000 and outcome['initial'] == 1
    if not learner.updates_enabled:
        assert before == after
    (directory / name).mkdir(exist_ok=True)
    row = dict(**assignment, outcome=outcome, arm=name,
        before=state_hash(before), after=state_hash(after), state=after,
        probe_frames=after['dither_count'] - before['dither_count'], surprise_episodes=0,
        recalibrations=after['calibrations'] - before['calibrations'],
        contradictions=after['contradictions'] - before['contradictions'],
        wall_seconds=time.monotonic() - started)
    native.save(directory / name / f"game-{assignment['index']:02}.json", row)
    (directory / 'active.json').unlink()
    return row


def run(directory):
    torch.set_num_threads(1)
    model_bytes()
    signal.signal(signal.SIGALRM, reached_deadline)
    native.CategoricalMotorController = VisualIntentController
    spec, assignments, digest = manifest(directory)
    begin = time.monotonic()
    deadline = begin + WALL_BUDGET
    native.save(directory / 'work-clock.json', dict(start=begin, deadline=deadline, spec_sha256=digest))
    learner = VisualButtonEffects(PROBE_SEED)
    native.save(directory / 'start.json', learner.state())
    completed, frozen = [], []
    acquired = None
    timeout = False
    try:
        for assignment in assignments:
            row = play_game(directory, assignment, learner, 'live', deadline)
            if row is None:
                timeout = True
                break
            completed.append(row)
            print('live', assignment['phase'], assignment['phase_index'] + 1, row['outcome']['returns'],
                  learner.effects, row['probe_frames'], flush=True)
            if assignment['phase'] == 'normal' and assignment['phase_index'] == PHASES[0][1] - 1:
                acquired = learner.state()
                native.save(directory / 'acquired.json', acquired)
        if acquired is not None and not timeout:
            for phase in ('cycle', 'inverse'):
                branch = VisualButtonEffects.from_state(acquired).frozen_branch()
                for assignment in assignments:
                    if assignment['phase'] == phase and assignment['phase_index'] < FROZEN_MATCHES:
                        row = play_game(directory, assignment, branch, 'frozen', deadline)
                        if row is None:
                            timeout = True
                            break
                        frozen.append(row)
                if timeout:
                    break
    except TimeoutError:
        timeout = True
    phases = {}
    for name, count, mapping in PHASES:
        rows = [r for r in completed if r['phase'] == name]
        match = [r for r in frozen if r['phase'] == name]
        phases[name] = dict(full=score(rows), first_four=score(rows[:4]),
            final_six=score([r for r in rows if r['phase_index'] >= count - 6]),
            paired_live=score(rows[:len(match)]), frozen=score(match))
    rows = completed + frozen
    source_unchanged = (native.sha(directory / 'spec.json') == digest
        and native.sha(directory / 'schedules.json') == spec['schedules_sha256']
        and source_signature() == spec['package_source']
        and all(native.sha(Path(__file__).with_name(name)) == value for name, value in spec['files'].items()))
    model_bytes()
    result = dict(phases=phases, exact_completed_games=len(rows), transition_cameras=sum(r['outcome']['transitions'] for r in rows),
                  initial_cameras=sum(r['outcome']['initial'] for r in rows),
                  campaign_complete=len(rows) == 44 and not timeout, timeout=timeout,
                  unsaved_camera_grant=1000 if (directory / 'active.json').exists() else 0,
                  spec_sha256=digest, sources_unchanged=source_unchanged, wall_seconds=time.monotonic()-begin)
    native.save(directory / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(run(parser.parse_args().directory.resolve())))
