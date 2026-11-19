"""Smoke-test the opt-in rise-time observer with the v3 FIFO restore correction."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from shutil import copyfile
from time import monotonic

import numpy as np
import torch

from fly_connectome.control.session import model_bytes, source_signature
from pong_actuator_response import ActuatorResponseController
from pong_timing_controller import TimedMotorController
from pong_transfer_arena import interception_policy, play
from pong_visual_button_effects import VisualButtonEffects
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import MODULUS, exclusions


ARMS = ('live', 'comparison')
# (physical actuator response, actual FIFO delay, held checkpoint before serve)
PHASES = (
    ('normal_first', .08, 0, (None, None, None)),
    ('retained_fifo', .08, 12, (12, 12, 12)),
    ('restored_fifo', .08, 0, (12, 21, 21)),
    ('slower_actuator', .32, 0, (None, None, None, None)),
)
SEED_GENERATOR = 92661439
DURATION = 8.


class EarlyRestoreTrace:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_rejections = []

    def _check_early_motion(self, ordinal, position):
        before = self.carried_delay
        super()._check_early_motion(ordinal, position)
        if before and not self.carried_delay:
            self.early_rejections.append(dict(carried=before, elapsed_seconds=self.seconds,
                age_cameras=ordinal - self.first_motion_request, selected=self.delay_commands))


class TimedReference(EarlyRestoreTrace, TimedMotorController):
    pass


class ResponseCandidate(EarlyRestoreTrace, ActuatorResponseController):
    pass


def read(path):
    return json.loads(path.read_text())


def prepare(directory, checkpoint, held):
    model_bytes()
    used, _ = exclusions(directory.parent)
    rng = np.random.default_rng(SEED_GENERATOR)
    schedule = []
    for phase, seconds, delay, starts in PHASES:
        for local_index, carried in enumerate(starts):
            while True:
                seed = int(rng.integers(1, MODULUS))
                if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == local_index % 2:
                    break
            used.add(seed)
            schedule.append(dict(phase=phase, seed=seed, seconds=seconds, delay=delay, carried=carried))
    directory.mkdir(exist_ok=False, parents=True)
    inputs = {'checkpoint-acquired.json': checkpoint, **{
        f'held-{delay}.json': held / f'held-{delay}.json' for delay in (12, 21)}}
    for name, path in inputs.items():
        copyfile(path, directory / name)
    save(directory / 'schedules.json', schedule)
    sources = ('check_pong_actuator_response_restore.py', 'pong_actuator_response.py', 'pong_timing_fast.py',
               'pong_timing_controller.py', 'pong_rapid_motor_recovery.py', 'pong_transfer_arena.py')
    spec = dict(contract='paired-native-actuator-response-with-v3-fifo-restore-smoke-v1',
        source=source_signature(), scripts={name: sha(Path(__file__).with_name(name)) for name in sources},
        inputs={name: sha(directory / name) for name in (*inputs, 'schedules.json')},
        standard_comparison='scalar v3 timing on normal, retained FIFO and restored FIFO',
        slower_comparison='same controller with response-time updates frozen',
        goal='same visual interception without score-based placement updates',
        max_seconds_per_point=DURATION, workers=2)
    save(directory / 'spec.json', spec)
    return schedule, spec


def run_arm(directory, schedule, arm):
    torch.set_num_threads(1)
    policy = interception_policy()
    learner = VisualButtonEffects.from_state(read(directory / 'checkpoint-acquired.json'))
    held = {delay: read(directory / f'held-{delay}.json') for delay in (12, 21)}
    slow_state = None
    rows = []
    for case in schedule:
        init = held.get(case['carried'])
        slow = case['phase'] == 'slower_actuator'
        if slow:
            init = slow_state
        if arm == 'live' or slow:
            controller = ResponseCandidate(policy, learner, timing_state=init,
                                           response_updates=arm == 'live')
        else:
            controller = TimedReference(policy, learner, timing_state=init)
        row = play(controller, case['seed'], activation_seconds=case['seconds'],
                   delay_frames=case['delay'], duration=DURATION)
        end_state = controller.state()
        if slow:
            slow_state = end_state
        rows.append(dict(**row, phase=case['phase'], end_state=end_state,
                         early_rejections=controller.early_rejections))
        save(directory / f'{arm}.json', rows)
        print(arm, case['phase'], case['seed'], row['hits'], row['opportunities'],
              end_state['delay_commands'], controller.early_rejections, flush=True)


def metrics(rows):
    opportunities = [row['contacts'] for row in rows if row['contacts']]
    later = [hit for trajectory in opportunities for _, hit in trajectory[1:]]
    early = [row['early_rejections'][0] for row in rows if row['early_rejections']]
    return dict(first_returns=sum(trajectory[0][1] for trajectory in opportunities),
        first_opportunities=len(opportunities), later_returns=sum(later), later_opportunities=len(later),
        early_rejection_seconds=[item['elapsed_seconds'] for item in early],
        early_rejection_age_cameras=[item['age_cameras'] for item in early],
        end_delays=[row['end_state']['delay_commands'] for row in rows],
        end_response_estimates=[row['end_state'].get('response_seconds') for row in rows])


def run(directory, checkpoint, held):
    schedule, spec = prepare(directory, checkpoint, held)
    start = monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_arm, directory, schedule, arm) for arm in ARMS]
        for job in jobs:
            job.result()
    receipts = {arm: read(directory / f'{arm}.json') for arm in ARMS}
    phases = {}
    for phase, _, _, cases in PHASES:
        groups = {arm: [row for row in receipts[arm] if row['phase'] == phase] for arm in ARMS}
        pairs = list(zip(*groups.values(), strict=True))
        assert len(pairs) == len(cases) and all(a['seed'] == b['seed'] for a, b in pairs)
        phases[phase] = dict(**{arm: metrics(rows) for arm, rows in groups.items()},
            identical_games=sum(a['action_sha256'] == b['action_sha256'] for a, b in pairs),
            identical_early_rejections=sum(a['early_rejections'] == b['early_rejections'] for a, b in pairs))
    assert source_signature() == spec['source']
    assert all(sha(Path(__file__).with_name(name)) == digest for name, digest in spec['scripts'].items())
    assert all(sha(directory / name) == digest for name, digest in spec['inputs'].items())
    report = dict(phases=phases, wall_seconds=monotonic() - start,
        transition_cameras=sum(row['cameras'] for rows in receipts.values() for row in rows),
        initial_cameras=2 * len(schedule), sources_unchanged=True, spec_sha256=sha(directory / 'spec.json'))
    save(directory / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('held_checkpoints', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.checkpoint.resolve(),
                         args.held_checkpoints.resolve()), sort_keys=True))
