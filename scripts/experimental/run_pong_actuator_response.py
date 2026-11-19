"""Pair visual learning and frozen response time under native motor changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from fly_connectome.control.session import model_bytes, source_signature
from pong_actuator_response import ActuatorResponseController
from pong_transfer_arena import interception_policy, play
from pong_visual_button_effects import VisualButtonEffects
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import MODULUS, exclusions


PHASES = (
    ('slower', .32, 0, 12),
    ('restored', .08, 0, 6),
    ('normal', .08, 0, 6),
    ('fifo12', .08, 12, 3),
    ('fifo17', .08, 17, 3),
    ('fifo21', .08, 21, 3),
)
SEED_GENERATOR = 622371195
ARMS = ('live', 'response_frozen')
DURATION = 10.


def read(path):
    return json.loads(path.read_text())


def prepare(directory, checkpoint):
    used, excluded = exclusions(directory.parent)
    previous = directory.parent / 'pong-inertia-transfer-20261119-r1/spec.json'
    if previous.exists():
        used.update(read(previous)['seeds'])
        excluded[str(previous)] = sha(previous)
    rng = np.random.default_rng(SEED_GENERATOR)
    schedule = []
    for phase, seconds, delay, count in PHASES:
        for i in range(count):
            while True:
                seed = int(rng.integers(1, MODULUS))
                if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == i % 2:
                    break
            used.add(seed)
            schedule.append(dict(phase=phase, seed=seed, seconds=seconds, delay=delay,
                                 reset_model=i == 0 and phase not in ('slower', 'restored')))
    directory.mkdir(exist_ok=False, parents=True)
    save(directory / 'schedules.json', schedule)
    experimental = ('pong_actuator_response.py', 'pong_timing_fast.py', 'pong_timing_controller.py',
                    'pong_rapid_motor_recovery.py', 'pong_transfer_arena.py')
    files = [Path(__file__).resolve(), *(Path(__file__).with_name(name) for name in experimental),
             directory / 'schedules.json', checkpoint]
    spec = dict(contract='paired-native-visual-response-time-transfer-v1',
                source=source_signature(), hashes={str(path): sha(path) for path in files},
                prior_seed_exclusions=excluded, arms=ARMS, duration_seconds=DURATION,
                response_carry='both arms carry timing and order within each condition and through slowdown/restoration',
                controller_inputs='public entities, past requests, elapsed camera time, ordinary own-hit/terminal feedback',
                outcome='first own opportunities, later returns, paired action hashes, timing and response diagnostic',
                phases=PHASES)
    save(directory / 'spec.json', spec)
    return schedule, spec


def run_arm(directory, schedule, acquired, arm):
    torch.set_num_threads(1)
    policy = interception_policy()
    learner = VisualButtonEffects.from_state(acquired)
    state = None
    rows = []
    for game in schedule:
        if game['reset_model']:
            state = None
        controller = ActuatorResponseController(policy, learner, timing_state=state,
            response_updates=arm == 'live')
        row = play(controller, game['seed'], activation_seconds=game['seconds'],
                   delay_frames=game['delay'], duration=DURATION)
        state = controller.state()
        rows.append(dict(**row, phase=game['phase'], end_state=state,
                         response_changes=controller.response_changes))
        save(directory / f'{arm}.json', rows)
        print(arm, game['phase'], game['seed'], row['hits'], row['opportunities'],
              state['response_seconds'], state['delay_commands'], flush=True)


def metrics(rows):
    contacts = [row['contacts'] for row in rows if row['contacts']]
    later = [hit for trajectory in contacts for _, hit in trajectory[1:]]
    return dict(first_returns=sum(bool(group[0][1]) for group in contacts),
                first_opportunities=len(contacts), later_returns=sum(later),
                later_opportunities=len(later), transition_cameras=sum(row['cameras'] for row in rows),
                initial_cameras=len(rows), final_response_seconds=[r['end_state']['response_seconds'] for r in rows],
                final_delays=[r['end_state']['delay_commands'] for r in rows])


def run(directory, checkpoint):
    model_bytes()
    schedule, spec = prepare(directory, checkpoint)
    acquired = read(checkpoint)
    start = monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_arm, directory, schedule, acquired, arm) for arm in ARMS]
        for job in jobs:
            job.result()
    receipts = {arm: read(directory / f'{arm}.json') for arm in ARMS}
    phases = {}
    for phase, _, _, count in PHASES:
        by_arm = {arm: [row for row in receipts[arm] if row['phase'] == phase] for arm in ARMS}
        pairs = list(zip(*(by_arm[arm] for arm in ARMS), strict=True))
        assert len(pairs) == count and all(a['seed'] == b['seed'] for a, b in pairs)
        first_hits = [(bool(a['contacts'] and a['contacts'][0][1]),
                       bool(b['contacts'] and b['contacts'][0][1])) for a, b in pairs]
        phases[phase] = dict(**{arm: metrics(rows) for arm, rows in by_arm.items()},
            live_only_first=first_hits.count((True, False)),
            frozen_only_first=first_hits.count((False, True)),
            identical_games=sum(a['action_sha256'] == b['action_sha256'] for a, b in pairs))
    unchanged = (source_signature() == spec['source'] and
                 all(sha(Path(path)) == expected for path, expected in spec['hashes'].items()))
    assert unchanged
    report = dict(phases=phases, sources_unchanged=True, wall_seconds=monotonic() - start,
                  spec_sha256=sha(directory / 'spec.json'))
    save(directory / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.checkpoint.resolve()), sort_keys=True))
