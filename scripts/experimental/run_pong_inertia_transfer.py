"""Test a delay-adaptive policy on a slower actuator without a command queue."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pong_timing_controller import TimedMotorController
from pong_transfer_arena import interception_policy, play
from pong_visual_button_effects import VisualButtonEffects
from run_pong_inverse_cycle import save, sha


def run_arm(root, acquired, seeds, live):
    torch.set_num_threads(1)
    name = 'live' if live else 'timing_frozen'
    policy = interception_policy()
    motor = VisualButtonEffects.from_state(acquired)
    state = None
    rows = []
    for i, seed in enumerate(seeds):
        slow = i < 12
        controller = TimedMotorController(policy, motor, timing_state=state, timing_updates=live)
        result = play(controller, seed, activation_seconds=.32 if slow else .08, duration=15.)
        state = controller.state()
        rows.append(dict(phase='slower' if slow else 'restored', **result,
                         delay=state['delay_commands'], order=state['motor_order']))
        save(root / f'{name}.json', rows)
    save(root / f'{name}-final.json', state)


def run(root, checkpoint):
    root.mkdir(exist_ok=False,parents=True)
    seeds = []
    rng = np.random.default_rng(81973475)
    while len(seeds) < 18:
        seed = int(rng.integers(1,2147483647))
        if seed not in seeds and ((seed*48271+1)%2147483647)%2 == len(seeds)%2:
            seeds.append(seed)
    save(root/'spec.json',dict(seeds=seeds, checkpoint_sha=sha(checkpoint), tau=[.32,.08],
        phase_games=[12,6], delay_frames=0, comparison='same motor learning, timing live vs frozen',
        outcome='all returns, first returns and end-lag estimates; descriptive off-model transfer'))
    acquired = json.loads(checkpoint.read_text())
    with ProcessPoolExecutor(2) as pool:
        jobs=[pool.submit(run_arm,root,acquired,seeds,live) for live in (True,False)]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    run(args.directory,args.checkpoint)
