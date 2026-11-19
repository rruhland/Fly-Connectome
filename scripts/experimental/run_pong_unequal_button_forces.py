"""Compare live and frozen transfer to asymmetric, nonzero actuator commands."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import MODEL_HASH, model_bytes
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
from run_pong_surprise_recovery import POLICY, exclusions
import run_pong_inverse_cycle as native

CONDITIONS = {
    'original-frozen': ((-1., 0., 1.), False),
    'unequal-frozen': ((.88, -.07, -.74), False),
    'unequal-live': ((.88, -.07, -.74), True),
}


def play_condition(directory, name, acquired, seeds):
    torch.set_num_threads(1)
    forces, learning = CONDITIONS[name]
    policy = native.PersistentRelationalPolicy(native.RelationalPolicy(
        *POLICY['choices'], POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    model = VisualButtonEffects.from_state(acquired)
    if not learning:
        model = model.frozen_branch()
    applied_state = model.state()
    native.save(directory / f'{name}-start.json', applied_state)

    def camera(seed, **kwargs):
        instance = PongCamera(seed, **kwargs)
        instance.actions = forces
        return instance

    rows = []
    with patch.object(native, 'PongCamera', camera), patch.object(native, 'CategoricalMotorController', VisualIntentController):
        for seed in seeds:
            out = native.play(model, policy, seed, [0, 1, 2])
            rows.append(dict(**out, calibrations=model.completed_calibrations, probe_frames=model.dither_count))
            native.save(directory / f'{name}-games.json', rows)
    if not learning:
        assert model.state() == applied_state
    native.save(directory / f'{name}-end.json', model.state())


def run(checkpoint, directory):
    model_bytes()
    used, _ = exclusions(directory.parent)
    rng = np.random.default_rng(66430298)
    seeds = []
    while len(seeds) < 24:
        draw = int(rng.integers(1, 2147483647))
        if draw not in used and ((draw * 48271 + 1) % 2147483647) % 2 == len(seeds) % 2:
            seeds.append(draw)
            used.add(draw)
    directory.mkdir(parents=True, exist_ok=False)
    native.save(directory / 'spec.json', dict(
        conditions=CONDITIONS, starts=seeds, checkpoint=native.sha(checkpoint),
        script=native.sha(Path(__file__)), learner=native.sha(Path(__file__).with_name('pong_visual_button_effects.py')),
        asset=MODEL_HASH, measurement='Full and final 12 of 24 games. First and later opportunities, matched frozen and original. One exploratory stream.'))
    acquired = json.loads(checkpoint.read_text())
    begin = time.monotonic()
    with ProcessPoolExecutor(3) as workers:
        jobs = [workers.submit(play_condition, directory, name, acquired, seeds) for name in CONDITIONS]
        for job in jobs:
            job.result()
    total = {}
    for name in CONDITIONS:
        rows = json.loads((directory / f'{name}-games.json').read_text())
        total[name] = {}
        for label, group in [('all', rows), ('final12', rows[-12:])]:
            hits = sum(r['returns'] for r in group)
            opp = sum(r['opportunities'] for r in group)
            first = sum(r['events'][0]['hit'] for r in group)
            total[name][label] = dict(first=[first,len(group)], later=[hits-first,opp-len(group)], total=[hits,opp])
    native.save(directory / 'summary.json', dict(outcomes=total, wall_seconds=time.monotonic()-begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.directory)
