"""Measure immediate transfer and online adaptation to faster native Pong."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes
from fly_connectome.pong import Pong
from pong_motor_recalibration import MotorGainController, MotorGainLearner
from run_pong_motor_recalibration import POLICY
import run_pong_inverse_cycle as native


def run(checkpoint, destination):
    model_bytes()
    torch.set_num_threads(1)
    rng = np.random.default_rng(71619021)
    seeds = []
    while len(seeds) < 12:
        seed = int(rng.integers(1, 2147483647))
        if ((seed * 48271 + 1) % 2147483647) % 2 == len(seeds) % 2 and seed not in seeds:
            seeds.append(seed)

    schedules = json.loads(checkpoint.with_name('schedules.json').read_text())
    assert set(seeds).isdisjoint(game['seed'] for replica in schedules for game in replica['games'])
    conditions = [('normal_frozen', .55, False), ('faster_frozen', .67, False), ('faster_live', .67, True)]
    destination.mkdir(parents=True, exist_ok=False)
    native.save(destination / 'spec.json', dict(
        seeds=seeds, conditions=conditions, checkpoint=native.sha(checkpoint),
        script=native.sha(Path(__file__)), games=36, max_transition_cameras=36000,
        measure='All 12 games in each condition. Compare immediate transfer with learning on the same starts.'))
    acquired = json.loads(checkpoint.read_text())
    policy = native.PersistentRelationalPolicy(native.RelationalPolicy(
        *POLICY['choices'], POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])

    rows = []
    started = time.monotonic()
    for name, speed, learn in conditions:
        def camera(seed, **kwargs):
            environment = PongCamera(seed, **kwargs)
            # Apply the new speed starting with the opening serve.
            environment.game = Pong([seed], replace(environment.game.config, ball_speed=speed))
            return environment

        model = MotorGainLearner.from_state(acquired)
        if not learn:
            model = model.frozen_branch()
        with patch.object(native, 'PongCamera', camera), patch.object(native, 'CategoricalMotorController', MotorGainController):
            for seed in seeds:
                outcome = native.play(model, policy, seed, [0, 1, 2])
                rows.append(dict(condition=name, **outcome))
                native.save(destination / 'games.json', rows)
        if not learn:
            assert model.updates == acquired['updates'] and model.state()['theta'] == acquired['theta']
        results = [r for r in rows if r['condition'] == name]
        print(name, sum(r['returns'] for r in results), sum(r['opportunities'] for r in results), flush=True)
    native.save(destination / 'cost.json', dict(
        games=len(rows), transitions=sum(r['transitions'] for r in rows), wall_seconds=time.monotonic() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.directory)
