"""Run a bounded proprioceptive learner during Pong and switch to fresh wrist drops."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM
from .tt24_motor_bank import I
from .tt24_online_tip_transfer import point
from .tt24_pong_arm import bounce
from .tt24_pong_arm_tip import Joystick, plan
from .tt24_rolling_motor import RollingMotorLearner


def run(i):
    torch.set_num_threads(1)
    asset = _read(ROOT/f'learned-tip-transfer-{i}.json.gz')
    wiring = np.array(asset['wiring'])
    native = np.array(_read(WARM)['weights'])
    starts, _ = plan(Joystick(), count=18, seed=80412718+i)
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    learner = RollingMotorLearner()
    checkpoint = ROOT/f'rolling-motor-{i}.npz'
    try:
        reports = []
        with TemporaryDirectory() as tmp:
            for number, ((_, pose), expected) in enumerate(zip(starts[:14], asset['clips'][:14], strict=True), 1):
                # The harness uses old tapes exclusively to assert every incoming event, never to train.
                frames = point(arena, Joystick(), pose, wiring, native, expected, learner)
                start = perf_counter()
                fitted = learner.end()
                elapsed = perf_counter() - start
                reports.append(dict(point=number, frames=frames, fitted=fitted, fit_seconds=elapsed,
                                    blocks=len(learner.blocks), encoder_map=learner.map.tolist()))
                if number == 6:
                    temp = Path(tmp)/'memory.npz'
                    learner.save(temp)
                    learner = RollingMotorLearner.load(temp)
                if number == 12:
                    reference = np.array([r['matrix_row'] for r in asset['fits']['source']])
                    if not np.array_equal(reference, learner.map):
                        raise ValueError('rolling fit does not reproduce the qualified estimate')
                    learner.save(checkpoint)
                if number >= 13 and (len(learner.blocks) != 12 or learner.completed != number):
                    raise ValueError('rolling learner stopped updating or exceeded its capacity')
        model = RollingMotorLearner.load(checkpoint)
        policy = CarriedWristPolicy()
        rng = np.random.default_rng(578311 + i)
        rows = []
        for seed in rng.integers(1, 2**31-1, 8).tolist():
            scenarios = {}
            for label, matrix in [('learned', model.map), ('stale', I)]:
                game = bounce(arena, policy, seed, wiring, matrix)
                repeated = bounce(arena, policy, seed, wiring, matrix, tape=game['actions'])
                if digest_json(game) != digest_json(repeated):
                    raise ValueError('closed-loop target score or action tape changed')
                scenarios[label] = game
            rows.append(dict(seed=seed, conditions=scenarios))
        result = dict(source=reports, portable=checkpoint.name, rows=rows, exact_fit_after_12=True,
                      trained_control_ticks=sum(r['frames'] for r in reports[:12]))
        _write(ROOT/f'rolling-motor-trial-{i}.json.gz', result)
        print(i, [(r['conditions']['learned']['metrics']['longest_consecutive_rebounds'],
                   r['conditions']['stale']['metrics']['longest_consecutive_rebounds']) for r in rows],flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,(0,1)))
