"""Can a reward-trained Pong player identify five coupled actuators from ordinary joystick play?"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import expm
import torch

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM, arcade
from .tt24_motor_bank import I
from .tt24_pong_arm import NATIVE_CAP, bounce
from .tt24_pong_arm_local import crosscheck, fit
from .tt24_pong_arm_tip import Joystick, PhysicalTipPaddle, plan


class NativeTip(PhysicalTipPaddle):
    def step(self, drive):
        super().step(-drive)


def native_point(arena, joystick, seed, pose, wiring, weights):
    paddle = None
    def attach(spec):
        nonlocal paddle
        paddle = NativeTip(arena, joystick, pose, wiring, spec)
        return paddle
    score, hits, ticks = arcade(weights, [seed], cap=NATIVE_CAP, opponent=.35, paddle_factory=attach)[0, 0]
    return dict(seed=seed, score=score, hits=hits, ticks=int(ticks), commands=paddle.commands, encoders=paddle.traces)


def run(index):
    torch.set_num_threads(1)
    rng = np.random.default_rng(29845128 + index)
    skew = rng.normal(size=(5, 5))
    skew -= skew.T.copy()
    wiring = expm(skew * ((.80 + .2 * index) / np.linalg.norm(skew, 2))) @ np.diag(rng.uniform(.60, 1.14, 5))
    targets = rng.integers(1, 2**31 - 1, 8).tolist()
    archive = _read(WARM)
    if not archive['gate']:
        raise ValueError('missing qualified native player')
    weights = archive['weights']
    joystick = Joystick()
    starts, _ = plan(joystick, count=18, seed=80412718+index)
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    report = dict(wiring=wiring.tolist(), target_seeds=targets, native_checkpoint_sha256=_hash(WARM))
    output = ROOT/f'learned-tip-transfer-{index}.json.gz'
    try:
        clips = [native_point(arena, joystick, seed, pose, wiring, weights) for seed, pose in starts]
        zero = [native_point(arena, joystick, seed, pose, wiring, np.zeros_like(weights)) for seed, pose in starts[12:]]
        for (seed, pose), expected in zip(starts[12:], clips[12:], strict=True):
            if digest_json(native_point(arena, joystick, seed, pose, wiring, weights)) != digest_json(expected):
                raise ValueError('held native actions or encoder readings changed')
        fits = {name: fit(clips[:12], shuffle=name=='shuffled') for name in ('source', 'shuffled')}
        checks = {k: crosscheck(v, clips[12:]) for k, v in fits.items()}
        maps = {k: np.array([x['matrix_row'] for x in v]) for k, v in fits.items()}
        counts = [sum(c['hits'] for c in g) for g in (clips[12:], zero)]
        usable = bool(counts[0] > counts[1] and all(
            x['p70_rad'] < x['stale_p70_rad'] and x['singular_values'][-1] > .005 * x['singular_values'][0]
            for x in checks['source']))
        report.update(clips=clips, zero=zero, fits=fits, validation=checks, usable=usable)
        _write(output, report)
        print(index, 'native', counts, 'usable', usable,
              'matrix error', np.linalg.norm(maps['source']-wiring)/np.linalg.norm(wiring), flush=True)
        if not usable:
            return
        policy = CarriedWristPolicy()
        # Compare four independent live feedback trajectories on identical releases.
        maps.update(stale=I, known=wiring)
        report['rows'] = []
        for seed in targets:
            conditions = {}
            for name, matrix in maps.items():
                row = bounce(arena, policy, seed, wiring, matrix)
                replayed = bounce(arena, policy, seed, wiring, matrix, tape=row['actions'])
                if digest_json(row) != digest_json(replayed):
                    raise ValueError('physical target replay changed')
                conditions[name] = row
            report['rows'].append(dict(seed=seed, conditions=conditions))
            _write(output, report)
        print(index, {n:[r['conditions'][n]['metrics']['longest_consecutive_rebounds'] for r in report['rows']]
                      for n in maps}, flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (0, 1)))
