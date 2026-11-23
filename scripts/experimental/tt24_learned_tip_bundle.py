"""Package the trained Pong readout, bouncing readout and observed drive without evaluator matrices."""
import numpy as np
import torch

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_continual_wrist import ContinualWrist
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM
from .tt24_learned_tip_transfer import native_point
from .tt24_motor_bank import MotorBank, decoder
from .tt24_pong_arm import bounce
from .tt24_pong_arm_tip import Joystick, plan


def main():
    torch.set_num_threads(1)
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    checks = []
    try:
        source_head = _read(WARM)['weights']
        for i in range(2):
            report = _read(ROOT/f'learned-tip-transfer-{i}.json.gz')
            learned = np.array([x['matrix_row'] for x in report['fits']['source']])
            destination = ROOT/f'learned-tip-learner-{i}.npz'
            learner = ContinualWrist(BUNDLE)
            learner.bank = MotorBank([np.eye(5), learned])
            learner.bank.selected = 1
            learner.bank.inverse = decoder(learned)
            learner.save_between_releases(destination)
            with np.load(destination, allow_pickle=False) as archive:
                payload = dict(archive)
            np.savez_compressed(destination, **payload, native_pong_readout=source_head)
            saved = ContinualWrist(destination)
            if saved.bank.selected != 1 or not np.array_equal(saved.bank.maps[1], learned):
                raise ValueError('motor memory changed on load')
            # The physical plant belongs to evaluation. The checkpoint contains only its learned estimate.
            wiring = np.array(report['wiring'])
            first = report['rows'][0]
            actual = bounce(arena, CarriedWristPolicy(destination), first['seed'], wiring, saved.bank.maps[saved.bank.selected])
            if digest_json(first['conditions']['source']) != digest_json(actual):
                raise ValueError('packaged policy changed the receiving wrist game')
            joystick = Joystick()
            starts, _ = plan(joystick, count=18, seed=80412718 + i)
            with np.load(destination, allow_pickle=False) as archive:
                native = archive['native_pong_readout'].copy()
            replayed = []
            for (seed, pose), old in zip(starts[12:], report['clips'][12:], strict=True):
                repeated = native_point(arena, joystick, seed, pose, wiring, native)
                if digest_json(old) != digest_json(repeated):
                    raise ValueError('source readout did not retain its behavior after wrist play')
                replayed.append(seed)
            checks.append(dict(file=destination.name, wrist_actions=len(actual['actions']), seed=first['seed'],
                               retained_pong_seeds=replayed))
        _write(ROOT/'learned-tip-reload.json.gz', dict(checks=checks))
    finally:
        arena.close()


if __name__ == '__main__':
    main()
