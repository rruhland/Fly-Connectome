"""Replay scored cursor sessions that physically changed the motor wiring twice."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _read
from .tt6_body_arena import TT6Wiring
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, BLOCKS_PER_TARGET, TargetArena
from .tt10_target_trial import SOURCES


def replay(path):
    data = _read(path)
    source = _read(ROOT / 'artifacts/tt9' / (SOURCES[data['site']][1] + '.json.gz'))
    arena = TargetArena(**dict(source['arena'], render_mode='projected'))
    ticks = BLOCK * BLOCKS_PER_TARGET
    count = 0
    try:
        for row in data.get('entries', data.get('rows')):
            seed = row['seed']
            native = source['installation_seed']
            for name, run in row.get('runs', row.get('trials')).items():
                if 'points' in run:
                    scores = [point['score'] for point in run['points']]
                    tape = [a for point in run['points'] for a in point['actions']]
                else:
                    scores, tape = run['scores'], run['actions']
                actions = np.asarray(tape)
                if (len(scores) != 16 or actions.shape != (16 * ticks, arena.action_size)
                        or not np.isfinite(actions).all() or np.max(abs(actions)) > 1.):
                    raise ValueError(f'Invalid tape for {name}, seed {seed}')
                wiring = DelayedWiring(TT6Wiring(arena, native),
                                       Disturbance(delays=((0, 0), (90, 2), (240, 1)),
                                                   noise=((0, 0.), (90, .025))), seed=seed)
                wiring.reset(seed=seed, render=False)
                for i, (goal, expected) in enumerate(zip(row['goals'], scores, strict=True)):
                    if i in (4, 10):
                        wiring.wiring = TT6Wiring(arena, seed if i == 4 else native)
                    arena.set_target(goal)
                    observed = sum(wiring.step(a, render=False)[1] for a in actions[i*ticks:(i+1)*ticks])
                    if observed != expected:
                        raise ValueError(f'{name}, seed {seed}, target {i}: {observed} != {expected}')
                count += 1
    finally:
        arena.close()
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', nargs='+', type=Path)
    print(sum(replay(path) for path in parser.parse_args().receipts), 'full sessions physically replayed')
