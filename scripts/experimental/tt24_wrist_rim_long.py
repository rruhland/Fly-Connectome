"""Freeze the image-based tool estimate and test two-minute play on six new changed-arm starts."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_long_wrist import CAP, play
from .tt24_wrist_rim_course import RimWrist


def case(item):
    i, plan = item
    rng = np.random.default_rng(649727001 + i)
    active = int(rng.integers(190, 300))
    changes = []
    while active < CAP:
        g = rng.uniform(.78, .92) if len(changes) % 2 == i % 2 else rng.uniform(1.11, 1.25)
        changes.append((active, g))
        active += int(rng.integers(270, 491))
    plan['changes'] = changes
    fit = _read(ROOT / 'wrist-rim.json.gz')
    arm = InstalledArm('alternate')
    arm.arena.max_steps = CAP
    server = BounceArena(view='none')
    try:
        server.reset(seed=plan['launcher_seed'], split='held', render=False)
        launch = [v.copy() for v in (server._ball_position, server._ball_velocity, server._ball_spin)]
        records = {}
        for mode in ('original', 'measured'):
            policy = RimWrist(fit, mode)
            row = play(arm, policy, plan, launch, 'live')
            repeated = play(arm, policy, plan, launch, 'live', row['actions'])
            if digest_json(row) != digest_json(repeated):
                raise ValueError('long-horizon replay changed')
            records[mode] = row
        _write(ROOT/f'wrist-rim-long-{i}.json.gz', dict(plan=plan, results=records))
        print(i, [(name, r['metrics']['longest_consecutive_rebounds'], len(r['actions']))
                  for name, r in records.items()], flush=True)
    finally:
        server.close()
        arm.arena.close()


if __name__ == '__main__':
    plans = [p for p in schedule(1476852231, 6) if p['body'] == 'alternate']
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(case, enumerate(plans)))
