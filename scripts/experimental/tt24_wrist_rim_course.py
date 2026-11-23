"""Blind held wrist-game comparison after estimating the normal from the onboard image."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import CAP, InstalledArm, episode, schedule
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt24_wrist_rim import usable


class RimWrist(ToolEquilibrium):
    def __init__(self, fit, mode):
        super().__init__(BUNDLE, .0557)
        self.mode = 'workspace_local'
        self.perception_mode = mode
        self.fit = fit

    def begin_release(self, observation):
        super().begin_release(observation)
        name = ('stock', 'alternate')[self.calibration_index]
        record = self.fit[name]
        if not usable(record) or self.perception_mode == 'original':
            return
        decoder = self.sensor.decoder
        local = np.array(record['fitted_link_normal'])
        if self.perception_mode == 'reverse':
            local = 2 * decoder.normal - local
            local /= np.linalg.norm(local)
        self.face_normals[self.calibration_index] = local



def case(plan):
    fit = _read(ROOT / 'wrist-rim.json.gz')
    arm = InstalledArm(plan['body'])
    arm.arena.max_steps = CAP
    server = BounceArena(view='none')
    try:
        server.reset(seed=plan['launcher_seed'], split='held', render=False)
        launch = [x.copy() for x in (server._ball_position, server._ball_velocity, server._ball_spin)]
        records = {}
        for mode in ('original', 'measured', 'reverse'):
            policy = RimWrist(fit, mode)
            row = episode(arm, policy, plan, launch, 'bundle')
            replay = episode(arm, policy, plan, launch, 'bundle', replay=row['actions'])
            if any(digest_json(row[k]) != digest_json(replay[k]) for k in ('metrics', 'actions', 'changes_reached')):
                raise ValueError('physical action replay failed')
            records[mode] = row
        if plan['body'] == 'stock' and len({digest_json(x['actions']) for x in records.values()}) != 1:
            raise ValueError('censored rim changed the original stock controller')
        return dict(plan=plan, modes=records)
    finally:
        server.close()
        arm.arena.close()


def main():
    whole = schedule(874009231, 12)
    roster = [p for p in whole if p['body'] == 'alternate'] + [p for p in whole if p['body'] == 'stock'][:4]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(case, roster))
    _write(ROOT / 'wrist-rim-course.json.gz', dict(roster=roster, rows=rows, cap_ticks=CAP))
    for body in ('alternate', 'stock'):
        for mode in ('original', 'measured', 'reverse'):
            selected = [r['modes'][mode]['metrics'] for r in rows if r['plan']['body'] == body]
            scores = [r['longest_consecutive_rebounds'] for r in selected]
            print(body, mode, scores, sum(scores), sum(x >= 10 for x in scores), flush=True)


if __name__ == '__main__':
    main()
