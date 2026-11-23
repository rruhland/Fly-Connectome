"""Move the drive learned in native Pong onto a second physical arm and camera."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, episode, schedule
from .tt24_motor_bank import I, decoder
from .tt24_pong_arm_local import fit
from .tt24_tool_equilibrium import ToolEquilibrium


class DrivePort:
    """The simulator applies its own wiring after the learner issues a bounded request."""
    def __init__(self, wrist, estimate, installation):
        self.wrist = wrist
        self.inverse = decoder(estimate)
        self.installation = installation

    def begin_release(self, obs):
        self.wrist.begin_release(obs)
        self.calibration_index = self.wrist.calibration_index

    def act(self, obs):
        command = np.clip(self.inverse @ self.wrist.act(obs), -1., 1.)
        return np.clip(self.installation @ command, -1., 1.)


def run(i):
    path = ROOT/f'learned-tip-transfer-{i}.json.gz'
    parent = _read(path)
    if not parent['usable']:
        raise ValueError('native physical source did not qualify')
    early = fit(parent['clips'][:4])
    matrices = {k: np.array([m['matrix_row'] for m in v])
                for k, v in [('four_points', early), ('twelve_points', parent['fits']['source'])]}
    wiring = np.array(parent['wiring'])
    matrices.update(stale=I, known=wiring)
    plans = [p for p in schedule(941721551 + i, 6) if p['body'] == 'alternate']
    arm = InstalledArm('alternate')
    server = BounceArena(view='none')
    try:
        wrist = ToolEquilibrium(BUNDLE, .0557)
        wrist.mode = 'workspace_local'
        data = dict(source=path.name, source_learning_sha256=digest_json({k:parent[k] for k in ("wiring","clips","fits")}), first_four_motor_map=matrices['four_points'].tolist(),
                    source_physical_seconds={n:sum(len(c['commands']) for c in parent['clips'][:n])/60 for n in (4,12)},
                    rows=[])
        for p in plans:
            server.reset(seed=p['launcher_seed'], split='held', render=False)
            launch = [v.copy() for v in (server._ball_position, server._ball_velocity, server._ball_spin)]
            conditions = {}
            for label, matrix in matrices.items():
                port = DrivePort(wrist, matrix, wiring)
                row = episode(arm, port, p, launch, 'bundle')
                again = episode(arm, port, p, launch, 'bundle', replay=row['actions'])
                if any(digest_json(row[key]) != digest_json(again[key]) for key in ('actions','metrics','changes_reached')):
                    raise ValueError('secondary arm physical replay changed')
                conditions[label] = row
            data['rows'].append(dict(plan=p, conditions=conditions))
            _write(ROOT/f'learned-tip-body-{i}.json.gz', data)
        print(i, data['source_physical_seconds'], {label:[r['conditions'][label]['metrics']['longest_consecutive_rebounds']
                                              for r in data['rows']] for label in matrices}, flush=True)
    finally:
        server.close()
        arm.arena.close()


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (0,1)))
