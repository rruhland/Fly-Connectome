"""Build a portable image-calibrated wrist checkpoint and replay it during body changes."""
import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_continual_wrist import ContinualWrist
from .tt24_current_pause import arm_group, execute, replay, scene
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, episode
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt24_wrist_rim import usable

CHECKPOINT = ROOT / 'wrist-rim-policy.npz'


def build(fit):
    indices, normals = [], []
    for index, name in enumerate(('stock', 'alternate')):
        record = fit[name]
        if usable(record):
            indices.append(index)
            normals.append(record['fitted_link_normal'])
    with np.load(BUNDLE, allow_pickle=False) as source:
        np.savez_compressed(CHECKPOINT, **source, wrist_rim_indices=indices, wrist_rim_normals=normals)


def main():
    build(_read(ROOT / 'wrist-rim.json.gz'))
    course = _read(ROOT / 'wrist-rim-course.json.gz')
    source = _read(ROOT / 'single-arm-course-1.json.gz')
    previous = _read(ROOT / 'current-pause-stream-1.json.gz')
    server = BounceArena(view='none')
    arms = arm_group()
    final_checkpoint = ROOT / 'wrist-rim-continual.npz'
    try:
        # Reproduce independent live-game decisions with the standalone artifact.
        for name in ('stock', 'alternate'):
            case = next(r for r in course['rows'] if r['plan']['body'] == name)
            plan = case['plan']
            physical = InstalledArm(name)
            try:
                server.reset(seed=plan['launcher_seed'], split='held', render=False)
                launch = [v.copy() for v in (server._ball_position, server._ball_velocity, server._ball_spin)]
                policy = ToolEquilibrium(CHECKPOINT, .0557)
                policy.mode = 'workspace_local'
                row = episode(physical, policy, plan, launch, 'bundle')
                expected = case['modes']['measured']
                if digest_json(row) != digest_json(expected):
                    raise ValueError(f'packaged policy changed {name}: {[k for k in row if digest_json(row[k]) !=digest_json(expected[k])]}')
            finally:
                physical.arena.close()
        learner = ContinualWrist(CHECKPOINT)
        rows = []
        for record, baseline in zip(source['rows'], previous['rows'], strict=True):
            fixture = scene(arms, source, record, server)
            played, _ = execute(fixture, learner)
            replay(fixture, played)
            learner.save_between_releases(final_checkpoint)
            learner = ContinualWrist(final_checkpoint)
            rows.append(dict(plan=record['plan'], baseline=baseline['current']['metrics'], current=played))
            _write(ROOT / 'wrist-rim-continual.json.gz', dict(rows=rows, replayed=len(rows)))
        print([(r['baseline']['rebounds'], r['current']['metrics']['rebounds']) for r in rows], flush=True)
    finally:
        server.close()
        for arm in arms.values():
            arm.arena.close()


if __name__ == '__main__':
    main()
