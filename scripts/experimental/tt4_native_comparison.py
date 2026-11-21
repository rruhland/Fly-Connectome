"""Audit paired RGB rendering and tracking runs on the same MuJoCo installations."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write


def _mode_free(geometry):
    return {**geometry, 'arena': {key: value for key, value in geometry['arena'].items()
                                  if key != 'render_mode'}}


def compare(first_path, second_path, *, temporal=False):
    first, second = _read(first_path), _read(second_path)
    modes = tuple(report['geometry']['arena']['render_mode'] for report in (first, second))
    if temporal:
        if modes != ('mujoco', 'mujoco') or first['schema'] != 'tt4-paired-native-temporal-v1':
            raise ValueError('expected two native RGB runs with the temporal tracker first')
        compared = ['native_temporal', 'native_local_regions']
    else:
        if modes != ('mujoco', 'projected'):
            raise ValueError('expected native MuJoCo RGB followed by projected RGB')
        compared = ['native_mujoco', 'projected_rgb']
    first_geometry, second_geometry = map(_mode_free, (first['geometry'], second['geometry']))
    for key in ('arena', 'probe_horizon'):
        if first_geometry[key] != second_geometry[key]:
            raise ValueError('comparison requires identical arena geometry and probe timing')
    if not temporal and first_geometry != second_geometry:
        raise ValueError('renderer comparison requires identical controller settings')
    for key in ('schedules', 'conditions', 'inputs_sha256'):
        if first[key] != second[key]:
            raise ValueError(f'comparison requires matching {key}')
    report = {}
    for view, conditions in first['conditions'].items():
        pairs = list(zip(first['courses'][view], second['courses'][view], strict=True))
        for a, b in pairs:
            for key in ('schedule', 'evaluator_wiring', 'calibration'):
                if a[key] != b[key]:
                    raise ValueError(f'comparison requires matching {key}')
        arms = {}
        for condition in conditions:
            points = []
            for a, b in pairs:
                examples = zip(a['runs'][condition]['points'], b['runs'][condition]['points'], strict=True)
                for left, right in examples:
                    if left['seed'] != right['seed']:
                        raise ValueError('comparison requires matching point seeds')
                    common = min(left['steps'], right['steps'])
                    changes = np.abs(np.array(left['actions'][:common]) - np.array(right['actions'][:common]))
                    different = np.flatnonzero(changes.max(axis=1) > 1e-8)
                    points.append(dict(
                        seed=left['seed'], shared_steps=common, differing_steps=len(different),
                        first_action_difference=int(different[0]) if len(different) else None,
                        action_mean_absolute_difference=float(changes.mean()),
                        legal_returns=[left['player_returns'], right['player_returns']],
                        actor_frames=[left['actor_frames'], right['actor_frames']],
                        relevant_frames=[left['relevant_frames'], right['relevant_frames']],
                        approach_frames=[left['approach_frames'], right['approach_frames']],
                    ))
            totals = {key: np.sum([point[key] for point in points], axis=0).tolist()
                      for key in ('legal_returns', 'actor_frames', 'relevant_frames', 'approach_frames')}
            arms[condition] = dict(
                total=totals, points=points, steps_compared=sum(p['shared_steps'] for p in points),
                action_steps_changed=sum(p['differing_steps'] for p in points),
                points_with_changed_actions=sum(p['first_action_difference'] is not None for p in points),
            )
        report[view] = dict(
            conditions=arms,
            estimated_ready_pose=[[left['adapter']['ready'], right['adapter']['ready']] for left, right in pairs],
            median_visual_fit_error=[[left['adapter']['median_fit_error'], right['adapter']['median_fit_error']]
                                     for left, right in pairs],
        )
    return dict(schema='tt4-paired-renderer-audit-v1', compared_order=compared,
                receipts_sha256={path.name: _hash(path) for path in (first_path, second_path)},
                geometry=first_geometry, views=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--temporal', action='store_true')
    args = parser.parse_args()
    _write(args.output, compare(args.first, args.second, temporal=args.temporal))


if __name__ == '__main__':
    main()
