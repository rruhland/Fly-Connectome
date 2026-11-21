"""Measure how physical wrist mount and aim affect precontact native RGB visibility."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_so101 import CAMERA_BODY, GRIP_BODY, WRIST_CAMERA
from .tt20_fixture import TT20Arena
from .tt20_fixture_audit import JAW, SOURCE, _course, _courses, _fixture, _mask_stat, _mesh_id, _summarize


class AimTrialArena(TT20Arena):
    def __init__(self, *args, elevation=0., trial_fovy=None, **kwargs):
        self.trial_elevation = elevation
        super().__init__(*args, **kwargs)
        if trial_fovy is not None:
            self.model.camera(WRIST_CAMERA).fovy[0] = self.wrist_fovy = trial_fovy

    def _orient_wrist(self):
        super()._orient_wrist()
        if self.trial_elevation:
            board = self.model.body(CAMERA_BODY)
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, board.quat)
            local = Rotation.from_euler('x', self.trial_elevation, degrees=True).as_matrix()
            rotation = matrix.reshape(3, 3) @ local
            mujoco.mju_mat2Quat(board.quat, np.ascontiguousarray(rotation).ravel())
            mujoco.mj_forward(self.model, self.data)


def _objects(arena, ids, *, geometry_only):
    eye, xyz = arena.data.camera(WRIST_CAMERA).xpos, arena.data.camera(WRIST_CAMERA).xmat
    axes = xyz.reshape(3, 3)
    focal = arena.lens_intrinsics()['wrist']['focal']
    center = np.array(arena.lens_intrinsics()['wrist']['center'])
    shape = np.array((arena.width, arena.height))
    positions = dict(ball=arena.ball.copy())
    positions.update({role: arena.data.geom_xpos[index] for role, index in ids.items() if role != 'ball'})
    stats = {}
    for role, target in positions.items():
        offset = target - eye
        direction = offset @ axes
        depth = -direction[2]
        pixel = direction[:2] * (1., -1.) * focal / max(.001, depth) + center
        inside = depth > 0 and np.all(pixel > 0) and np.all(pixel < shape)
        hit = np.empty(1, dtype=np.int32)
        distance = mujoco.mj_ray(arena.model, arena.data, eye, offset, None, True, -1, hit)
        nearest = None if hit[0] < 0 else (arena.model.geom(hit[0]).name or arena.model.body(arena.model.geom_bodyid[hit[0]]).name)
        stats[role] = dict(center_inside=bool(inside), center_forward=bool(depth > 0.),
                           center_projection_px=pixel.tolist(), first_hit=nearest,
                           first_hit_same=bool(hit[0] == ids[role] or .99 <= distance <= 1.01))
        if geometry_only and role.endswith('jaw'):
            model, data = arena.model, arena.data
            geom = ids[role]
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            points = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            transform = data.geom_xmat[geom].reshape(3, 3)
            positions_in_view = (points @ transform.T + target - eye) @ axes
            depths = -positions_in_view[:, 2]
            projections = positions_in_view[:, :2] * (1., -1.) * focal / np.maximum(.001, depths[:, None]) + center
            stats[role]['outline_inside'] = bool(np.all(depths > 0.) and np.all(projections > 0.)
                                                  and np.all(projections < shape))
    if geometry_only:
        return stats
    renderer = arena._view_renderer()
    renderer.enable_segmentation_rendering()
    try:
        segmentation = arena.render_wrist()
    finally:
        renderer.disable_segmentation_rendering()
    for role in ids:
        stats[role].update(_mask_stat(segmentation, ids[role]))
    ball = (segmentation[..., 0] == ids['ball']) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    ys, xs = np.nonzero(ball)
    stats['ball']['visible_centroid_px'] = None if not len(xs) else [float(xs.mean()), float(ys.mean())]
    return stats


def _rows(split):
    source = _read(SOURCE)['games']
    selected = source[:1] if split == 'development' else source[1:3]
    rows = [row for game in selected for row in game['points']]
    return [row for row in rows if row['server'] == -1][:4] + [row for row in rows if row['server'] == 1][:4]


def measure(settings, split, *, geometry_only=False, detail='cad', size=(480, 360), antialias=True):
    samples = []
    arena = AimTrialArena(view_mode='wrist', width=size[0], height=size[1], strict_endline=True,
                         detail='preview' if geometry_only else detail, **settings)
    try:
        if not antialias:
            arena.model.vis.quality.offsamples = 0
        ids = dict(blade=arena.model.geom('learner_racket').id, ball=arena.model.geom('ball').id,
                   moving_jaw=_mesh_id(arena.model, JAW, 'moving_jaw_so101_v1'),
                   fixed_jaw=_mesh_id(arena.model, GRIP_BODY, 'wrist_roll_follower_so101_v1'))
        for row in _rows(split):
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            for tick, action in enumerate(row['actions'][:31]):
                if arena.contacts:
                    break
                if tick % 3 == 0:
                    mujoco.mj_kinematics(arena.model, arena.data)
                    mujoco.mj_camlight(arena.model, arena.data)
                    stage = ('toss' if row['server'] == 1 else 'incoming'
                             if arena.ball[1] < .21 and arena.velocity[1] < 0 else 'far')
                    role = _objects(arena, ids, geometry_only=geometry_only)
                    samples.append(dict(seed=row['seed'], server=row['server'], tick=tick, stage=stage, objects=role))
                _, _, ended, capped, _ = arena.step(action, render=False)
                if ended or capped:
                    break
        arena.reset(seed=7, server=-1, render=False)
        home = arena.render_wrist() if not geometry_only else None
    finally:
        arena.close()
    return samples, home


def summarize(samples, common=None, *, geometry_only=False):
    result = {}
    for stage in ('toss', 'incoming', 'far'):
        rows = [row['objects'] for row in samples if row['stage'] == stage and
                (common is None or (row['seed'], row['tick']) in common)]
        inside = [parts for parts in rows if parts['ball']['center_inside']]
        score = dict(samples=len(rows), center_behind=sum(not parts['ball']['center_forward'] for parts in rows),
                     center_outside=len(rows) - len(inside),
                     center_occluded=sum(not parts['ball']['first_hit_same'] for parts in inside))
        if not geometry_only:
            score['resolved_ball'] = sum(parts['ball']['pixels'] >= 4 and not parts['ball']['at_border']
                                         for parts in rows)
            score['whole_blade_and_jaws'] = sum(all(parts[role]['pixels'] >= 40 and not parts[role]['at_border']
                                                   for role in ('blade', 'fixed_jaw', 'moving_jaw')) for parts in rows)
        else:
            score['centers_visible'] = sum(all(parts[role]['center_inside'] and parts[role]['first_hit_same']
                                               for role in ('blade', 'ball')) for parts in rows)
            score['jaw_centers_inside'] = sum(all(parts[role]['center_inside'] for role in ('fixed_jaw', 'moving_jaw'))
                                              for parts in rows)
            score['jaw_outlines_inside'] = sum(all(parts[role]['outline_inside'] for role in ('fixed_jaw', 'moving_jaw'))
                                               for parts in rows)
        result[stage] = score
    return result


def _aperture_audit(spec, size, detail, antialias, *, capture_toss=False):
    arena = AimTrialArena(view_mode='wrist', width=size[0], height=size[1], strict_endline=True,
                         detail=detail, **spec)
    try:
        if not antialias:
            arena.model.vis.quality.offsamples = 0
        neutral = _fixture(arena)
        trials = {name: _course(arena, row) for name, row in _courses().items()}
        toss = None
        if capture_toss:
            row = next(row for row in _rows('held') if row['server'] == 1)
            arena.reset(seed=row['seed'], server=1, render=False)
            for action in row['actions'][:15]:
                arena.step(action, render=False)
            toss = arena.render_wrist()
        return dict(fixture=neutral, trials=trials, summary=_summarize(trials)), toss
    finally:
        arena.close()


def run(folder, *, development=False, optimum=None, detail='cad', ultrawide=False):
    settings = {'outboard': dict(wrist_optics='outboard'), 'raised': dict(wrist_optics='raised')}
    if development and ultrawide:
        for fovy in (120., 126., 130.):
            for elevation in (15., 20., 25., 30.):
                settings[f'outboard-{int(fovy)}-{int(elevation)}'] = dict(wrist_optics='outboard', trial_fovy=fovy,
                                                                         elevation=elevation)
    elif development:
        for mount in ('outboard', 'raised'):
            for fovy in (101., 111.):
                for elevation in (14., 23., 32.):
                    settings[f'{mount}-{int(fovy)}-{int(elevation)}'] = dict(wrist_optics=mount, wrist_fovy=fovy,
                                                                             elevation=elevation)
    if optimum:
        mount, field, pitch = optimum.split('-')
        settings['wider_aim'] = dict(wrist_optics=mount, trial_fovy=float(field), elevation=float(pitch))
    if ultrawide and not development:
        settings.pop('raised')
    data, homes = {}, {}
    for label, spec in settings.items():
        size = (960, 720) if ultrawide and 'trial_fovy' in spec else (480, 360)
        samples, homes[label] = measure(spec, 'development' if development else 'held', geometry_only=development,
                                       detail=detail, size=size, antialias=not ultrawide)
        data[label] = dict(settings=spec, samples=samples, image_size=size)
        print(label, summarize(samples, geometry_only=development), flush=True)
    common = set.intersection(*({(row['seed'], row['tick']) for row in result['samples']} for result in data.values()))
    for item in data.values():
        item['paired'] = summarize(item['samples'], common, geometry_only=development)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = 'wrist-ultrawide' if ultrawide else 'wrist-optics'
    payload = dict(schema='tt20-wrist-optics-1', source_sha256=_hash(SOURCE),
                   detail='preview' if development else detail, views=data)
    if not development and ultrawide and optimum:
        payload['aperture_audit'], toss = _aperture_audit(
            settings['wider_aim'], (960, 720), detail, False, capture_toss=True)
        Image.fromarray(toss).save(folder / 'fixture-wrist-ultrawide-toss.png')
    elif not development and not ultrawide:
        clearance, _ = _aperture_audit(settings['raised'], (480, 360), detail, True)
        payload['qualified_alternate'] = dict(fixture=clearance['fixture'], clearance_courses=clearance['trials'],
                                              clearance_summary=clearance['summary'])
    _write(folder / (prefix + ('-development.json.gz' if development else '-held.json.gz')),
           payload)
    if not development:
        for label in homes:
            Image.fromarray(homes[label]).save(folder / f'{prefix}-{label}.png')
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path, nargs='?', default=ROOT / 'artifacts' / 'tt20')
    p.add_argument('--development', action='store_true')
    p.add_argument('--optimum')
    p.add_argument('--detail', choices=('cad', 'preview'), default='cad')
    p.add_argument('--ultrawide', action='store_true')
    args = p.parse_args()
    run(args.folder, development=args.development, optimum=args.optimum, detail=args.detail, ultrawide=args.ultrawide)
