"""Check clamp clearances and opaque, physical wrist optics on held SO-101 strokes."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_so101 import CAMERA_BODY, GRIP_BODY, WRIST_CAMERA
from .tt20_fixture import TT20Arena


SOURCE = ROOT / 'artifacts' / 'tt19' / 'match-full-from-wide.json.gz'
SAMPLES = (0, 5, 10, 15, 20, 26)
JAW = 'learner_moving_jaw_so101_v1'


def _mesh_id(model, body_name, mesh_name):
    body = model.body(body_name).id
    return next(index for index in range(model.ngeom)
                if model.geom_bodyid[index] == body and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
                and model.mesh(model.geom_dataid[index]).name == mesh_name)


def _camera_pose(arena):
    data = arena.data
    camera = data.camera(WRIST_CAMERA)
    return camera.xpos.copy(), camera.xmat.reshape(3, 3).copy()


def _rays(arena, image):
    """Compare solid geometric first hits with sampled native pixel IDs."""
    model, data = arena.model, arena.data
    eye, axes = _camera_pose(arena)
    focal = arena.lens_intrinsics()['wrist']['focal']
    center = np.array(arena.lens_intrinsics()['wrist']['center'])
    near = model.vis.map.znear * model.stat.extent
    names = [model.body(model.geom_bodyid[index]).name for index in range(model.ngeom)]
    group = np.ones(6, dtype=np.uint8)
    rows, columns = np.linspace(9, arena.height - 10, 11).astype(int), np.linspace(9, arena.width - 10, 15).astype(int)
    counts = {'rays': int(len(rows) * len(columns)), 'opaque_hits_before_clip': 0,
              'wrist_hardware_hits': 0, 'wrist_hits_rendered_as_same_geom': 0, 'wrist_hits_hidden_by_clip': 0}
    wrist_names = ('learner_wrist', GRIP_BODY, JAW, CAMERA_BODY, 'learner_wrist_camera_mount')
    near_hits = []
    geom = np.empty(1, dtype=np.int32)
    for y in rows:
        for x in columns:
            u, v = (np.array((x, y)) - center) / focal
            ray = axes @ np.array((u, -v, -1.))
            depth = mujoco.mj_ray(model, data, eye, ray, group, True, -1, geom)
            if depth < 0.:
                continue
            index = int(geom[0])
            transparent = model.geom_rgba[index, 3] < .95
            if depth < near and not transparent:
                counts['opaque_hits_before_clip'] += 1
                near_hits.append(float(depth))
            if names[index] in wrist_names and not transparent:
                counts['wrist_hardware_hits'] += 1
                visible = image[y, x, 0] == index and image[y, x, 1] == mujoco.mjtObj.mjOBJ_GEOM
                counts['wrist_hits_rendered_as_same_geom'] += bool(visible)
                counts['wrist_hits_hidden_by_clip'] += bool(depth < near and not visible)
    counts['nearest_clipped_hit_m'] = None if not near_hits else min(near_hits)
    return counts


def _mask_stat(image, geom):
    mask = (image[..., 0] == geom) & (image[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    at_border = np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
    return dict(pixels=int(mask.sum()), at_border=bool(at_border))


def _measurement(arena):
    renderer = arena._view_renderer()
    renderer.enable_segmentation_rendering()
    try:
        objects = arena.render_wrist()
    finally:
        renderer.disable_segmentation_rendering()
    model, data = arena.model, arena.data
    eye, axes = _camera_pose(arena)
    targets = {'blade': model.geom('learner_racket').id, 'handle': model.geom('learner_handle').id,
               'ball': model.geom('ball').id, 'table': model.geom('table').id,
               'fixed_jaw_assembly': _mesh_id(model, GRIP_BODY, 'wrist_roll_follower_so101_v1'),
               'moving_jaw': _mesh_id(model, JAW, 'moving_jaw_so101_v1')}
    clamp = [index for index in range(model.ngeom) if model.geom(index).name.startswith('tt20_clamp_')]

    def support_distance(item):
        return min((mujoco.mj_geomDistance(model, data, item, other, 2., None) for other in clamp), default=None)

    return dict(tick=arena.steps, eye_m=eye.tolist(), forward=(-axes[:, 2]).tolist(),
                world_x_right=float(axes[0, 0]), paddle_m=arena.paddle.tolist(),
                blade_to_clamp_m=support_distance(targets['blade']),
                jaw_to_clamp_m=support_distance(targets['moving_jaw']),
                images={role: _mask_stat(objects, geom) for role, geom in targets.items()},
                rays=_rays(arena, objects))


def _fixture(arena):
    model, data = arena.model, arena.data
    table = model.geom('table')
    clamp = [index for index in range(model.ngeom) if model.geom(index).name.startswith('tt20_clamp_')
             and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX]
    high = max((data.geom_xpos[index, 2] + model.geom_size[index, 2] for index in clamp), default=None)
    flange = data.body('learner_base').xpos.copy()
    top_z = None if not clamp else data.geom('tt20_clamp_flange').xpos[2] + model.geom('tt20_clamp_flange').size[2]
    camera = data.body(CAMERA_BODY)
    eye, aim = _camera_pose(arena)
    aperture = camera.xmat.reshape(3, 3).T @ (eye - camera.xpos)
    return dict(base_flange_m=flange.tolist(), support_deck_top_m=top_z, highest_support_box_m=high,
                tabletop_m=float(table.pos[2] + table.size[2]), underside_m=float(table.pos[2] - table.size[2]),
                near_clip_m=float(model.vis.map.znear * model.stat.extent),
                board_mass_g=float(model.body(CAMERA_BODY).mass[0] * 1000.),
                aperture_in_board_frame_m=aperture.tolist(), lens=arena.lens_intrinsics(),
                wrist_forward=(-aim[:, 2]).tolist(), neutral_eye_m=eye.tolist())


def _courses():
    first = _read(SOURCE)['games'][0]['points']
    serve = next(row for row in first if row['server'] == 1 and row['result']['legal_serve'])
    receive = next(row for row in first if row['server'] == -1 and row['result']['legal_returns'] > 0)
    stroke = np.zeros((34, 6))
    stroke[:6] = (.35, -.32, .33, -.16, .37, 0.)
    stroke[6:13] = (-.5, .25, -.31, .21, -.46, 0.)
    stroke[13:19] = (.2, -.14, .22, -.23, .31, 0.)
    return dict(receive=receive, serve=serve, sweep=dict(seed=11, server=-1, actions=stroke.tolist()))


def _course(arena, row):
    arena.reset(seed=row['seed'], server=row['server'], render=False)
    samples = [_measurement(arena)]
    for tick, action in enumerate(row['actions'], 1):
        _, _, ended, capped, _ = arena.step(action, render=False)
        if tick in SAMPLES:
            samples.append(_measurement(arena))
        if ended or capped:
            break
    return dict(seed=row['seed'], server=row['server'], samples=samples, scorer=arena.point_result())


def _montage(old_image, new_image, path):
    old, new = Image.fromarray(old_image), Image.fromarray(new_image)
    art = Image.new('RGB', (old.width + new.width, old.height + 32), '#171e23')
    art.paste(old, (0, 32))
    art.paste(new, (old.width, 32))
    label = ImageDraw.Draw(art)
    font = ImageFont.load_default(size=15)
    label.text((14, 8), 'TT19 pedestal and wrist aperture', fill='#edf1f3', font=font)
    label.text((old.width + 14, 8), 'TT20 moved clamp and outboard wrist', fill='#edf1f3', font=font)
    art.save(path)


def _module_view(arena):
    overview = mujoco.MjvCamera()
    overview.type = mujoco.mjtCamera.mjCAMERA_FREE
    overview.lookat[:] = (0., -arena.table_half_length - .135, .87)
    overview.distance = .69
    overview.azimuth = 171.
    overview.elevation = -18.
    return arena._render_view(arena._view_renderer(), arena.data, overview)


def _summarize(trials):
    records = [row for course in trials.values() for row in course['samples']]
    visibility = {role: dict(seen=sum(row['images'][role]['pixels'] > 0 for row in records),
                             minimum_pixels=min(row['images'][role]['pixels'] for row in records),
                             cropped=sum(row['images'][role]['at_border'] for row in records))
                  for role in records[0]['images']}
    ray_keys = ('rays', 'opaque_hits_before_clip', 'wrist_hardware_hits', 'wrist_hits_rendered_as_same_geom',
                'wrist_hits_hidden_by_clip')
    return dict(poses=len(records), visible=visibility,
                rays={key: sum(row['rays'][key] for row in records) for key in ray_keys},
                blade_clamp_clearance_m=min((row['blade_to_clamp_m'] for row in records
                                             if row['blade_to_clamp_m'] is not None), default=None),
                moving_jaw_clearance_m=min((row['jaw_to_clamp_m'] for row in records
                                            if row['jaw_to_clamp_m'] is not None), default=None))


def run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(view_mode='wrist', detail='cad', width=480, height=360, strict_endline=True)
    result = dict(schema='tt20-fixture-audit-1', settings=settings, source=_hash(SOURCE), courses={})
    initial, courses = {}, _courses()
    for label, mount, optics in (('prior', 'pedestal', 'legacy'), ('clamp', 'clamp', 'outboard')):
        arena = TT20Arena(**settings, mount=mount, wrist_optics=optics)
        try:
            initial[label] = arena.reset(seed=7, server=-1)['wrist']
            fixture = _fixture(arena)
            if label == 'clamp':
                Image.fromarray(initial[label]).save(folder / 'fixture-wrist.png')
                Image.fromarray(arena.render_broadcast()).save(folder / 'fixture-full-view.png')
                Image.fromarray(_module_view(arena)).save(folder / 'fixture-module.png')
            print(label, 'geometry', fixture['base_flange_m'], fixture['near_clip_m'], flush=True)
            trials = {name: _course(arena, row) for name, row in courses.items()}
            result['courses'][label] = dict(fixture=fixture, trials=trials, summary=_summarize(trials))
            print(label, result['courses'][label]['summary'], flush=True)
        finally:
            arena.close()
    _montage(initial['prior'], initial['clamp'], folder / 'fixture-wrist-comparison.png')
    _write(folder / 'fixture-audit.json.gz', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=ROOT / 'artifacts' / 'tt20')
    run(parser.parse_args().folder)
