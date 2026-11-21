"""Check official SO-101 mechanics against the table model and record physical probes."""

import argparse
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _write
from .tt18_plant_audit import opening_quality
from .tt19_so101 import (ASSET_DIR, AXES, BLADE, CAMERA_BODY, GRIP_BODY, PADDLE_BODY, SO101_HOME,
                         SOURCE_MODEL, SOURCE_REVISION, TIP, WRIST_CAMERA, SO101Arena)


def upstream_match(arena):
    original = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    rows = []
    for upstream in ('base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper',
                     'moving_jaw_so101_v1', 'wrist_camera_mount', 'wrist_camera'):
        name = 'learner_elbow' if upstream == 'lower_arm' else f'learner_{upstream}'
        reference, deployed = original.body(upstream), arena.model.body(name)
        for field in ('mass', 'inertia', 'ipos', 'iquat'):
            np.testing.assert_array_equal(getattr(reference, field), getattr(deployed, field))
        if upstream != 'base':
            for field in ('pos', 'quat'):
                np.testing.assert_array_equal(getattr(reference, field), getattr(deployed, field))
        rows.append(dict(source=upstream, deployed=name, mass_g=float(deployed.mass[0]) * 1000,
                         local_origin=deployed.pos.tolist(), local_quaternion=deployed.quat.tolist(),
                         center_of_mass=deployed.ipos.tolist(), principal_inertia=deployed.inertia.tolist()))

    joints = []
    for axis in AXES:
        source, target = original.joint(axis), arena.model.joint(f'learner_{axis}')
        for field in ('pos', 'range', 'axis', 'type'):
            np.testing.assert_array_equal(getattr(source, field), getattr(target, field))
        source_drive, target_drive = original.actuator(axis), arena.model.actuator(f'learner_{axis}')
        for field in ('forcerange', 'gainprm', 'biasprm'):
            np.testing.assert_array_equal(getattr(source_drive, field), getattr(target_drive, field))
        joints.append(dict(name=axis, axis=target.axis.tolist(), limits_rad=target.range.tolist(),
                           actuator_force_range_nm=target_drive.forcerange.tolist()))
    return dict(source_revision=SOURCE_REVISION, source_sha256=_hash(SOURCE_MODEL), reference_bodies=rows,
                joints=joints, source_mesh_sha256={p.name: _hash(p) for p in sorted((ASSET_DIR / 'assets').glob('*.stl'))})


def check_joint_travel(arena):
    limits = np.stack([arena.model.joint(f'learner_{axis}').range.copy() for axis in AXES])
    drives = [arena.model.actuator(f'learner_{axis}').id for axis in AXES]
    arena.reset(seed=19, server=1, render=False)
    for _ in range(12):
        arena.step(np.array([1., 1., 1., -1., -1., -1.]), render=False)
    # Some requested targets reach opposite mechanical stops within the same point.
    targets = arena.data.ctrl[drives]
    if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]) or not arena.grasped:
        raise ValueError('the command driver or gripper escaped its supported range')
    return dict(limits=limits.tolist(), active_targets=targets.tolist(), actual_joints=arena.joints.tolist())


def check_grasp(arena):
    arena.reset(seed=9, server=1, render=False)
    clamps = []
    parent = arena.model.body(GRIP_BODY).id
    paddle = arena.model.body(PADDLE_BODY).id
    for _ in range(5):
        arena.step(np.array((.3, -.2, .3, .2, .6, 0.)), render=False)
        rotation = arena.data.xmat[parent].reshape(3, 3)
        local = rotation.T @ (arena.data.xpos[paddle] - arena.data.xpos[parent])
        clamps.append(local.tolist())
    if not arena.grasped:
        raise ValueError('the neutral jaw dropped the paddle during a stroke')
    for _ in range(11):
        arena.step(np.array((0., 0., 0., 0., 0., 1.)), render=False)
    release = next(e for e in arena.events if e['kind'] == 'grip_release')
    if arena.grasped:
        raise ValueError('opening the physical jaw failed to release the paddle')
    height_before = float(arena.paddle[2])
    for _ in range(5):
        arena.step(np.zeros(arena.action_size), render=False)
    height_after = float(arena.paddle[2])
    if height_after >= height_before - .08:
        raise ValueError('released paddle did not fall')
    return dict(held_poses=clamps, release_step=release['step'], open_angle_rad=float(arena.joints[-1]),
                paddle_height_before_fall_m=height_before, paddle_height_after_fall_m=height_after)


def replay_point(arena, *, seed, server, tape):
    arena.reset(seed=seed, render=False, server=server)
    for action in tape:
        arena.step(action, render=False)
        if arena.terminated or arena.truncated:
            break
    if not (arena.terminated or arena.truncated):
        raise ValueError('point tape ended before the scored point')
    return dict(seed=seed, server=server, evaluator=arena.point_result(), events=arena.events,
                actions=np.asarray(tape[:arena.steps]).tolist())


def motion_parity(full, preview):
    tape = np.zeros((120, 6))
    tape[:6, :5] = np.array((.18, -.25, .32, -.28, .31))
    tape[19:24, :5] = -.31 * tape[:1, :5]
    points = []
    for server in (1, -1):
        for seed in (0, 1, 3, 19):
            exact = replay_point(full, seed=seed, server=server, tape=tape)
            reduced = replay_point(preview, seed=seed, server=server, tape=tape)
            if exact != reduced:
                raise ValueError('visual detail changed a scored action tape')
            np.testing.assert_array_equal(full.data.qpos, preview.data.qpos)
            np.testing.assert_array_equal(full.data.qvel, preview.data.qvel)
            points.append(exact)
    return points


def scoring_witnesses(arena):
    # This one command sequence was selected with evaluator trajectory evidence.
    # It witnesses a reachable mechanism, not a player learning result.
    tape = np.zeros((120, 6))
    tape[:6, :5] = np.array((-.502044077, -1.053878132, -1.852107825, -.803937260, .210591024)) / 6
    tape[24:28, :5] = [.324668587, .471259605, .163672395, -.118069757, .011159709]
    received = replay_point(arena, seed=0, server=-1, tape=tape)
    served = replay_point(arena, seed=588725045, server=1, tape=np.zeros_like(tape))
    if (not received['evaluator']['legal_serve'] or received['evaluator']['legal_returns'] != 1
            or received['evaluator']['points_for'] != 1 or not served['evaluator']['legal_serve']):
        raise ValueError('the physical scoring witnesses no longer qualify')
    return dict(return_point=received, serve_point=served)


def camera_probe(arena, folder):
    # Use a native red-rubber sample for both photographs.
    seed = next(seed for seed in range(25) if _red_reset(arena, seed))
    wrist = arena.render_wrist()
    Image.fromarray(wrist).save(folder / 'so101-wrist.png')
    Image.fromarray(arena.render_cameras()[0]).save(folder / 'so101-court.png')
    camera_id = arena.model.camera(WRIST_CAMERA).id
    parent = arena.model.body(CAMERA_BODY).id
    before = arena.data.cam_xpos[camera_id].copy()
    renderer = arena._mounted_renderer
    renderer.enable_segmentation_rendering()
    try:
        arena._update_wrist_scene()
        ids = renderer.render()[..., 0]
    finally:
        renderer.disable_segmentation_rendering()
    blade = arena.model.geom(BLADE).id
    jaw = arena.model.body('learner_moving_jaw_so101_v1').id
    jaw_geom = np.flatnonzero(arena.model.geom_bodyid == jaw)
    pixels = dict(blade=int(np.count_nonzero(ids == blade)), moving_jaw=int(np.count_nonzero(np.isin(ids, jaw_geom))))
    if min(pixels.values()) < 40 or arena.model.camera(WRIST_CAMERA).bodyid[0] != parent:
        raise ValueError('the physical wrist RGB failed to resolve the blade and jaw')
    for _ in range(3):
        arena.step(np.array((.5, .4, -.2, .6, -.4, 0.)), render=False)
    moved = arena.render_wrist()
    return dict(seed=seed, wrist_shape=list(wrist.shape), visible_pixels=pixels, near_clip_m=arena.wrist_near,
                center_motion_m=float(np.linalg.norm(arena.data.cam_xpos[camera_id] - before)),
                changed_pixels=float(np.any(wrist != moved, axis=2).mean()))


def _red_reset(arena, seed):
    arena.reset(seed=seed, render=False, server=1)
    return arena.model.geom(BLADE).rgba[0] > .5


def _triangles(directory):
    return sum(struct.unpack_from('<I', p.read_bytes(), offset=80)[0] for p in directory.glob('*.stl'))


def audit(folder):
    folder.mkdir(parents=True, exist_ok=True)
    arena = SO101Arena(venue='hall', width=384, height=288, wrist_near=.012)
    preview = SO101Arena(venue='hall', detail='preview', width=384, height=288)
    try:
        dimensions = dict(width_m=2 * arena.table_half_width, length_m=2 * arena.table_half_length,
                          height_m=TABLE_HEIGHT, control_hz=1 / arena.control_dt, home_rad=list(SO101_HOME),
                          learner_base=arena.model.body('learner_base').pos.tolist(),
                          rival_base=arena.model.body('rival_base').pos.tolist())
        receipt = dict(schema='tt19-so101-audit-v1', physical=dimensions, upstream=upstream_match(arena),
                       joint_driver=check_joint_travel(preview), grasp=check_grasp(preview),
                       camera=camera_probe(arena, folder), rival_openings=opening_quality(preview, 64),
                       matched_points=motion_parity(arena, preview),
                       scoring_witnesses=scoring_witnesses(arena),
                       visual_triangles={mode: _triangles(directory) for mode, directory in
                                         [('cad', ASSET_DIR / 'assets'), ('preview', ASSET_DIR / 'assets_preview')]})
    finally:
        arena.close()
        preview.close()
    _write(folder / 'so101-audit.json.gz', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    audit(parser.parse_args().folder)
