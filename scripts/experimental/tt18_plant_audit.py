"""Audit TT18's compiled MuJoCo plants with matched states and TT17 command tapes."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt17_motor_protocol import wiring_for
from .tt18_plant import MOUNTED_CAMERA, PAYLOAD, PLANTS, PlantSettings, TT18PlantArena, make_arena


def _launch(arena, position, velocity, spin=(0., 0., 0.)):
    arena.reset(seed=41, render=False)
    arena.ball[:] = position
    arena.velocity[:] = velocity
    arena.data.qvel[arena._ball_dof + 3:arena._ball_dof + 6] = spin
    mujoco.mj_forward(arena.model, arena.data)


def flight(arena):
    _launch(arena, (.12, .92, 1.65), (1.4, -3.2, .6), (95., -17., 22.))
    arena._aerodynamics()
    force = arena.data.xfrc_applied[arena._ball_body].copy()
    for _ in range(6):
        arena.step(np.zeros(arena.action_size), render=False)
    if arena.events:
        raise ValueError('free-flight probe encountered another geom')
    return dict(seconds=float(arena.data.time), applied_wrench=force.tolist(), ball=arena.ball.tolist(),
                velocity=arena.velocity.tolist(), spin=arena.spin.tolist())


def drop(arena, x, y=.74):
    _launch(arena, (x, y, TABLE_HEIGHT + .34), (0., 0., -1.7))
    for _ in range(160):
        arena.step(np.zeros(arena.action_size), render=False)
        bounces = [event for event in arena.events if event['kind'] == 'bounce']
        separated = next((event for event in arena.events
                          if event['kind'] == 'separation' and event['collider'] == 'table'), None)
        if separated or arena.terminated or arena.truncated:
            return dict(bounces=len(bounces), impact=None if not bounces else bounces[0]['incoming'][2],
                        rebound=None if separated is None else separated['velocity'][2], reason=arena.reason)
    raise ValueError('drop probe did not finish')


def racket_impact(arena):
    arena.reset(seed=41, render=False)
    blade = arena._racket_ids[1]
    normal = arena.data.geom_xmat[blade].reshape(3, 3)[:, 2].copy()
    normal *= np.sign(normal[1])
    _launch(arena, arena.paddle + normal * (BALL_RADIUS + .006 - .00008), -normal * 3.7)
    arena._received_bounce, arena._serving = True, False
    for _ in range(12):
        arena.step(np.zeros(arena.action_size), render=False)
        separated = next((event for event in arena.events
                          if event['kind'] == 'separation' and event['collider'] == 'learner_racket'), None)
        if separated:
            contact = next(event for event in arena.events if event['kind'] == 'contact' and event['side'] == 1)
            return dict(incoming_normal_mps=float(normal @ contact['incoming']),
                        separation_normal_mps=float(normal @ separated['velocity']))
    raise ValueError('racket probe did not find a physical contact and separation')


def draw_camera(arena, tape):
    arena.reset(seed=41, render=False)
    camera = arena.model.camera(MOUNTED_CAMERA)
    camera_id = camera.id
    before = arena.render_mounted_camera()
    start = arena.data.cam_xpos[camera_id].copy()
    forward = -arena.data.cam_xmat[camera_id].reshape(3, 3)[:, 2].copy()
    for action in tape[:2]:
        arena.step(action, render=False)
    after = arena.render_mounted_camera()
    following = -arena.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    body = int(camera.bodyid[0])
    return dict(shape=list(before.shape), changed_pixel_fraction=float(np.any(before != after, axis=-1).mean()),
                displacement_m=float(np.linalg.norm(arena.data.cam_xpos[camera_id] - start)),
                rotation_deg=float(np.degrees(np.arccos(np.clip(forward @ following, -1., 1.)))),
                housing_on_camera_body=body == int(arena.model.geom(PAYLOAD).bodyid[0]),
                parent=arena.model.body(body).name, parent_mass_kg=float(arena.model.body_mass[body]),
                frame_sha256=[sha256(im.tobytes()).hexdigest() for im in (before, after)])


def _body_masses(arena):
    return {b.name: dict(mass=float(b.mass[0]), principal_inertia=b.inertia.tolist())
            for i in range(arena.model.nbody) for b in (arena.model.body(i),)
            if b.name.startswith('learner_') and b.name != 'learner_base'}


def motion(arena, tape):
    arena.reset(seed=41, render=False)
    track = [arena.paddle.copy()]
    for action in tape:
        arena.step(action, render=False)
        track.append(arena.paddle.copy())
    actuators = arena._arm_ids[1][2]
    return np.asarray(track), dict(body_masses=_body_masses(arena),
                                   control_range=np.stack((arena._joint_low[actuators],
                                                           arena._joint_high[actuators]), axis=-1).tolist())


def _motion_variants(body):
    zero = PlantSettings()
    proximal = 'bridge' if body == 'gantry' else 'carriage'
    stock = TT8ServiceArena(body=body)
    # Redistribute the same incremental mass between links, with no camera.
    amount = .05
    multipliers = {name: 1 + amount / float(stock.model.body(f'learner_{name}').mass[0])
                   for name in (proximal, 'wrist')}
    stock.close()
    return dict(stock=zero, proximal_heavy=replace(zero, link_mass_scales={proximal: multipliers[proximal]}),
                distal_heavy=replace(zero, link_mass_scales={'wrist': multipliers['wrist']}),
                wrist_camera=replace(zero, payload_mount='wrist'),
                proximal_camera=replace(zero, payload_mount='proximal'),
                **PLANTS)


def _audit_motion(body):
    width = 4 if body == 'gantry' else 5
    signs = np.resize([1., -1.], width)
    tape = np.vstack([np.tile(signs * rate, (frames, 1)) for rate, frames in ((.95, 9), (-1., 9), (.7, 7))])
    report, baseline = {}, None
    for name, settings in _motion_variants(body).items():
        arena = TT18PlantArena(plant=settings, body=body, venue='hall', width=96, height=72)
        try:
            track, receipt = motion(arena, tape)
            if name == 'stock':
                baseline = track
            receipt['max_blade_delta_m'] = float(np.linalg.norm(track - baseline, axis=1).max())
            receipt['blade_track_m'] = track.tolist()
            if settings.payload_mount is not None:
                receipt['camera'] = draw_camera(arena, tape)
            report[name] = receipt
        finally:
            arena.close()
    a = np.asarray(report['proximal_heavy']['blade_track_m'])
    b = np.asarray(report['distal_heavy']['blade_track_m'])
    return dict(tape=tape.tolist(), variants=report, proximal_distal_delta_m=float(np.linalg.norm(a - b, axis=1).max()))


def opening_quality(arena, count=100):
    counts = {}
    for seed in range(count):
        arena.reset(seed=seed, render=False, server=-1)
        while not (arena.terminated or arena.truncated or arena.legal_serve):
            arena.step(np.zeros(arena.action_size), render=False)
        result = 'legal' if arena.legal_serve else arena.reason
        counts[result] = counts.get(result, 0) + 1
    return counts


def _audit_ball_and_table():
    result = {}
    choices = dict(stock=PlantSettings(), heavy_ball=PlantSettings(ball_mass_scale=1.34),
                   left_soft=PlantSettings(left_damping=48., right_damping=15.),
                   right_soft=PlantSettings(left_damping=15., right_damping=48.), **PLANTS)
    for name, settings in choices.items():
        arena = TT18PlantArena(plant=settings, body='gantry', frameskip=1, max_steps=430)
        airborne = TT18PlantArena(plant=settings, body='gantry')
        try:
            result[name] = dict(settings=asdict(settings), flight=flight(airborne),
                                drops={str(x): drop(arena, x) for x in (-.37, 0., .37, .74)},
                                mass_kg=float(arena.model.body('ball').mass[0]),
                                ball_inertia=arena.model.body('ball').inertia.tolist())
            if name in ('stock', 'heavy_ball'):
                result[name]['racket_impact'] = racket_impact(arena)
            if name == 'stock' or name in PLANTS:
                result[name]['rival_openings'] = opening_quality(arena)
        finally:
            arena.close()
            airborne.close()
    return result


def _audit_tapes(folder, case):
    setup = folder / f'motor-{case}-setup.json.gz'
    archive = folder / f'motor-{case}-allocate.json.gz'
    spec = _read(setup)
    game = _read(archive)['games'][0]
    treatments = {}
    for name in ('stock', *PLANTS):
        arena = make_arena(spec if name == 'stock' else dict(spec, plant=name))
        rows = []
        try:
            wires = wiring_for(arena, spec)
            for row in game['points']:
                interface = PointInterface(wires, prepare_point(arena, row['phase']), server=row['server'],
                                           seed=game['seed'])
                interface.reset(seed=row['seed'], render=False)
                for action in row['actions']:
                    if arena.terminated or arena.truncated:
                        break
                    interface.step(action, render=False)
                ready = arena.terminated or arena.truncated
                outcome = arena.point_result()
                if name == 'stock' and (not ready or outcome != row['evaluator'] or interface.applied != row['applied']
                                        or json.loads(json.dumps(arena.events)) != row['events']):
                    raise ValueError('the unchanged TT17 point failed its physical replay')
                rows.append(dict(seed=row['seed'], recorded_role=row['server'], phase=row['phase'],
                                 recorded_action_sha256=row['action_sha256'], commands_executed=arena.steps,
                                 finished=ready, evaluator=outcome, events=arena.events))
        finally:
            arena.close()
        treatments[name] = rows
    return dict(setup_sha256=_hash(setup), game_sha256=_hash(archive), treatments=treatments)


def audit(output, tt17):
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema='tt18-plant-audit-v1', ball_table=_audit_ball_and_table(),
                        arms={body: _audit_motion(body) for body in ('gantry', 'offset')},
                        tapes={case: _audit_tapes(tt17, case)
                               for case in ('gantry-coupling', 'five-gain-tt14')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tt17', type=Path, default=Path('artifacts/tt17'))
    args = parser.parse_args()
    audit(args.output, args.tt17)
