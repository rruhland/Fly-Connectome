"""Audit launch acquisition and qualify body-specific paddle-effect feedback."""

import argparse
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_metric_stereo import Camera, triangulate
from .tt24_posture import WidePosture
from .tt24_public_body import DIRECTIONS, RIM, blade_profile, probe, stationary
from .tt24_public_morph import ProbeEyes
from .tt24_ready import prepare_step
from .tt24_redundant import MarkerBounce, _CALIBRATION_CAMERAS
from .tt24_redundant_trial import markers
from .tt24_tool_equilibrium import ToolEquilibrium, paddle


AUDIT = ROOT / 'acquisition-selfmodel-audit.json.gz'
FIT = ROOT / 'acquisition-selfmodel-fit.json.gz'
MARKED = ROOT / 'acquisition-selfmodel-marked.json.gz'
OUTPUT = ROOT / 'acquisition-selfmodel-held.json.gz'
AUDIT_CAP = 300
LIVE_CAP = 480
FIT_SEEDS = (481714101, 481714102, 481714103, 481714104)
CHECK_SEEDS = (481714201, 481714202)
HELD_SEED = 158340049
PHASES = ('frozen', 'absolute', 'stock_gauge', 'hold')
WORKSPACE = .0557


def fixture(plan, server):
    server.reset(seed=plan['launcher_seed'], split='held', render=False)
    return tuple(value.copy() for value in (server._ball_position, server._ball_velocity, server._ball_spin))


def advance(arm, action, launch, tick, preparation, render):
    arena = arm.arena
    location, velocity, spin = launch
    if tick < preparation:
        prepare_step(arena, action, location)
        if tick + 1 == preparation:
            arena._ball_velocity[:] = velocity
            arena._ball_spin[:] = spin
            mujoco.mj_forward(arena._model, arena._data)
        return arena._observation(render), False
    observation, _, done = arena.step(action, render=render)
    return observation, done


def metrology(arm):
    """The archived-replay referee writes these measurements after action selection."""
    arena = arm.arena
    data = arena._data
    normal = data.geom_xmat[arena._blade].reshape((3, 3))[:, 2]
    normal = normal if normal[2] > 0 else -normal
    return np.r_[data.geom_xpos[arena._blade], normal[:2]], arena._ball_position.copy()


def audit(path):
    server = BounceArena(view='none')
    arm = {name: InstalledArm(name) for name in ('stock', 'alternate')}
    policy = ToolEquilibrium(BUNDLE, WORKSPACE)
    policy.mode = 'workspace_local'
    reports = []
    try:
        for index in range(12):
            episode = _read(ROOT / f'long-wrist-{index}.json.gz')
            plan = episode['plan']
            recorded = episode['results']['live']
            live = arm[plan['body']]
            live.arena.max_steps = AUDIT_CAP
            launch = fixture(plan, server)
            obs = live.reset(plan, launch[0], True)
            policy.begin_release(obs)
            preparation = plan['prep_ticks']
            gravity = dict(plan['changes'])
            trace = []
            for tick, archived in enumerate(recorded['actions'][:AUDIT_CAP + preparation]):
                if tick - preparation in gravity:
                    live.switch_gravity(tick-preparation, gravity[tick-preparation])
                predicted = policy.act(obs)
                if not np.array_equal(predicted, archived):
                    raise ValueError(f'archived command differs on start {index}, tick {tick}')
                true_pose, ball = metrology(live)
                estimated = paddle(policy.sensor.decoder, obs['joints'])
                goal = paddle(policy.sensor.decoder, policy.reference + policy.issued)
                trace.append(dict(tick=tick-preparation, at=float(live.arena._data.time),
                                  actual=true_pose.tolist(), known=estimated.tolist(), aim=goal.tolist(),
                                  ball=ball.tolist(), seen=policy.sensor.previous.tolist(),
                                  q=obs['joints'].tolist(), action=predicted.tolist()))
                obs, done = advance(live, predicted, launch, tick, preparation, True)
                if done:
                    break
            closest = []
            for event in recorded['metrics']['rebound_events'][:5]:
                matched = min(trace, key=lambda row: abs(row['at']-event['contact_time']))
                if abs(matched['at']-event['contact_time']) < .018:
                    closest.append(matched | dict(rise=event['rise_m']))
            reports.append(dict(index=index, body=plan['body'], capture=closest, early_trace=trace,
                                elapsed=recorded['metrics']['elapsed_s']-preparation/60,
                                termination=recorded['metrics']['termination']))
            print(index, plan['body'], round(reports[-1]['elapsed'], 2), len(trace), flush=True)
    finally:
        server.close()
        for installed in arm.values():
            installed.arena.close()
    _write(path, reports)


class ProbeArm(WidePosture):
    fixed_images = ProbeEyes.fixed_images
    configure_reference_cameras = ProbeEyes.configure_reference_cameras

    def __init__(self, body):
        super().__init__(view='wrist', width=192, height=144, arm=body)
        self.configure_reference_cameras()


def outline_frame(arena, cameras):
    silhouettes = np.asarray([blade_profile(rgb) for rgb in arena.fixed_images()])
    if np.isfinite(silhouettes).all():
        return dict(outline=silhouettes[:, 2:].tolist())


def collect_probe(arena, seed, body, acquire, read, cameras):
    """Watch a contact-free paddle from ordinary motor babble and camera pixels."""
    regime = 'changed' if body == 'alternate' else 'standard'
    obs = arena.reset(seed=seed, split='held', regime=regime, render=False)
    home = obs['joints'].copy()
    absent = np.array((5., 5., 5.))
    arena._ball_position[:] = absent
    mujoco.mj_forward(arena._model, arena._data)
    cameras = acquire(arena) if cameras is None else cameras
    rows, tape = [], []
    for tick in range(72):
        q = obs['joints']
        if tick % 3 == 0:
            sample = read(arena, cameras)
            if sample is not None:
                rows.append(dict(tick=tick, encoders=q.tolist(), **sample))
        command = probe(seed, tick, home, q)
        tape.append(command.tolist())
        prepare_step(arena, command, absent)
        obs = arena._observation(False)
    return dict(seed=seed, rows=rows, actions=tape), cameras


def local_normals(decoder, encoders, equilibrium, theta):
    """A small full-angle normal correction conditioned on proprioception."""
    normal = decoder.normal
    axis = np.array((0., 0., 1.))
    tangent = np.stack((axis, np.cross(normal, axis)))
    design = np.column_stack((np.ones(len(encoders)), (encoders - equilibrium) / .25))
    coefficients = theta[:12].reshape(6, 2)
    return Rotation.from_rotvec((design @ coefficients) @ tangent).apply(normal)


def tool_poses(decoder, q, equilibrium, theta):
    translation, rotation = decoder.links.read(q)
    centers = translation + decoder.offset[:3] + theta[12:15]
    centers += rotation @ (decoder.offset[3:] + theta[15:18])
    normals = np.einsum('nij,nj->ni', rotation, local_normals(decoder, q, equilibrium, theta))
    normals = np.where(normals[:, 2, None] >= 0., normals, -normals)
    return centers, normals


def observations(records):
    rows = [row for record in records for row in record['rows']]
    return np.asarray([r['encoders'] for r in rows]), np.asarray([r['outline'] for r in rows])


def silhouette_error(decoder, q, silhouette, equilibrium, theta, cameras):
    center, normal = tool_poses(decoder, q, equilibrium, theta)
    tangent = np.cross(normal, (0., 1., 0.))
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    plane = np.stack((tangent, np.cross(normal, tangent)), axis=1)
    rim = center[:, None, :] + theta[18] * (RIM @ plane)
    breadth = theta[19] * normal[:, None, :]
    plate = np.concatenate((rim + breadth, rim - breadth), axis=1)
    predicted = []
    for camera in cameras:
        pixels, _ = camera.project(plate.reshape(-1, 3))
        supports = (pixels.reshape(len(plate), -1, 2) @ DIRECTIONS.T).max(axis=1)
        predicted.append(supports)
    return np.stack(predicted, axis=1) - silhouette


def fit_geometry(decoder, records, equilibrium, cameras):
    """Fit only observed image supports, using the carried kinematic chain as a prior."""
    start = np.r_[np.zeros(18), .075, .003]
    scale = np.r_[[.20] * 2, [.055] * 10, [.012] * 3, [.009] * 3, .006, .005]
    lower, upper = start - 2 * scale, start + 2 * scale
    lower[-1] = 0.
    q, observed = observations(records)
    if len(q) < 50:
        raise ValueError('insufficient independently imaged paddle silhouettes')
    half = len(DIRECTIONS) // 2

    def loss(theta):
        error = silhouette_error(decoder, q, observed, equilibrium, theta, cameras)
        width = .5 * (error[..., :half] + error[..., half:])
        center = .5 * (error[..., :half] - error[..., half:])
        return np.r_[1.5 * width.ravel(), .8 * center.ravel(), 1.5 * (theta - start) / scale]

    result = least_squares(loss, start, bounds=(lower, upper), loss='soft_l1',
                           f_scale=.72, max_nfev=50)
    if not result.success:
        raise ValueError(f'body calibration did not converge: {result.message}')
    return start, result.x


def image_diagnostics(decoder, records, equilibrium, cameras, theta):
    q, detected = observations(records)
    predicted = silhouette_error(decoder, q, detected, equilibrium, theta, cameras)
    half = len(DIRECTIONS) // 2
    width = .5 * (predicted[..., :half] + predicted[..., half:])
    return dict(sightings=len(q), support_rms_px=float(np.sqrt(np.mean(predicted ** 2))),
                width_rms_px=float(np.sqrt(np.mean(width ** 2))))


def check_physical(arena, decoder, records, body, equilibrium, start, fitted):
    """After fit, grade tool response on separate optical probe trajectories."""
    values = []
    regime = 'changed' if body == 'alternate' else 'standard'
    absent = np.array((5., 5., 5.))
    for record in records:
        obs = arena.reset(seed=record['seed'], split='held', regime=regime, render=False)
        true_pose, original, visual = [], [], []
        for tick, action in enumerate(record['actions']):
            if tick % 3 == 0:
                normal = arena._data.geom_xmat[arena._blade].reshape(3, 3)[:, 2]
                if normal[2] < 0.:
                    normal = -normal
                true_pose.append(np.r_[arena._data.geom_xpos[arena._blade], normal[:2]])
                q = obs['joints'][None, :]
                for fit, storage in ((start, original), (fitted, visual)):
                    center, predicted = tool_poses(decoder, q, equilibrium, fit)
                    storage.append(np.r_[center[0], predicted[0, :2]])
            prepare_step(arena, np.asarray(action), absent)
            obs = arena._observation(False)
        values.append(np.stack((true_pose, original, visual)))
    traces = np.concatenate(values, axis=1)
    errors = traces[1:] - traces[0]
    movement = np.concatenate([block[:, 3:] - block[:, :-3] for block in values], axis=1)
    movement_error = movement[1:] - movement[0]
    return dict(tool_position_rms_mm=1000 * np.sqrt(np.mean(errors[:, :, :3] ** 2, axis=(1, 2))),
                normal_xy_rms_deg=np.rad2deg(np.sqrt(np.mean(errors[:, :, 3:] ** 2, axis=(1, 2)))),
                normal_motion_rms_deg=np.rad2deg(np.sqrt(np.mean(movement_error[:, :, 3:] ** 2, axis=(1, 2)))))


class MarkedProbe(WidePosture, MarkerBounce):
    """The ordinary five-axis body with three display-only planar marks."""


def marker_cameras(arena):
    arena.probe_images()
    cameras = []
    for name in _CALIBRATION_CAMERAS:
        arena._renderer.update_scene(arena._data, camera=name, scene_option=arena._probe_option)
        cameras.append(Camera.from_fixed_renderer(arena._renderer.scene))
    return cameras


def marker_frame(arena, cameras):
    seen = markers(arena)
    if seen is None:
        return None
    pixels = seen.reshape(2, 3, 2) * arena.height
    pixels -= np.array(((arena.width - 1) / 2, (arena.height - 1) / 2))
    pixels[:, :, 1] *= -1
    xyz, _, _ = triangulate(cameras, pixels.transpose(1, 0, 2), tolerance=.008)
    if not np.isfinite(xyz).all():
        return None
    a, b, c = xyz
    normal = np.cross(b - a, c - a)
    area = np.linalg.norm(normal)
    if not .012 < area < .026:
        return None
    direction = normal / area
    direction = direction if direction[2] >= 0 else -direction
    return dict(pixels=pixels.tolist(), normal=direction.tolist())


def marker_observations(records):
    usable = [r for record in records for r in record['rows']]
    return np.asarray([r['encoders'] for r in usable]), np.asarray([r['normal'] for r in usable])


def fit_markers(decoder, equilibrium, records):
    q, normals = marker_observations(records)
    if len(q) < 40:
        raise ValueError(f'only {len(q)} frames contain three common blade points')
    scale = np.r_[[.20] * 2, [.042] * 10]
    _, rotation = decoder.links.read(q)

    def loss(params):
        fitted = np.einsum('nij,nj->ni', rotation, local_normals(decoder, q, equilibrium, params))
        fitted = np.where(fitted[:, 2, None] >= 0., fitted, -fitted)
        return np.r_[((fitted - normals) / .013).ravel(), 2.0 * params / scale]

    found = least_squares(loss, np.zeros(12), bounds=(-2 * scale, 2 * scale), loss='soft_l1')
    if not found.success:
        raise ValueError(f'marker fit did not converge: {found.message}')
    return np.zeros(20), np.r_[found.x, np.zeros(8)]


def marker_diagnostics(decoder, equilibrium, records, theta):
    q, observed = marker_observations(records)
    _, fitted = tool_poses(decoder, q, equilibrium, theta)
    return dict(sightings=len(q),
                observed_normal_rms_deg=np.rad2deg(np.sqrt(np.mean((observed[:, :2] - fitted[:, :2]) ** 2))))


def calibrate(path, *, marked=False):
    policy = ToolEquilibrium(BUNDLE, WORKSPACE)
    receipt = dict(fit_seeds=FIT_SEEDS, independently_checked_seeds=CHECK_SEEDS, fit_ticks_per_clip=72,
                   target_reward_fitting=0, bundle_sha256=_hash(BUNDLE), bodies={})
    if marked:
        receipt['method'] = 'three optical marks'
    acquire, read = (marker_cameras, marker_frame) if marked else (stationary, outline_frame)
    for body, decoder in (('stock', policy.stock_decoder), ('alternate', policy.altered_decoder)):
        arena = (MarkedProbe(view='none', width=192, height=144, arm=body, calibration=True)
                 if marked else ProbeArm(body))
        try:
            tape, cameras = [], None
            for seed in (*FIT_SEEDS, *CHECK_SEEDS):
                clip, cameras = collect_probe(arena, seed, body, acquire, read, cameras)
                tape.append(clip)
            train, held = tape[:len(FIT_SEEDS)], tape[len(FIT_SEEDS):]
            if marked:
                start, fitted = fit_markers(decoder, policy.equilibrium, train)
                checks = [marker_diagnostics(decoder, policy.equilibrium, held, theta)
                          for theta in (start, fitted)]
            else:
                start, fitted = fit_geometry(decoder, train, policy.equilibrium, cameras)
                checks = [image_diagnostics(decoder, held, policy.equilibrium, cameras, theta)
                          for theta in (start, fitted)]
            physical = check_physical(arena, decoder, held, body, policy.equilibrium, start, fitted)
            row = dict(theta=fitted.tolist(), neutral=start.tolist(), held_image=checks,
                       held_metrology={key: value.tolist() for key, value in physical.items()}, clips=tape)
            receipt['bodies'][body] = row
            print('geometry', body, checks, row['held_metrology'], flush=True)
        finally:
            arena.close()
    _write(path, receipt)


def frozen_metrology(fitted, output):
    """Verify general angle changes on held probe clips after the marked coefficients freeze."""
    result = _read(fitted)
    policy = ToolEquilibrium(BUNDLE, WORKSPACE)
    values = {}
    for body, decoder in (('stock', policy.stock_decoder), ('alternate', policy.altered_decoder)):
        arena = MarkedProbe(view='none', width=192, height=144, arm=body, calibration=True)
        try:
            row = result['bodies'][body]
            clips = [clip for clip in row['clips'] if clip['seed'] in CHECK_SEEDS]
            measures = check_physical(arena, decoder, clips, body, policy.equilibrium,
                                      np.asarray(row['neutral']), np.asarray(row['theta']))
            values[body] = {key: value.tolist() for key, value in measures.items()}
        finally:
            arena.close()
    _write(output, dict(fit=_hash(fitted), changes_in_normal_over_ticks=9, checked=values))
    print(values, flush=True)


def correction(decoder, q, equilibrium, fitted, neutral):
    # Apply the learned offset only within its wide probe neighborhood.
    query = np.clip(q, equilibrium - .36, equilibrium + .36)[None, :]
    _, corrected = tool_poses(decoder, query, equilibrium, fitted)
    _, prior = tool_poses(decoder, query, equilibrium, neutral)
    return (corrected - prior)[0, :2]


def observed_face(observe, policy, model, gauge, obs):
    state = observe(obs)
    q = np.asarray(obs['joints'])
    for body, decoder, sign in (('alternate', policy.altered_decoder, 1.),
                                ('stock', policy.stock_decoder, -float(gauge))):
        if sign:
            state[9:11] += sign * correction(decoder, q, policy.equilibrium, *model[body])
    return state


def qualify(arm, server, plan, policy, model, mode, replay=None):
    launch = fixture(plan, server)
    render = replay is None and mode != 'hold'
    obs = arm.reset(plan, launch[0], render)
    if render:
        policy.begin_release(obs)
        if policy.calibration_index and mode in ('absolute', 'stock_gauge'):
            policy.sensor.observe = partial(observed_face, policy.sensor.observe, policy, model, mode == 'stock_gauge')
    preparation = plan['prep_ticks']
    gravity = dict(plan['changes'])
    actions, events = [], []
    for tick in range(LIVE_CAP + preparation):
        live_tick = tick - preparation
        if live_tick in gravity:
            events.append(arm.switch_gravity(live_tick, gravity[live_tick]))
        if replay is not None:
            action = np.asarray(replay[tick])
        elif render:
            action = policy.act(obs)
        else:
            action = np.zeros(5)
        actions.append(action.tolist())
        obs, done = advance(arm, action, launch, tick, preparation, render)
        if done:
            break
    return dict(actions=actions, events=events, metrics=arm.arena.judge_metrics())


def held_score(rows):
    summary = {}
    for mode in PHASES:
        scores = [row['results'][mode]['metrics'] for row in rows]
        counts = [row['rebounds'] for row in scores]
        lifetimes = np.array([judge['elapsed_s'] - row['plan']['prep_ticks'] / 60
                              for judge, row in zip(scores, rows, strict=True)])
        summary[mode] = dict(total=sum(counts), first=sum(score >= 1 for score in counts),
                             second=sum(score >= 2 for score in counts), counts=counts,
                             longest=[row['longest_consecutive_rebounds'] for row in scores],
                             five_seconds=int(np.sum(lifetimes >= 5 - 1.e-8)),
                             eight_seconds=int(np.sum(lifetimes >= 8 - 1.e-8)),
                             median_seconds=float(np.median(lifetimes)))
    return summary


def plan_held():
    drawn = schedule(HELD_SEED, 12)
    return [x for x in drawn if x['body'] == 'alternate'] + [x for x in drawn if x['body'] == 'stock'][:4]


def evaluate(path, fitted):
    fit = _read(fitted)
    if fit['bundle_sha256'] != _hash(BUNDLE):
        raise ValueError('carried policy differs from optical calibration')
    report = dict(held_seed=HELD_SEED, fit=_hash(fitted), horizon=LIVE_CAP, conditions=PHASES,
                  target_reward_updates=0, public_calibration_ticks=2 * len(FIT_SEEDS) * 72, cases=[])
    alternate = fit['bodies']['alternate']
    optical = 'observed_normal_rms_deg' if fit.get('method') else 'width_rms_px'
    before, after = (row[optical] for row in alternate['held_image'])
    physical_before, physical_after = alternate['held_metrology']['normal_xy_rms_deg']
    report['gate'] = dict(passed=bool(after < .9 * before and physical_after < .8 * physical_before),
                          optical=optical, optical_before=before, optical_after=after,
                          physical_before_deg=physical_before, physical_after_deg=physical_after)
    if not report['gate']['passed']:
        _write(path, report)
        print('held geometry gate failed', report['gate'], flush=True)
        return
    model = {name:(np.asarray(row['theta']), np.asarray(row['neutral']))
             for name, row in fit['bodies'].items()}
    policies = ToolEquilibrium(BUNDLE, WORKSPACE)
    policies.mode = 'workspace_local'
    arms = {name:InstalledArm(name) for name in model}
    for arm in arms.values():
        arm.arena.max_steps = LIVE_CAP
    server = BounceArena(view='none')
    try:
        for case in plan_held():
            arm = arms[case['body']]
            conditions = {}
            for mode in PHASES:
                if case['body'] == 'stock' and mode in ('absolute', 'stock_gauge'):
                    # A matching recalled stock observer receives the unchanged native readout.
                    conditions[mode] = conditions['frozen']
                    continue
                played = qualify(arm, server, case, policies, model, mode)
                replay = qualify(arm, server, case, policies, model, mode, played['actions'])
                if digest_json(played) != digest_json(replay):
                    raise ValueError(f"game replay differs in {mode} on {case['learner_seed']}")
                conditions[mode] = played
            report['cases'].append(dict(plan=case, results=conditions))
            report['summary'] = {name:held_score([row for row in report['cases'] if row['plan']['body'] == name])
                                 for name in {row['plan']['body'] for row in report['cases']}}
            _write(path, report)
            print('launch', case['body'], case['learner_seed'],
                  [x['metrics']['rebounds'] for x in conditions.values()], flush=True)
    finally:
        server.close()
        for arm in arms.values():
            arm.arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--calibrate', type=Path)
    parser.add_argument('--marked', type=Path)
    parser.add_argument('--evaluate', type=Path)
    parser.add_argument('--fit', type=Path, default=FIT)
    parser.add_argument('--metrology', type=Path)
    options = parser.parse_args()
    if options.calibrate:
        calibrate(options.calibrate)
    elif options.marked:
        calibrate(options.marked, marked=True)
    elif options.metrology:
        frozen_metrology(options.fit, options.metrology)
    elif options.evaluate:
        evaluate(options.evaluate, options.fit)
    else:
        audit(options.audit)


if __name__ == '__main__':
    main()
