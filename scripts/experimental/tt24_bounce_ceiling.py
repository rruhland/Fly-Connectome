"""Privileged physical feasibility audit for TT24 paddle bouncing.

This controller reads MuJoCo state and stays outside independent unaided
training and evaluation. Any demonstrations from it require a teacher label.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .tt19_so101 import WRIST_CAMERA
from .tt24_bounce import BALL_RADIUS, BounceArena, mujoco


DEVELOPMENT_SEEDS = tuple(range(242_001, 242_017))
FRESH_SEEDS = tuple(range(243_001, 243_017))
RESET_SCREEN_SEEDS = tuple(range(244_001, 244_193))
SHORT_ARC_FRESH_SEEDS = tuple(range(246_001, 246_017))


@dataclass(frozen=True)
class CeilingSettings:
    height: float = .116
    servo_lead: float = .023
    follow_fraction: float = .64
    return_fraction: float = .65
    tilt_limit: float = .14
    position_weight: tuple[float, ...] = (4., 3., 5.)
    regularization: float = .22
    correction: float = .16
    return_limit: float = .30
    rate_limit: float = .70
    vertical_gain: float = 1.28


class PrivilegedCeiling:
    """Solve paddle pose and speed from the true ballistic intercept and Jacobian."""

    def __init__(self, arena, settings=CeilingSettings()):
        self.arena = arena
        self.settings = settings
        self.model, self.data = arena._model, arena._data
        self.qadr = arena._joint_position[:arena.action_size]
        self.dofs = arena._joint_velocity[:arena.action_size]
        self.max_change = arena._increments[:arena.action_size] * (arena.dt / self.model.opt.timestep)
        self.jac_position = np.zeros((3, self.model.nv))
        self.jac_rotation = np.zeros_like(self.jac_position)
        self.center = self.data.geom_xpos[arena._blade].copy()
        self.origin = self.data.qpos[self.qadr].copy()
        self.position_weight = np.array(settings.position_weight)
        self.restitution = .945 - .0035 * arena._drop.damping
        normal = self._normal()
        jacobian = self._jacobian(self.center, normal)
        hessian = jacobian.T @ jacobian
        hessian.flat[::len(self.dofs) + 1] += settings.regularization**2
        self.inverse = np.linalg.solve(hessian, jacobian.T)
        self.neutral_tilt = normal[:2].copy()
        self.track = {}

    def _normal(self):
        normal = self.data.geom_xmat[self.arena._blade].reshape(3, 3)[:, 2]
        return normal if normal[2] >= 0. else -normal

    def _jacobian(self, blade, normal):
        mujoco.mj_jac(self.model, self.data, self.jac_position, self.jac_rotation, blade, self.arena._grip)
        angular = self.jac_rotation[:, self.dofs]
        tilt_x = angular[1] * normal[2] - angular[2] * normal[1]
        tilt_y = angular[2] * normal[0] - angular[0] * normal[2]
        return np.vstack((self.position_weight[:, None] * self.jac_position[:, self.dofs], tilt_x, tilt_y))

    def _intercept(self, ball, velocity, height):
        gravity = 9.81
        fall_distance = ball[2] - height
        radicand = velocity[2] ** 2 + 2. * gravity * fall_distance
        arrival = max(-.04, (velocity[2] + math.sqrt(max(0., radicand))) / gravity)
        drag = -self.arena._drag_force_scale / self.model.body_mass[self.arena._ball_body]
        # Approximate the small quadratic-drag correction over one flight.
        speed = np.linalg.norm(velocity)
        horizontal_factor = 1. / (1. + .5 * drag * speed * max(0., arrival))
        xy = ball[:2] + velocity[:2] * arrival * horizontal_factor
        vxy = velocity[:2] * horizontal_factor**2
        vz = (velocity[2] - gravity * arrival) * (1. - .12 * abs(arrival))
        return arrival, xy, vxy, min(-.22, vz)

    def command(self, trace=False):
        settings, arena, model, data = self.settings, self.arena, self.model, self.data
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        blade = data.geom_xpos[arena._blade]
        normal = self._normal()
        ball = arena._ball_position
        velocity = arena._ball_velocity
        impact_level = self.center[2] + BALL_RADIUS + model.geom_size[arena._blade, 1]
        arrival, point, incoming_xy, incoming_z = self._intercept(ball, velocity, impact_level)
        apex_speed = math.sqrt(2. * 9.81 * settings.height) * 1.022
        hit_speed = np.clip((apex_speed + self.restitution * incoming_z) / (1. + self.restitution), -.08, .36)
        frequency = 9.81 * math.pi / max(.9, -incoming_z)
        z = self.center[2] - settings.vertical_gain * hit_speed * math.sin(frequency * (arrival - settings.servo_lead)) / frequency
        remaining_xy = np.clip(point - self.center[:2], -.091, .091)
        following_xy = np.clip(settings.follow_fraction * remaining_xy, -.063, .063)
        center_xy = self.center[:2] + following_xy
        return_speed = np.clip(-settings.return_fraction * remaining_xy / (2. * apex_speed / 9.81),
                               -settings.return_limit, settings.return_limit)
        ball_spin = data.xmat[arena._ball_body].reshape(3, 3) @ arena._ball_spin
        bottom_spin = BALL_RADIUS * np.array((-ball_spin[1], ball_spin[0]))
        tangent_loss = (2. / 7.) * (incoming_xy + bottom_spin)
        normal_response = (1. + self.restitution - 2. / 7.) * (hit_speed - incoming_z)
        tilt = np.clip((return_speed - incoming_xy + tangent_loss) / max(.4, normal_response),
                       -settings.tilt_limit, settings.tilt_limit)
        goal = np.r_[center_xy, z]
        error = np.r_[self.position_weight * (goal - blade), tilt - normal[:2]]
        displacement = np.r_[self.position_weight * (goal - self.center), tilt - self.neutral_tilt]
        target = self.origin + self.inverse @ (displacement + settings.correction * error)
        requested = (target - arena._control[:arena.action_size]) / self.max_change
        if trace:
            self._jacobian(blade, normal)
            face_offset = ball - blade
            plane_distance = float(face_offset @ normal)
            radial_offset = float(np.linalg.norm(face_offset - plane_distance * normal))
            force_caps = model.actuator_forcerange[:arena.action_size, 1]
            self.track = dict(predicted_arrival_s=float(arrival), estimated_strike_xy=point.tolist(),
                              desired_position=goal.tolist(), desired_tilt=tilt.tolist(),
                              desired_vertical_speed=float(hit_speed), blade=blade.tolist(), normal=normal.tolist(),
                              ball=ball.tolist(), ball_velocity=velocity.tolist(),
                              blade_velocity=(self.jac_position[:, self.dofs] @ data.qvel[self.dofs]).tolist(),
                              normalized_requested_rates=requested.tolist(), blade_radial_offset_m=radial_offset,
                              ball_center_to_face_plane_m=plane_distance,
                              servo_force_fraction=(np.abs(data.actuator_force[:arena.action_size]) / force_caps).tolist())
        return np.clip(requested, -settings.rate_limit, settings.rate_limit)


def rollout(arena, seed, settings=CeilingSettings(), trace=False):
    arena.reset(seed=seed, split='held')
    controller = PrivilegedCeiling(arena, settings)
    tape = []
    saturations = 0
    steps = 0
    for step in range(arena.max_steps):
        command = controller.command(trace)
        saturations += int(np.abs(command).max() >= settings.rate_limit)
        steps = step + 1
        time = float(arena._data.time)
        _, reward, done = arena.step(command)
        if trace:
            tape.append(dict(time_s=time, command=command.tolist(), reward=reward, **controller.track))
        if done:
            break
    return dict(**arena.judge_metrics(), saturation_fraction=saturations / steps, trace=tape)


def _loss(result, arena):
    if result['termination'] == 'timeout':
        return dict(kind='horizon')
    tape = result['trace']
    events = result['rebound_events']
    since = events[-1]['apex_time'] if events else 0.
    contact_level = BALL_RADIUS + float(arena._model.geom_size[arena._blade, 1])
    blade_radius = float(arena._model.geom_size[arena._blade, 0])
    approaching = [row for row in tape if row['time_s'] >= since and row['ball_velocity'][2] < 0.]
    close = min(approaching, key=lambda row: abs(row['ball_center_to_face_plane_m'] - contact_level), default=None)
    if close is None:
        return dict(kind='no_downward_crossing', collider=result['other_collider'])
    force_fraction = max(close['servo_force_fraction'])
    radius = close['blade_radial_offset_m']
    separation = abs(close['ball_center_to_face_plane_m'] - contact_level)
    if separation > .032:
        kind = 'no_close_face_crossing'
    elif radius > blade_radius:
        kind = 'outside_blade_radius'
    elif force_fraction >= .98:
        kind = 'near_face_with_force_saturation'
    else:
        kind = 'near_face_without_force_saturation'
    return dict(kind=kind, collider=result['other_collider'], radial_offset_m=radius,
                closest_plane_separation_error_m=separation, max_servo_force_fraction=force_fraction,
                blade_vz_m_s=close['blade_velocity'][2], ball_vz_m_s=close['ball_velocity'][2])


def _summarize(rows):
    runs = np.array([row['longest_consecutive_rebounds'] for row in rows])
    rises = [event['rise_m'] for row in rows for event in row['rebound_events']]
    air = [event['airborne_s'] for row in rows for event in row['rebound_events'] if event['airborne_s'] is not None]
    power = [row['force_saturated_step_fraction'] for row in rows]
    return dict(episodes=len(rows), mean_consecutive=float(runs.mean()), median_consecutive=float(np.median(runs)),
                min_consecutive=int(runs.min()), max_consecutive=int(runs.max()),
                at_least_ten=int(np.count_nonzero(runs >= 10)), at_least_fifteen=int(np.count_nonzero(runs >= 15)),
                median_rise_m=float(np.median(rises)) if rises else None,
                median_airborne_s=float(np.median(air)) if air else None,
                mean_command_limit_step_fraction=float(np.mean([row['saturation_fraction'] for row in rows])),
                mean_force_limit_step_fraction=float(np.mean(power)),
                outcomes=dict(Counter(row['loss']['kind'] for row in rows)))


def _cohort(arena, seeds, settings):
    rows = []
    tape = None
    for seed in seeds:
        result = rollout(arena, seed, settings, trace=True)
        result['loss'] = _loss(result, arena)
        trajectory = result.pop('trace')
        result['force_saturated_step_fraction'] = float(np.mean([
            max(step['servo_force_fraction']) >= .98 for step in trajectory]))
        rows.append(result)
        if tape is None or result['longest_consecutive_rebounds'] > tape['result']['longest_consecutive_rebounds']:
            tape = dict(result=result, steps=trajectory)
    return dict(summary=_summarize(rows), episodes=rows), tape


def _wrist_ball(arena):
    rgb = arena.render_wrist()
    camera = arena._data.camera(WRIST_CAMERA)
    point = camera.xmat.reshape(3, 3).T @ (arena._ball_position - camera.xpos)
    depth = -float(point[2])
    if depth <= BALL_RADIUS:
        return dict(visible=False, orange_pixels=0, ball_center_px=None, full_disc_in_frame=False)
    fovy = arena._model.camera(WRIST_CAMERA).fovy[0]
    focal = arena.height / (2. * math.tan(math.radians(fovy) / 2.))
    center = ((arena.width - 1) / 2. + focal * point[0] / depth,
              (arena.height - 1) / 2. - focal * point[1] / depth)
    radius = focal * BALL_RADIUS / math.sqrt(depth**2 - BALL_RADIUS**2)
    left, right = max(0, math.floor(center[0] - 1.4 * radius)), min(arena.width, math.ceil(center[0] + 1.4 * radius))
    top, bottom = max(0, math.floor(center[1] - 1.4 * radius)), min(arena.height, math.ceil(center[1] + 1.4 * radius))
    pixels = 0
    if left < right and top < bottom:
        red, green, blue = np.moveaxis(rgb[top:bottom, left:right], -1, 0)
        pixels = int(np.count_nonzero((red > 68) & (green > .20 * red) & (green < .68 * red) & (blue < .34 * red)))
    full = bool(radius <= center[0] <= arena.width - 1. - radius
                and radius <= center[1] <= arena.height - 1. - radius)
    return dict(visible=pixels >= 5, orange_pixels=pixels, ball_center_px=center, full_disc_in_frame=full)


def _screen(arena, settings, seeds):
    zero_trials = []
    action = np.zeros(arena.action_size)
    for seed in seeds:
        arena.reset(seed=seed, split='held')
        for _ in range(arena.max_steps):
            _, reward, done = arena.step(action)
            if reward or done:
                break
        if reward:
            continue
        passive = arena.judge_metrics()

        arena.reset(seed=seed, split='held')
        controller = PrivilegedCeiling(arena, settings)
        release = _wrist_ball(arena)
        visibility = [release]
        for _ in range(arena.max_steps):
            _, reward, done = arena.step(controller.command())
            if arena._paddle_contacts == 0 and not done:
                visibility.append(_wrist_ball(arena))
            if reward or done:
                break
        active = arena.judge_metrics()
        first = active['rebound_events'][0] if active['rebound_events'] else None
        zero_trials.append(dict(seed=seed, passive=passive, active_first_rebound=first,
                                active_paddle_contacts=active['paddle_contacts'], active_termination=active['termination'],
                                active_intercepts_first_impact=bool(first and first['impact_index'] == 1),
                                release=release, preimpact_frames=len(visibility),
                                visible_preimpact_frames=sum(frame['visible'] for frame in visibility)))
    return dict(held_trials=len(seeds), zero_passive=len(zero_trials),
                release_visible=sum(row['release']['visible'] for row in zero_trials),
                release_fully_framed=sum(row['release']['full_disc_in_frame'] for row in zero_trials),
                first_impact_rescued=sum(row['active_intercepts_first_impact'] for row in zero_trials),
                episodes=zero_trials)


def _write(path, payload):
    contents = json.dumps(payload, separators=(',', ':'))
    with gzip.open(path, 'wt', encoding='utf-8') as output:
        output.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--reset-screen', type=int, default=len(RESET_SCREEN_SEEDS))
    parser.add_argument('--short-arc', action='store_true', help='run a separate 65 mm target on disjoint fresh seeds')
    args = parser.parse_args()
    if not 1 <= args.reset_screen <= len(RESET_SCREEN_SEEDS):
        parser.error(f'--reset-screen must be between 1 and {len(RESET_SCREEN_SEEDS)}')
    args.out.mkdir(parents=True, exist_ok=True)
    settings = CeilingSettings(height=.065, regularization=.26) if args.short_arc else CeilingSettings()
    fresh_seeds = SHORT_ARC_FRESH_SEEDS if args.short_arc else FRESH_SEEDS
    profile = 'short_arc' if args.short_arc else 'standard'
    evidence = dict(evaluation_only=True, privileged=True, profile=profile, settings=asdict(settings),
                    mujoco=mujoco.__version__, arms={})
    tapes = dict(evaluation_only=True, privileged=True, profile=profile, arms={})
    for arm in ('stock', 'alternate'):
        arena = BounceArena(view='none', arm=arm)
        try:
            development, _ = _cohort(arena, DEVELOPMENT_SEEDS, settings)
            fresh, tape = _cohort(arena, fresh_seeds, settings)
            evidence['arms'][arm] = dict(development=development, fresh=fresh)
            tapes['arms'][arm] = tape
            print(arm, 'development', development['summary'], flush=True)
            print(arm, 'fresh', fresh['summary'], flush=True)
            if not args.short_arc:
                screen = _screen(arena, settings, RESET_SCREEN_SEEDS[:args.reset_screen])
                evidence['arms'][arm]['zero_reset_screen'] = screen
                print(arm, 'zero-reset', {key: value for key, value in screen.items() if key != 'episodes'}, flush=True)
        finally:
            arena.close()
    stem = 'bounce-ceiling-short' if args.short_arc else 'bounce-ceiling'
    _write(args.out / f'{stem}.json.gz', evidence)
    _write(args.out / f'{stem}-trajectories.json.gz', tapes)


if __name__ == '__main__':
    main()
