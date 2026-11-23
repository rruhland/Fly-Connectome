"""Test a geometry-only collision correction for the wrist body servo."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .tt19_so101 import _GRASP_POS
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import _INC
from .tt24_clearance_audit import ClearanceAudit, _write, interacting_geoms, summarize, surface
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_world_arm import Grounding
from .tt24_world_prior import WorldPrior


PRIOR = Path('artifacts/tt24/gantry-source.npz')
HELD = tuple(range(258_201, 258_217))
MOVING_SURFACES = ('paddle_handle', 'camera_mounting_hardware', 'hand_or_jaw', 'arm_link')
PREDICTION_SECONDS = (.028, .054)


class ShapeProxy:
    """A static vendor and known-tool model with its own unsimulated data buffer."""

    def __init__(self):
        model_source = BounceArena(view='none')
        self.model = model_source._model
        self.data = mujoco.MjData(self.model)
        self.ball = model_source._ball_geom
        self.ball_q = self.model.joint('ball_free').qposadr[0]
        self.gripper = model_source._grip
        self.joints = model_source._joint_position[:5].copy()
        self.jaw = model_source._joint_position[5]
        self.paddle = model_source._paddle
        self.blade = model_source._blade
        self.data.qpos[self.jaw] = -.10
        checked = interacting_geoms(self.model, self.ball)
        self.solids = np.array([i for i in checked if surface(self.model.geom(int(i)).name) in MOVING_SURFACES])
        self.sizes = self.model.geom_rbound[self.solids]
        model_source.close()

    def _pose(self, q, ball=None):
        data = self.data
        data.qpos[self.joints] = q
        if ball is not None:
            data.qpos[self.ball_q:self.ball_q + 3] = ball
        mujoco.mj_kinematics(self.model, data)
        grip = self.gripper
        data.qpos[self.paddle:self.paddle + 3] = data.xpos[grip] + data.xmat[grip].reshape(3, 3) @ _GRASP_POS
        data.qpos[self.paddle + 3:self.paddle + 7] = data.xquat[grip]
        mujoco.mj_kinematics(self.model, data)

    def align(self, q, centered_position, radius):
        self._pose(q)
        self.offset = centered_position - self.data.geom_xpos[self.blade]
        self.radius = float(radius)
        self.model.geom_size[self.ball, 0] = self.radius

    def nearest(self, q, ball):
        self._pose(q, ball - self.offset)
        ball_center = self.data.geom_xpos[self.ball]
        bounds = np.linalg.norm(self.data.geom_xpos[self.solids] - ball_center, axis=1) - self.sizes - self.radius
        candidates = self.solids[bounds < .043]
        return min(((mujoco.mj_geomDistance(self.model, self.data, self.ball, int(item), .13, None), int(item))
                    for item in candidates), default=(.13, -1))

    def distance(self, q, ball, item):
        self._pose(q, ball - self.offset)
        return mujoco.mj_geomDistance(self.model, self.data, self.ball, item, .13, None)


class ClearanceStep:
    """Project a commanded joint target away from a predicted moving-body contact."""

    def __init__(self, geometry, grounding, radius, margin):
        self.geometry, self.grounding, self.margin = geometry, grounding, margin
        self.geometry.align(grounding.home, grounding.origin[:3], radius)
        self.delta = np.eye(5) * .009
        self.applied = []
        self.risks = Counter()

    def _risks(self, current, velocity, command_target, ball, flow):
        for time in PREDICTION_SECONDS:
            response = 1. - np.exp(-time / .045)
            future = current + .35 * time * velocity + response * (command_target - current)
            obstacle = ball + time * flow
            distance, geometry = self.geometry.nearest(future, obstacle)
            yield distance, geometry, future, obstacle, response

    def command(self, base_command, issued_before, obs, ball, flow, has_sighting):
        goal = self.grounding.issued.copy()
        if not has_sighting:
            return base_command
        q, dq = obs['joints'], np.clip(obs['joint_velocity'], -5.2, 5.2)
        risk = min(self._risks(q, dq, goal, ball, flow), key=lambda row: row[0])
        gap, geom, future, obstacle, blend = risk
        if gap >= self.margin or geom < 0:
            return base_command
        group = surface(self.geometry.model.geom(geom).name)
        self.risks[group] += 1
        gradient = np.array([self.geometry.distance(future + d, obstacle, geom) - gap for d in self.delta]) * (blend / .009)
        jacobian = self.grounding.jacobian(goal) / self.grounding.scale[:, None]
        preference = np.eye(5) + .16 * jacobian.T @ jacobian
        difference = np.linalg.solve(preference, gradient)
        strength = max(.0005, float(gradient @ difference))
        adjusted = goal + np.clip(difference * ((self.margin - gap) / strength), -.070, .070)
        adjusted = np.clip(adjusted, self.grounding.home - .55, self.grounding.home + .55)
        command = np.clip((adjusted - issued_before) / _INC, -.75, .75)
        target = issued_before + _INC * command
        improved = min(self._risks(q, dq, target, ball, flow), key=lambda row: row[0])[0]
        if improved < gap + .002:
            return base_command
        self.grounding.issued = target
        self.applied.append(dict(group=group, gap_m=float(gap), corrected_gap_m=float(improved),
                                 projected_joint_change_rad=(target - goal).tolist()))
        return command


def episode(arena, geometry, prior, calibration, seed, mode, margin):
    optics = load_public(calibration) if mode != 'stationary' else None
    obs = arena.reset(seed=seed, split='held', render=optics is not None)
    ground = Grounding(optics.decoder, obs['joints']) if optics else None
    guard = ClearanceStep(geometry, ground, optics.decoder.radius, margin) if mode.startswith('guard') else None
    commands = []
    for _ in range(arena.max_steps):
        command = np.zeros(arena.action_size)
        if ground:
            issued = ground.issued.copy()
            missing_before = optics.gaps
            state = optics.observe(obs)
            command = ground.target(prior.target(state))
            if guard:
                ball = state[:3] + ground.origin[:3]
                flow = state[3:6]
                visible = missing_before == optics.gaps
                if mode == 'guard_diagnostic_clean_ball':
                    ball = arena._ball_position + geometry.offset
                    flow = arena._ball_velocity.copy()
                    visible = True
                command = guard.command(command, issued, obs, ball, flow, visible)
        commands.append(command.tolist())
        obs, _, done = arena.step(command, render=optics is not None)
        if done:
            break
    return dict(judge=arena.judge_metrics(), audit=arena.contact_receipt(), commands=commands,
                corrections=guard.applied if guard else [], foresight_risks=dict(guard.risks) if guard else {})


def verify(arena, row):
    arena.reset(seed=row['judge']['seed'], split='held', render=False)
    for action in row['commands']:
        _, _, done = arena.step(action, render=False)
    assert done and arena.judge_metrics() == row['judge']
    assert arena.contact_receipt() == row['audit']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24/clearance-servo.json.gz'))
    parser.add_argument('--calibration', type=Path, default=FUSED_OUTPUT)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--margin', type=float, default=.014)
    parser.add_argument('--oracle-control', action='store_true')
    args = parser.parse_args()
    if not 0. < args.margin <= .04:
        parser.error('choose a clearance margin between 0 and .04 m')
    arena = ClearanceAudit(view='wrist', width=192, height=144)
    correction = ShapeProxy()
    actor = WorldPrior.load(args.prior)
    modes = ['stationary', 'public_servo', 'guard_public']
    if args.oracle_control:
        modes.append('guard_diagnostic_clean_ball')
    results = dict(schema='tt24-clearance-servo-1', held_seeds=HELD, margin_m=args.margin,
                   prediction_s=PREDICTION_SECONDS, prior=str(args.prior), calibration=str(args.calibration),
                   evaluation='unaltered stock physics and referee', target_practice=False, outcomes={})
    try:
        for mode in modes:
            rows = [episode(arena, correction, actor, args.calibration, seed, mode, args.margin) for seed in HELD]
            for row in rows:
                verify(arena, row)
            report = summarize(rows)
            report.update(mean_rebounds=float(np.mean([row['judge']['rebounds'] for row in rows])),
                          longest=max(row['judge']['longest_consecutive_rebounds'] for row in rows),
                          median_consecutive=float(np.median([row['judge']['longest_consecutive_rebounds'] for row in rows])),
                          motion_corrections=sum(len(row['corrections']) for row in rows),
                          corrected_episodes=sum(bool(row['corrections']) for row in rows), exact_replays=len(rows))
            results['outcomes'][mode] = dict(summary=report, episodes=rows)
            print(mode, report, flush=True)
    finally:
        arena.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(args.out, results)


if __name__ == '__main__':
    main()
