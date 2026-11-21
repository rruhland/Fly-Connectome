"""Evaluator-only, finite SO-101 reachability search after a rival strike."""

from copy import copy, deepcopy
from itertools import product

import numpy as np

from .tt4_arena import AIR_DRAG, BALL_INERTIA, BALL_MASS, MAGNUS_COEFFICIENT, SPIN_DECAY, TABLE_HEIGHT, mujoco
from .tt19_factory import CONTACT_OFFSET, FactoryModel, HOME, moving_joint_rate
from .tt19_model_rollout import physical_ranking
from .tt20_curriculum import CLAMP_SHIFT, FRAME, apparatus
from .tt20_opponent import RivalSearch
from .tt21_coach import receive_window


PHASE = ('steps', 'contacts', 'legal_returns', 'rival_returns', 'rival_contacts', 'server', 'service_contact',
         'legal_serve', '_last_hitter', '_received_bounce', '_serving', '_server_bounce', '_serve_net',
         '_waiting_for_strike', '_toss_height', '_toss_apex', '_rival_attempt', '_rival_bias', '_rival_target',
         '_previous_touches', '_decoy_phase', '_rng', 'reward', 'terminated', 'truncated', 'reason')


def reply_released(arena):
    return (arena._last_hitter == -1 and arena.velocity[1] < -.25
            and arena._racket_ids[-1] not in arena._previous_touches)


class Reachability:
    def __init__(self):
        self.arena = apparatus(False)
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        arena = self.arena
        self.dry = RivalSearch(arena, arena._aim)
        self.dry.axes = arena._arm_ids[1][2][:5]
        self.dry.increment = arena._joint_increments[self.dry.axes]
        # A legal landing is scored normally. The evaluator keeps the far arm
        # parked on the far side after the learner's counterstrike.
        arena._choose_rival_response = self._park_rival
        self.ghost = copy(arena.model)
        self.ghost_data = mujoco.MjData(self.ghost)
        kept = {arena.model.geom(name).id for name in (*arena.table_panels, 'ball', 'floor', 'net')}
        disabled = list(set(range(arena.model.ngeom)) - kept)
        self.ghost.geom_contype[disabled] = 0
        self.ghost.geom_conaffinity[disabled] = 0
        self.surface = {arena.model.geom(name).id for name in arena.table_panels}

    def _park_rival(self):
        self.arena._rival_attempt = False

    def assess(self, live, *, search=True):
        if not reply_released(live) or live.terminated or live.truncated:
            raise ValueError('the reach evaluator requires a live reply released from the rival paddle')
        arena = self.arena
        source = mujoco.MjData(arena.model)
        mujoco.mj_copyData(source, arena.model, live.data)
        available = {key: deepcopy(getattr(live, key)) for key in PHASE if hasattr(live, key)}
        opening = deepcopy(live.events)

        def restore():
            mujoco.mj_copyData(arena.data, arena.model, source)
            for key, value in available.items():
                setattr(arena, key, deepcopy(value))
            arena.events = list(opening)

        restore()
        path = self.path()
        record = dict(tick=live.steps, ball=live.ball.tolist(), spin=live.spin.tolist(),
                      joints=live.joints.tolist(), targets=live.data.ctrl[self.dry.axes].tolist(),
                      incoming_landing=path['landing'], corridor_samples=len(path['window']),
                      horizon=path['horizon'], candidates=0, solver_contacts=0, solver_legal=0,
                      exact_trials=0, witnessed=False, contact_witness=False)
        if not search or not len(path['window']):
            return record
        goals, parameters = self.tapes(path)
        record['candidates'] = len(goals)
        if not len(goals):
            return record
        tapes, trajectories = self.dry.predict(np.asarray(goals))
        p, v = self.dry.position, self.dry.velocity
        scored = physical_ranking(trajectories[..., p:p+3] - FRAME, trajectories[..., v:v+3],
                                  arena._received_bounce, self.dry.micro)
        record['solver_legal'] = sum(row['legal'] for row in scored)
        record['solver_contacts'] = sum(row['hit'] for row in scored)
        order = sorted(range(len(scored)), key=lambda k: (not scored[k]['legal'], not scored[k]['hit'],
                                                         scored[k]['cost'], parameters[k]['alignment_m']))
        for index in order[:12]:
            restore()
            commands = []
            record['exact_trials'] += 1
            for command in tapes[index]:
                action = np.r_[command, 0.]
                commands.append(action.tolist())
                _, _, done, timeout, _ = arena.step(action, render=False)
                if done or timeout or arena.legal_returns > available['legal_returns']:
                    break
            record['contact_witness'] |= arena.contacts > available['contacts']
            if arena.legal_returns > available['legal_returns']:
                record.update(witnessed=True, exact_actions=commands, exact_events=arena.events[len(opening):],
                              exact_result=arena.point_result(), parameters=parameters[index])
                break
        return record

    def path(self):
        arena, model, data = self.arena, self.ghost, self.ghost_data
        mujoco.mj_copyData(data, model, arena.data)
        p, v = arena._ball_joint, arena._ball_dof
        body = arena._ball_body
        ball_id = arena.model.geom('ball').id
        pos = data.qpos[p:p+3]
        vel = data.qvel[v:v+3]
        floor = TABLE_HEIGHT + .025
        bounce = 0 if arena._received_bounce else None
        landing = None
        points, speeds = [pos.copy()], [vel.copy()]
        mx = min(58, arena.max_steps - arena.steps)
        for micro in range(1, mx * self.dry.micro + 1):
            quat = data.qpos[p+3:p+7]
            local = data.qvel[v+3:v+6]
            axis = np.cross(quat[1:], local)
            spin = local + 2 * (quat[0] * axis + np.cross(quat[1:], axis))
            data.xfrc_applied[body, :3] = BALL_MASS * (-AIR_DRAG * np.linalg.norm(vel) * vel
                                                     + MAGNUS_COEFFICIENT * np.cross(spin, vel))
            data.xfrc_applied[body, 3:] = -BALL_INERTIA * SPIN_DECAY * spin
            old_vertical = vel[2]
            mujoco.mj_step(model, data)
            if (old_vertical < 0. < vel[2] and -.81 < pos[1] < 0. and pos[2] < floor
                    and any(contact.efc_address >= 0 and ball_id in contact.geom
                            and self.surface.intersection(contact.geom) for contact in data.contact)):
                if bounce is not None:
                    break
                bounce = micro
                landing = pos.tolist()
            if pos[1] < -1.21 or pos[2] < TABLE_HEIGHT - .075 or (bounce is None and pos[1] < -.83):
                break
            points.append(pos.copy())
            speeds.append(vel.copy())
        points, speeds = np.asarray(points), np.asarray(speeds)
        window = [] if bounce is None else receive_window(points, speeds, bounce, lateral=.43,
                                                           depth=(-1.18, -.72), height=(.048, .49))
        return dict(points=points, speeds=speeds, window=window, landing=landing, horizon=mx)

    def tapes(self, path):
        arena = self.arena
        time = np.arange(path['horizon'])
        window = path['window']
        indices = np.unique(window[np.round(np.linspace(0, len(window) - 1, min(9, len(window)))).astype(int)])
        reference = arena.data.ctrl[self.dry.axes].copy()
        goals, parameters = [], []
        for micro in indices:
            xyz = path['points'][micro] - FRAME
            eta = micro / self.dry.micro
            vx, vy, _ = path['speeds'][micro]
            yaw = float(np.clip(.34 * vx / max(.4, -vy) - .35 * xyz[0], -.24, .24))
            for pitch, pace, phase in product((.12, .28, .46, .65, .85), (.014, .036, .060), (1.6, 3.6)):
                norm = np.array((np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)))
                face_center = xyz - CONTACT_OFFSET * norm
                solution = self.factory.solve(face_center, norm, reference if eta < 14. else HOME)
                target, normal, _, _ = self.factory.pose(solution)
                alignment = np.linalg.norm(target - face_center)
                if alignment > .115 or normal @ norm < .56:
                    continue
                velocity = moving_joint_rate(self.factory, solution, pace)
                run = np.clip(time - eta + phase, -4.4, 4.4)
                goals.append(solution + run[:, None] * velocity)
                parameters.append(dict(eta_frames=float(eta), pitch=pitch, phase=phase, pace=pace,
                                       alignment_m=float(alignment), target=xyz.tolist()))
        return goals, parameters

    def close(self):
        self.arena.close()
