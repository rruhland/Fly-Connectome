"""A separate comparison that plans only from a reconstructed binocular state."""

import numpy as np
import mujoco
from scipy.stats import qmc

from .tt19_so101 import GRIP_BODY, _GRASP_POS
from .tt25_second_ball import feed
from .tt26_grip_course import Grip, ReferenceReceiver
from .tt26_match_continuation import Snapshot, candidate, command
from .tt27_reply import TABLE_HEIGHT, Recall

SEARCH = qmc.Sobol(5, scramble=True, seed=23195).random_base2(4)


class ShadowPlanner:
    def __init__(self):
        self.arena = ReferenceReceiver()
        self.arena.max_steps = 55
        self.grip = Grip(self.arena.model)
        self.gripper = self.arena.model.body(GRIP_BODY).id
        self.tool_joint = int(self.arena.model.joint('learner_paddle_free').qposadr[0])
        self.tool_dof = int(self.arena.model.joint('learner_paddle_free').dofadr[0])
        self.joints, _, self.axes, _, _ = self.arena._arm_ids[1]
        self.joint_rates = self.arena.model.jnt_dofadr[self.arena.model.actuator_trnid[self.axes, 0]]

    def setup(self, point, velocity, spin, joints, rates, target, angle):
        arena = self.arena
        position = point + [0., 0., TABLE_HEIGHT]
        shot = {"id": 1, **feed(dict(ball=position, velocity=velocity, spin=spin))}
        self.grip.set(angle)
        arena.launch(shot)
        arena._received_bounce = True
        arena.legal_returns = 1
        arena.data.qpos[self.joints] = joints
        arena.data.ctrl[self.axes] = target
        arena.data.qvel[self.joint_rates] = rates
        mujoco.mj_kinematics(arena.model, arena.data)
        mujoco.mj_comPos(arena.model, arena.data)
        rotation = arena.data.xmat[self.gripper].reshape(3, 3)
        origin = arena.data.xpos[self.gripper] + rotation @ _GRASP_POS
        arena.data.qpos[self.tool_joint:self.tool_joint + 7] = np.r_[origin, arena.data.xquat[self.gripper]]
        jacp, jacr = np.zeros((3, arena.model.nv)), np.zeros((3, arena.model.nv))
        mujoco.mj_jac(arena.model, arena.data, jacp, jacr, origin, self.gripper)
        arena.data.qvel[self.tool_dof:self.tool_dof + 6] = np.r_[jacp @ arena.data.qvel, rotation.T @ jacr @ arena.data.qvel]
        mujoco.mj_forward(arena.model, arena.data)
        return Snapshot(arena)

    def search(self, snapshot, options, elapsed):
        tested = 0

        def legal(action):
            nonlocal tested
            goal, impact, rates, lead = action
            shifted = goal, impact - elapsed, rates, lead
            tested += 1
            return candidate(self.arena, snapshot, {"rival_actions": []}, shifted)["second"]

        for goal, impact, velocity, lead, jacobian in options:
            action = goal, impact, velocity, lead
            if legal(action):
                return action, tested, True
            for point in SEARCH:
                displacement = np.array([.085 * (point[0] - .5), .050 * (point[1] - .5), .100 * (point[2] - .5)])
                offset = jacobian @ displacement
                action = (goal + offset, impact - 2.6 + 5.2 * point[3], velocity * (.65 + .85 * point[4]), lead)
                if legal(action):
                    return action, tested, True
        return None, tested, False

    def close(self):
        self.arena.close()


class ShadowRecall(Recall):
    def __init__(self, eye, body, poses, flight, memory, angle, planner):
        super().__init__(eye, body, poses, flight, memory, "feedback")
        self.angle = angle
        self.planner = planner
        self.measured = None
        self.last = None
        self.planned = None
        self.scans = 0
        self.last_scan = None

    def observe_joints(self, joints):
        self.last, self.measured = self.measured, joints

    def act(self, tick):
        if self.planned is not None:
            return command(len(self.body.target), self.body.target, self.planned, tick, self.eye.incoming[0])
        restored = super().act(tick)
        if restored is None or not self.bounced or self.last_update != tick or self.eye.outgoing or self.scans >= 2:
            return restored
        if self.last_scan is not None and tick < self.last_scan + 2:
            return restored
        estimate = self.flight.estimate(self.eye.reply)
        if estimate is None:
            return restored
        _, p, v, _ = estimate
        path = self.flight.predict(estimate)
        if not len(path):
            return restored
        elapsed = tick - self.eye.incoming[0]
        options = []
        ordered = [self.example] + [e for e in self.memory if e is not self.example]
        for e in ordered:
            i = np.argmin(np.abs(path[:, 2] - e["hit"][1]))
            prediction = path[i]
            displacement = prediction[1:4] - e["hit"]
            if abs(displacement[1]) > .055 or np.linalg.norm(displacement) > .40 or prediction[0] - tick < 3.9:
                continue
            shift, error = self.poses.offset(e["q"], displacement, np.zeros(3))
            if error >= 2.1:
                continue
            q = e["q"] + shift
            j = np.stack([(self.poses.face(q + d)[0] - self.poses.face(q - d)[0]) / .002 for d in .001 * np.eye(5)], axis=1)
            jacobian = j.T @ np.linalg.solve(j @ j.T + .0009 * np.eye(3), np.eye(3))
            goal, impact, velocity, lead = e["action"]
            options.append((np.asarray(goal) + shift, impact + prediction[0] - self.eye.incoming[0] - e["contact"],
                            np.asarray(velocity), lead, jacobian))
        if not options:
            return restored
        rates = np.zeros_like(self.measured) if self.last is None else np.clip(30 * (self.measured - self.last), -6., 6.)
        state = self.planner.setup(p, v, self.flight.spin, self.measured, rates, self.body.target, self.angle)
        self.planned, tests, legal = self.planner.search(state, options, elapsed)
        self.scans += 1
        self.last_scan = tick
        self.updates.append(dict(tick=tick, shadow_trials=tests, predicted_legal=legal))
        if self.planned is None:
            return restored
        return self.act(tick)
