"""Evaluate native stroke planning from camera hypotheses in an isolated nominal plant."""

import argparse
from collections import Counter, deque
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _write
from .tt19_so101 import AXES, GRIP_BODY
from .tt29_arm_clearance import ArmClearance
from .tt29_partner import PartnerArena
from .tt29_rally_physics import MotionPlanner, RallyArena, event_metrics, measured_launches
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt29_vision_assay import CONDITIONS, OUT
from .tt27_world_camera import CameraMounts


COURSE = (4197, 3728, 4230, 7637, 15489, 8230, 14562, 10318)
PARTNER_TARGETS = [(-.042, -1.070, 1.170, 3.85), (.042, -1.082, 1.190, 4.05),
                   (.015, -1.061, 1.160, 3.65), (-.020, -1.082, 1.200, 4.10),
                   (0., -1.068, 1.180, 3.95)]


class CandidateUncertainty:
    """Prefer strokes that also land when the optical estimate is slightly off."""

    OFFSETS = ((.004, -.003, .018, 24.), (-.003, .003, -.014, 46.))

    def __init__(self, planner):
        self.planner = planner
        self.nominal = planner.predict

    def trajectories(self, goals, target, cutoff):
        rates, scored = self.nominal.trajectories(goals, target, cutoff)
        candidates = np.argsort(scored[:, 0])[:14]
        candidates = candidates[np.isfinite(scored[candidates, 0])]
        if candidates.size < 2:
            return rates, scored
        a = self.planner.arena
        position, velocity, spin = a.ball.copy(), a.velocity.copy(), a.spin.copy()
        accumulated = scored[candidates, 0].copy()
        try:
            for dy, dz, dvy, wx in self.OFFSETS:
                a.ball[:] = position + (0., dy, dz)
                a.velocity[:] = velocity + (0., dvy, 0.)
                a.data.qvel[a._ball_dof + 3:a._ball_dof + 6] = (wx, 0., 0.)
                _, variation = self.nominal.trajectories(goals[candidates], target, cutoff)
                quality = variation[:, 0]
                accumulated += .48 * np.where(np.isfinite(quality), quality, 20.)
        finally:
            a.ball[:] = position
            a.velocity[:] = velocity
            a.data.qvel[a._ball_dof + 3:a._ball_dof + 6] = spin
        result = scored.copy()
        result[:, 0] = np.inf
        result[candidates, 0] = accumulated
        return rates, result


class OpticalPlanner:
    """Populate a separate calibrated dynamics model with RGB and local actuator observations."""

    def __init__(self, twin, *, seed, robust=False):
        self.twin = twin
        self.nominal = MotionPlanner(twin, 1, seed=seed)
        if robust:
            self.nominal.predict = CandidateUncertainty(self.nominal)
        self.positions, _, self.actuators, _, _ = twin._arm_ids[1]
        self.dofs = np.array([twin.model.joint(f'learner_{axis}').dofadr[0] for axis in AXES])
        self.grip = twin.model.body(GRIP_BODY).id
        paddle = twin.model.joint('learner_paddle_free')
        self.paddle_q, self.paddle_v = int(paddle.qposadr[0]), int(paddle.dofadr[0])
        weld = twin.model.eq_data[twin.model.equality('tt19_grasp').id]
        self.relative_pos = weld[3:6].copy()
        self.relative_quat = weld[6:10].copy()
        self.home = self.nominal.home
        self.tape = None
        self.first = 0
        self.previous_try = -40
        self.incoming = False
        self.bounced = False

    def _refresh(self, step, flight, angles, rate, target):
        a = self.twin
        d = a.data
        p, v = flight.advance(step * a.control_dt)
        a.ball[:] = p
        a.velocity[:] = v
        d.qpos[a._ball_joint + 3:a._ball_joint + 7] = [1., 0., 0., 0.]
        d.qvel[a._ball_dof + 3:a._ball_dof + 6] = 0.   # Spin is unobservable in the unmarked ball.
        d.time = step * a.control_dt
        d.qpos[self.positions] = angles
        d.qvel[self.dofs] = rate
        d.ctrl[self.actuators] = target
        mujoco.mj_kinematics(a.model, d)
        grip = d.body(self.grip)
        d.qpos[self.paddle_q:self.paddle_q + 3] = grip.xpos + grip.xmat.reshape(3, 3) @ self.relative_pos
        paddle_quat = d.qpos[self.paddle_q + 3:self.paddle_q + 7]
        mujoco.mju_mulQuat(paddle_quat, grip.xquat, self.relative_quat)
        paddle_rotation = np.empty(9)
        mujoco.mju_quat2Mat(paddle_rotation, paddle_quat)
        mujoco.mj_comPos(a.model, d)
        linear = np.zeros((3, a.model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jac(a.model, d, linear, angular, d.qpos[self.paddle_q:self.paddle_q + 3], self.grip)
        d.qvel[self.paddle_v:self.paddle_v + 3] = linear @ d.qvel
        d.qvel[self.paddle_v + 3:self.paddle_v + 6] = (angular @ d.qvel) @ paddle_rotation.reshape(3, 3)
        a.steps = step
        self.bounced |= flight.bounced and p[1] < .1
        a._received_bounce = self.bounced
        d.qacc_warmstart[:] = 0.
        mujoco.mj_forward(a.model, d)

    def command(self, step, flight, angles, rate, target):
        now = step * self.twin.control_dt
        reads = flight is not None and now - flight.time < .105
        if reads and flight.velocity[1] > 1.05:
            if self.incoming:
                self.tape, self.bounced = None, False
            self.incoming = False
        elif reads and flight.velocity[1] < -.90:
            if self.tape is not None:
                elapsed = (step - self.first) * self.twin.control_dt
                # The old command may outlast a briefly occluded reversal. A
                # reliable return already approaching on the far half is a fresh leg.
                if (elapsed > max(.16, len(self.tape) * self.twin.control_dt - .56) and
                        flight.position[1] > .16 and flight.rms < 1.4 and flight.depth_sigma < .12):
                    self.tape, self.bounced = None, False
            self.incoming = True
        age = step - self.first
        if self.tape is not None and age < len(self.tape):
            return self.tape[age]
        if (reads and self.incoming and step - self.previous_try >= 5 and flight.position[1] > -1.28 and
                flight.rms < 2.4 and flight.depth_sigma < .16):
            self._refresh(step, flight, angles, rate, target)
            self.previous_try = step
            self.tape = self.nominal.plan()
            self.first = step
            if self.tape is not None:
                return self.tape[0]
        goal = self.home if not self.incoming or self.tape is None else target
        return np.clip((goal - target) / (self.twin.control_dt * self.nominal.rates), -.80, .80)


def play(arena, twin, rig, course, shot, mode, *, seed=803145, target_length=12, learn=False, robust=False, audit_model=None):
    arena.launch(shot)
    arena.sample_clearance(audit_model is not None)
    if audit_model is not None:
        arena._cad_reader = (ArmClearance(audit_model), mujoco.MjData(audit_model))
    twin.reset(seed=seed, server=-1, render=False)
    rig.begin()
    tracker = OpticalFlight(rig, course, learn=learn)
    policy = OpticalPlanner(twin, seed=seed, robust=robust)
    target = policy.home.copy()
    encoder = arena.joints.copy()
    lower, upper = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
    captures = deque()
    actions = []
    reads = Counter()
    plan_cpu = camera_cpu = 0.
    for step in range(arena.max_steps):
        now = step * arena.control_dt
        indices = mode.eyes
        if mode.mask:
            if 10 <= step % 40 < 17:
                indices = ('left',)
            if 20 <= step % 40 < 24:
                indices = ()
        start = perf_counter()
        arena.streams = indices
        frames = arena.render_cameras() if indices else {}
        q = arena.joints.copy()
        captures.append((now, frames, q))
        if len(captures) > mode.lag:
            tracker.observe(*captures.popleft())
            reads.update(tracker.last_frame_views)
        camera_cpu += perf_counter() - start
        start = perf_counter()
        action = policy.command(step, tracker.latest, q, (q - encoder) / arena.control_dt, target)
        plan_cpu += perf_counter() - start
        encoder = q
        target = np.clip(target + arena.control_dt * policy.nominal.rates * action, lower, upper)
        _, _, ended, truncated, _ = arena.exchange(action, np.zeros(5))
        actions.append(action.tolist())
        if ended or truncated or arena.legal_returns + arena.rival_returns >= target_length + 1:
            break
    audit = arena.clearance_result() if audit_model is not None else None
    arena.sample_clearance(False)
    return dict(id=shot['id'], result=arena.point_result(), events=arena.events.copy(),
                decisions=policy.nominal.decisions, actions=actions, reads=dict(reads),
                camera_wall_seconds=camera_cpu, planning_wall_seconds=plan_cpu,
                registrations=rig.registrations.copy(), landed=event_metrics(arena.events),
                motor_audit=arena.motor_audit(), audit=audit)


def run(feeds, output, mode_names, ids, *, ready=True, width=480, height=360, partner=False, robust=False, guard=False, audit=False):
    shots = measured_launches(feeds)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema='tt29-vision-twin-1', source_sha256=_hash(feeds), dimensions=[width, height],
                  control_hz=60, start_pose=ready, partner=partner, robust=robust, guard_clearance=guard,
                  full_cad_audit=audit, partner_targets=PARTNER_TARGETS if partner else None, assessment={})
    course = FlightPractice()
    twin = RallyArena(strike_ready=ready, width=width, height=height, max_steps=450, clearance_guard=guard)
    klass = PartnerArena if partner else RallyArena
    options = {'targets': PARTNER_TARGETS} if partner else {}
    cad = klass(strike_ready=ready, detail='cad', **options) if audit else None
    try:
        for name in mode_names:
            setting = CONDITIONS[name]
            arena = klass(strike_ready=ready, width=width, height=height, max_steps=450, view_mode='wrist+fixed', **options)
            try:
                arena.streams = ('left', 'right')
                reference = arena.render_cameras() if setting.register else None
                rig = Rig.from_arena(arena, reference=reference)
                if setting.turn:
                    CameraMounts(arena).move(((2.8, -2.2, 2.3), (-2.7, 2.1, -2.9)))
                for panel in arena.table_panels:
                    arena.model.geom(panel).solref[1] *= setting.brake
                arena.model.opt.gravity[2] *= setting.height
                records = []
                result['assessment'][name] = records
                for sid in ids:
                    row = play(arena, twin, rig, course, shots[sid], setting, target_length=12 if partner else 1,
                               learn=name == 'combined', robust=robust, audit_model=None if cad is None else cad.model)
                    records.append(row)
                    m = row['result']
                    print(name, sid, 'contacts', m['contacts'], m['rival_contacts'], 'legal', m['legal_returns'],
                          m['rival_returns'], m['reason'], 'plans', len(row['decisions']), 'reads', row['reads'],
                          flush=True)
                    _write(output, result)
            finally:
                arena.close()
    finally:
        twin.close()
        if cad is not None:
            cad.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('--output', type=Path, default=OUT / 'tt29_vision_twin.json.gz')
    p.add_argument('--modes', nargs='+', choices=tuple(k for k in CONDITIONS if k != 'state_diagnostic'), default=('combined', 'stereo', 'wrist', 'outside'))
    p.add_argument('--ids', nargs='+', type=int, default=COURSE)
    p.add_argument('--ready', choices=('strike', 'camera', 'deep'), default='strike')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--partner', action='store_true')
    p.add_argument('--robust', action='store_true')
    p.add_argument('--guard-arm', action='store_true')
    p.add_argument('--audit-cad', action='store_true')
    a = p.parse_args()
    ready = dict(strike=True, camera=False, deep='deep')[a.ready]
    run(a.feeds, a.output, a.modes, a.ids, ready=ready, width=a.width, height=a.height, partner=a.partner, robust=a.robust,
        guard=a.guard_arm, audit=a.audit_cad)
