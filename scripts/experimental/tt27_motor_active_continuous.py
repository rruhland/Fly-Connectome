"""Diagnose all motor channels and receive a real feed without resetting the arm."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt19_view_full_game import digest_json
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt25_reference_receiver import ReferenceReceiver
from .tt27_motor_active import encoder, install_course, stable_choice
from .tt27_motor_active_serial import NativeTemplates, PULSE
from .tt27_motor_interface import AXIS_COUNT, IDENTITY, Candidates, Interface, course, describe_fit
from .tt27_world_camera import FEEDS, POSES


OUT = Path("artifacts/tt27/motor-active-continuous.json.gz")
SEEDS = (417601127, 665311037)
WARMUP = 8
PACKET_TICKS = 10
PREP_TICKS = WARMUP + AXIS_COUNT * PACKET_TICKS
SETTINGS = 4
ZERO = np.zeros(AXIS_COUNT)


class ContinuousReceiver(ReferenceReceiver):
    """Charge the physical arm continuously while a separate holder retains the ball."""

    def __init__(self):
        self.holding = False
        super().__init__()
        self.max_steps += PREP_TICKS

    def _aerodynamics(self):
        super()._aerodynamics()
        if self.holding:
            self.data.xfrc_applied[self._ball_body, :3] -= (
                self.model.body_mass[self._ball_body] * self.model.opt.gravity
            )

    def hold(self, seed):
        self.reset(seed=seed, server=-1, render=False)
        self.holding = True
        self._serving = False
        self.ball[:] = (0., 1.45, 2.75)
        self.velocity[:] = 0.
        self.data.qvel[self._ball_dof + 3 : self._ball_dof + 6] = 0.
        mujoco.mj_forward(self.model, self.data)

    def release(self, shot):
        self.holding = False
        self.ball[:] = [shot["pos_x"], shot["pos_y"], TABLE_HEIGHT + shot["pos_z"]]
        self.velocity[:] = [shot["vel_" + axis] for axis in "xyz"]
        self.data.qpos[self._ball_joint + 3 : self._ball_joint + 7] = [1., 0., 0., 0.]
        self.data.qvel[self._ball_dof + 3 : self._ball_dof + 6] = [shot["w_vel_" + axis] for axis in "xyz"]
        self.data.xfrc_applied[self._ball_body] = 0.
        mujoco.mj_forward(self.model, self.data)


def prepare(arena, dofs, actual, seed, sign=1):
    arena.hold(seed)
    previous = ZERO

    def advance(request):
        nonlocal previous
        applied = actual.apply(previous if actual.delay else request)
        _, _, done, cap, _ = arena.step(np.r_[applied, 0.], render=False)
        if done or cap or not arena.grasped:
            raise ValueError("A preparation step ended or released the paddle")
        previous = request

    warmup = [encoder(arena, dofs)]
    for _ in range(WARMUP):
        advance(ZERO)
        warmup.append(encoder(arena, dofs))
    traces = []
    for port in range(AXIS_COUNT):
        request = np.zeros(AXIS_COUNT)
        request[port] = sign * PULSE
        states = [encoder(arena, dofs)]
        for tick in range(PACKET_TICKS):
            issued = request if tick == 0 else -request if tick == 1 else ZERO
            advance(issued)
            states.append(encoder(arena, dofs))
        traces.append(np.asarray(states))
    if len(arena.events) or arena.steps != PREP_TICKS:
        raise ValueError("The ball or referee changed before the feed")
    return np.asarray(warmup), np.asarray(traces)


class SettledTemplates(NativeTemplates):
    """Learn all native templates on four continuous preparation sequences."""

    def __init__(self, arena, dofs):
        self.responses = np.empty((AXIS_COUNT, 2, 2, PACKET_TICKS, AXIS_COUNT * 2))
        for reverse in range(2):
            for lag in range(2):
                native = Interface(IDENTITY.permutation, IDENTITY.signs, lag)
                _, traces = prepare(arena, dofs, native, 287601133, -1 if reverse else 1)
                self.responses[:, reverse, lag] = traces[:, 1:] - traces[:, :1]
        self.receipt = dict(preparations=4, ticks=4 * PREP_TICKS, normalized_peak_request=PULSE,
                            responses=self.responses.tolist())


def arm_digest(arena):
    """Use evaluator state only to audit physical start equality across branches."""
    ball = arena._ball_body
    indices = np.where(arena.model.dof_bodyid != ball)[0]
    retained = np.r_[arena.data.qpos[:arena._ball_joint], arena.data.qpos[arena._ball_joint + 7 :],
                     arena.data.qvel[indices], arena.data.ctrl, arena.data.time]
    return sha256(retained.tobytes()).hexdigest()


def receive(arena, shot, actual, inferred, pose_memory, tape=None):
    before = arm_digest(arena)
    arena.release(shot)
    if arm_digest(arena) != before or arena.steps != PREP_TICKS:
        raise ValueError("Release changed the physical arm or reset its clock")
    body, tracker = PublicBody(arena), ViewHistory()
    target_error = np.max(np.abs(arena.data.ctrl[arena._arm_ids[1][2]] - body.target))
    if target_error > 1e-8:
        raise ValueError("Balanced preparation changed a servo target")
    keys, goals, scale = pose_memory
    action_log, executed = [], []
    previous, goal_index = ZERO, None
    for tick in range(arena.max_steps - PREP_TICKS):
        if tape is None:
            if tick <= OBSERVE:
                pixels = tracker.observe(arena.render_cameras(), arena.joints, body.target)[:14]
                if tick == OBSERVE:
                    goal_index = int(np.argmin(np.linalg.norm((keys - pixels) / scale, axis=1)))
            wanted = ZERO
            if goal_index is not None:
                pending = inferred.apply(previous) * inferred.delay
                target = np.clip(body.target[:AXIS_COUNT] + STEP_SIZE[:AXIS_COUNT] * pending,
                                 body.lower[:AXIS_COUNT], body.upper[:AXIS_COUNT])
                wanted = np.clip((goals[goal_index] - target) / STEP_SIZE[:AXIS_COUNT], -1., 1.)
            request = inferred.invert(wanted)
        else:
            request = np.asarray(tape[tick])
        applied = actual.apply(previous if actual.delay else request)
        _, _, done, cap, _ = arena.step(np.r_[applied, 0.], render=False)
        body.issued(np.r_[inferred.apply(previous if inferred.delay else request), 0.])
        previous = request
        action_log.append(request.tolist())
        executed.append(applied.tolist())
        if done or cap or arena.legal_returns:
            break
    assert arena.steps == PREP_TICKS + len(action_log)
    return dict(preparation_state_sha256=before, goal_index=goal_index, servo_recenter_error=float(target_error),
                legal=arena.legal_returns, contact=arena.contacts, incoming=arena.rival_returns,
                events=digest_json(arena.events), requests=action_log, executed=executed)


def summarize(receipt):
    settings = [row for stream in receipt["streams"] for row in stream["settings"]]
    rows = [row for stage in settings for row in stage["trials"]]
    return {name: dict(legal=sum(x["results"][name]["legal"] for x in rows),
                       contacts=sum(x["results"][name]["contact"] for x in rows),
                       exact_maps=sum(x["inferred"][name]["exact"] for x in settings))
            for name in rows[0]["results"]}


def run(path, per_setting):
    old = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    donors = {r["shot"]["id"]: r["shot"] for r in old}
    with np.load(POSES, allow_pickle=False) as data:
        memory = tuple(data[k] for k in ("keys", "goals", "scale"))
    arena = ContinuousReceiver()
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    candidates = Candidates(None, None, None, arena.control_dt)
    receipt = dict(schema="tt27-motor-active-continuous-1", pose_source_sha256=_hash(POSES),
                   feed_source_sha256=_hash(FEEDS), seeds=SEEDS, per_setting=per_setting, streams=[],
                   budget=dict(total_ticks=PREP_TICKS, warmup_ticks=WARMUP, ticks_per_port=PACKET_TICKS,
                               nonzero_ticks=2 * AXIS_COUNT, inter_pulse_zero_ticks=PACKET_TICKS - 2,
                               simulated_seconds=PREP_TICKS * arena.control_dt,
                               intermediate_arm_resets=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        templates = SettledTemplates(arena, dofs)
        receipt["source_templates"] = templates.receipt
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            carried = IDENTITY
            stream = dict(seed=seed, settings=[])
            receipt["streams"].append(stream)
            for index, (actual, shift, speed) in enumerate(install_course(rng)[:SETTINGS]):
                starts = course(rng, donors, per_setting, shift, speed)
                prep_seed = int(rng.integers(2**31))
                warmup, template_trace = prepare(arena, dofs, actual, prep_seed)
                error = templates.loss(template_trace, candidates)
                active, info = stable_choice(error, candidates, carried, carried.permutation)
                reset, reset_info = stable_choice(error, candidates, IDENTITY)
                modes = dict(active=active, active_reset=reset, stale=carried, expert=actual,
                             factory=IDENTITY)
                scores = []
                for shot in starts:
                    observations, implementations = {}, {}
                    for name, choice in modes.items():
                        if choice not in implementations:
                            prepare(arena, dofs, actual, prep_seed)
                            row = receive(arena, shot, actual, choice, memory)
                            prepare(arena, dofs, actual, prep_seed)
                            repeated = receive(arena, shot, actual, choice, memory, tape=row["requests"])
                            if any(row[k] != repeated[k] for k in row if k != "goal_index"):
                                raise ValueError("Continuous feed replay changed the physical test")
                            implementations[choice] = row
                        observations[name] = implementations[choice]
                    if len({r["preparation_state_sha256"] for r in observations.values()}) != 1:
                        raise ValueError("Paired feeds did not share the same physical arm preparation")
                    scores.append(dict(shot=shot, results=observations, verified=len(implementations)))
                stream["settings"].append(dict(
                    index=index, actual=asdict(actual), shot_shift_m=shift, speed_ratio=speed, prep_seed=prep_seed,
                    start_warmup=warmup.tolist(), encoders=template_trace.tolist(),
                    inferred={name: describe_fit(value, actual) for name, value in modes.items()},
                    active=info, reset=reset_info, trials=scores))
                carried = active
                receipt["summary"] = summarize(receipt)
                _write(path, receipt)
                print(seed, index, {name: value == actual for name, value in modes.items()},
                      [info["confirm_tick"], reset_info["confirm_tick"]],
                      {name: sum(r["results"][name]["legal"] for r in scores) for name in modes}, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--per-setting", type=int, default=5)
    args = parser.parse_args()
    if args.per_setting <= 0:
        parser.error("--per-setting must be positive")
    run(args.output, args.per_setting)
