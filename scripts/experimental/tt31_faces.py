"""Search both physical blade faces in a stereo-guided SO-101 rally."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_action_memory import Example, Experience, RallyObjective, RecallPlanner
from .tt29_arm_clearance import ArmClearance
from .tt29_conditions import cooperative_targets
from .tt29_endurance import audit
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_partner_fast import FastPartnerArena, NativeBallFlight
from .tt29_rally_physics import BladeIK, LATERAL_STRIKE_READY, RallyArena, measured_launches, mujoco
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt29_vision_replay import clearance_final
from .tt29_vision_twin import OpticalPlanner


READY = (1.62, -1.29, .22, 1.08, -.745, -.10)
ROLL = 4
SETTINGS = dict(strike_ready="lateral", width=384, height=288, clearance_guard=True,
                fast_search=True, max_steps=9000, rate_scale=.8, velocity_guard=True,
                motor_envelope=True)


class NeutralReady:
    def __init__(self, *, neutral=True, **settings):
        self.neutral = neutral
        super().__init__(**settings)

    def _arm_home(self, side):
        if side == 1 and self.neutral:
            return READY, ()
        return super()._arm_home(side)


class ObserveFaces:
    """Record native sphere-to-paddle contacts for the referee audit only."""

    def reset(self, *args, **kwargs):
        self.contacts_on_face = []
        self.face_geometry = []
        return super().reset(*args, **kwargs)

    def _racket_contact(self, side, location, incoming):
        if side == 1:
            blade = self._racket_ids[1]
            ball = self._geom_ids["ball"]
            pairs = (
                c for c in self.data.contact if c.efc_address >= 0 and set(c.geom) == {blade, ball}
            )
            contact = min(pairs, key=lambda c: np.linalg.norm(c.pos - location))
            axes = self.data.geom_xmat[blade].reshape(3, 3)
            normal = axes[:, 2]
            towards_ball = contact.frame[:3] * (1 if contact.geom[0] == blade else -1)
            projection = float(normal @ towards_ball)
            local = axes.T @ (contact.pos - self.data.geom_xpos[blade])
            shoulder = self.data.body("learner_shoulder").xpos
            self.face_geometry.append(dict(
                seconds=self.data.time, local_xyz=local.tolist(),
                contact_radius_m=float(np.linalg.norm(local[:2])),
                blade_radius_m=float(self.model.geom_size[blade, 0]),
                lateral_m=float(contact.pos[0] - shoulder[0]),
            ))
            self.contacts_on_face.append(dict(
                seconds=self.data.time, event=len(self.events), normal=normal.tolist(),
                world_y=float(normal[1]), contact_alignment=projection,
                surface=1 if projection >= .8 else -1 if projection <= -.8 else 0,
                paddle_roll=float(self.joints[ROLL]),
            ))
        return super()._racket_contact(side, location, incoming)


class FaceTwin(NeutralReady, RallyArena):
    pass


class FaceArena(ObserveFaces, NeutralReady, FastPartnerArena):
    pass


class FaceMemory(Experience):
    """Propose a half-turn of recalled wrist trajectories for verification."""

    def __init__(self, limits, control_hz):
        super().__init__()
        self.low, self.high = limits
        self.control_hz = control_hz

    def nearest(self, key, limit):
        nearest = super().nearest(key, limit)
        proposed = list(nearest)
        for example in nearest:
            # Only rotate the part before the original recovery. A roll sweep
            # back to neutral is computed afresh after the next predicted impact.
            end = min(len(example.targets),
                      max(1, round((example.time_to_contact + .1) * self.control_hz)))
            pre = example.targets[:end].copy()
            roll = pre[:, ROLL]
            for delta in (-np.pi, np.pi):
                rotated = roll + delta
                if np.mean((rotated >= self.low) & (rotated <= self.high)) >= .75:
                    goals = example.targets.copy()
                    goals[:end, ROLL] = np.clip(rotated, self.low, self.high)
                    goals[end:, ROLL] = goals[end - 1, ROLL]
                    proposed.append(Example(example.flight, example.time_to_contact, goals))
        return proposed


class SignedBladeIK(BladeIK):
    """Reuse the existing kinematics while retaining a chosen geometric face."""

    def __init__(self, arena, sign):
        super().__init__(arena)
        self.sign = sign
        self.roll_joint = arena.model.joint("learner_wrist_roll").id

    def pose(self, joints, *, jacobian=False):
        position, face, *jac = super().pose(joints, jacobian=jacobian)
        raw = self.data.xmat[self.gripper].reshape(3, 3) @ self.normal
        multiplier = self.sign if raw[1] >= 0 else -self.sign
        if not jacobian:
            return position, multiplier * face
        return position, multiplier * face, jac[0], multiplier * jac[1]

    def fit(self, center, normal, start):
        _, current = self.pose(start)
        axis = self.data.xaxis[self.roll_joint]
        lateral = current - (current @ axis) * axis
        target = normal - (normal @ axis) * axis
        delta = np.arctan2(axis @ np.cross(lateral, target), lateral @ target)
        seed = start.copy()
        roll = seed[ROLL] + delta
        options = np.array([roll, roll + 2 * np.pi, roll - 2 * np.pi])
        nominal = np.clip(options, self.low[ROLL], self.high[ROLL])
        seed[ROLL] = nominal[np.argmin(np.abs(nominal - options))]
        return super().fit(center, normal, seed)


class ReturnToReady:
    """Check the wrist recovery as part of the simulated stroke."""

    def __init__(self, objective, dt, *, recovery_delay=.045):
        self.objective = objective
        self.dt = dt
        self.recovery_delay = recovery_delay

    def trajectories(self, goals, target, cutoff):
        rates, scores = self.objective.trajectories(goals, target, cutoff)
        hits = np.flatnonzero(np.isfinite(scores[:, 0]))
        if not len(hits):
            return rates, scores
        ranked = hits[np.argsort(scores[hits, 0], kind="stable")][:20]
        edits = goals[ranked].copy()
        for index, trial in zip(ranked, edits, strict=True):
            impact = int(np.ceil((scores[index, 1] + self.recovery_delay) / self.dt))
            trial[impact:, ROLL] = READY[ROLL]
        checked_rates, checked_scores = self.objective.trajectories(edits, target, cutoff)
        legal = np.isfinite(checked_scores[:, 0])
        rates[ranked[legal]] = checked_rates[legal]
        scores[ranked[legal]] = checked_scores[legal]
        return rates, scores


class FacePlanner(RecallPlanner):
    def __init__(self, arena, memory, *, seed):
        super().__init__(arena, memory, seed=seed, blend=False, recent=True)
        self.branches = tuple(SignedBladeIK(arena, sign) for sign in (1, -1))

    def candidates(self, windows, target, timeline):
        old = self.ik
        proposals, details = [], []
        try:
            for self.ik in self.branches:
                goals, params = super().candidates(windows, target, timeline)
                if len(goals):
                    proposals.extend(goals)
                    details.extend(params)
        finally:
            self.ik = old
        return np.asarray(proposals), details


def signed_returns(arena):
    landings = {r["hit"] for r in audit(arena.events)["shots"] if r["side"] == 1}
    contacts = [
        dict(contact, legal=contact["seconds"] in landings)
        for contact in arena.contacts_on_face
    ]
    if sum(contact["legal"] for contact in contacts) != len(landings):
        raise ValueError("A legal return lacked its native sphere-to-paddle contact")
    return contacts


def play(arena, twin, renderer, rig, experience, shot, opponent_seed, cap, mode):
    arena.launch(shot)
    arena.partner.reset(seed=opponent_seed)
    twin.reset(seed=shot["id"], server=-1, render=False)
    rig.begin()
    eye = OpticalFlight(rig, FlightPractice(), learn=True, detector=native_ball_pixels)
    policy = OpticalPlanner(twin, seed=290110)
    if mode == "original":
        planner = RecallPlanner(twin, experience, seed=290110, blend=True, recent=True)
        planner.predict = RallyObjective(planner.predict)
    else:
        planner = FacePlanner(twin, experience, seed=290110)
        planner.predict = ReturnToReady(RallyObjective(planner.predict), twin.control_dt)
    policy.nominal = planner
    planner.flight = NativeBallFlight(twin)
    x = previous = arena.joints.copy()
    lo, hi = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
    actions = []
    far = np.zeros(5)
    start = process_time()
    initial_examples = len(experience.examples)
    for tick in range(arena.max_steps):
        t = tick * arena.control_dt
        q = arena.joints.copy()
        if tick % 2 == 0:
            eye.observe(t, renderer.capture(), q)
        action = policy.command(tick, eye.latest, q, (q - previous) / arena.control_dt, x)
        previous = q
        x = np.clip(x + action * planner.rates * arena.control_dt, lo, hi)
        _, _, done, capped, info = arena.exchange(action, far)
        actions.append(action.tolist())
        if any(event["kind"] == "legal_return" and event["side"] == 1 for event in info["events"]):
            planner.remember()
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= cap:
            break
    return dict(
        shot=shot, opponent_seed=opponent_seed, score=audit(arena.events),
        signed_contacts=signed_returns(arena), events=arena.events.copy(), actions=actions,
        initial_examples=initial_examples, final_examples=len(experience.examples),
        retrieved=planner.retrieved, planned=planner.fallback, process_cpu_s=process_time() - start,
    )


def run(feeds, output, ids, *, mode, checkpoint, spread, offset, cap):
    if (mode == "cold") == (checkpoint is not None):
        raise ValueError("Use a movement checkpoint for warm and original modes only")
    if cap < 1:
        raise ValueError("The return cap must be positive")
    shots = measured_launches(feeds)
    examples = _read(checkpoint)["checkpoint"] if checkpoint else []
    target = cooperative_targets("lateral", spread)
    is_neutral = mode != "original"
    twin = FaceTwin(neutral=is_neutral, robust_contact=True, **SETTINGS)
    arena = FaceArena(neutral=is_neutral, targets=target, guard_table=True, net_rescue=True, **SETTINGS)
    renderer = CachedStereoRenderer(arena)
    rig = Rig.from_arena(arena)
    rows = []
    try:
        for point, sid in enumerate(ids):
            axes = twin._arm_ids[1][2]
            axis = axes[ROLL]
            limits = twin._joint_low[axis], twin._joint_high[axis]
            memory = FaceMemory(limits, round(1 / twin.control_dt)) if is_neutral else Experience()
            for record in examples:
                memory.add(np.array(record["flight"]), np.array(record["targets"]), record["time_to_contact"])
            row = play(arena, twin, renderer, rig, memory, shots[sid], sid + offset + point, cap, mode)
            rows.append(row)
            _write(output, dict(
                schema="tt31-two-sided-v1", mode=mode, cap=cap, spread=spread, seed_offset=offset,
                ready=list(READY if is_neutral else LATERAL_STRIKE_READY),
                feed_sha256=_hash(feeds), movement_sha256=_hash(checkpoint) if checkpoint else None, rows=rows,
            ))
            print(mode, sid, row["score"]["returns"],
                  [c["surface"] for c in row["signed_contacts"] if c["legal"]], flush=True)
    finally:
        renderer.close()
        arena.close()
        twin.close()


def replay(source, output, shot_id):
    tape = _read(source)
    row = next(item for item in tape["rows"] if item["shot"]["id"] == shot_id)
    is_neutral = tape["mode"] != "original"
    targets = cooperative_targets("lateral", tape["spread"])
    options = dict(neutral=is_neutral, targets=targets, guard_table=True, net_rescue=True, **SETTINGS)
    arena = FaceArena(**options)
    cad = FaceArena(detail="cad", **options)
    guard = ArmClearance(cad.model, tolerance=0.)
    shadow = mujoco.MjData(cad.model)
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        arena.sample_clearance(True)
        arena._cad_reader = guard, shadow
        for action in row["actions"]:
            _, _, done, capped, _ = arena.exchange(action, np.zeros(5))
            if done or capped:
                break
        if arena.steps != len(row["actions"]) or digest_json(arena.events) != digest_json(row["events"]):
            raise ValueError("Contact and referee events diverged during replay")
        if signed_returns(arena) != row["signed_contacts"]:
            raise ValueError("Physical face measurements diverged during replay")
        clearance = clearance_final(arena, guard, shadow)
        if clearance["samples"] != round(arena.data.time / arena.model.opt.timestep) + 1:
            raise ValueError("Clearance audit omitted a physics state")
        receipt = dict(
            schema="tt31-two-sided-replay-v1",
            shot_id=shot_id, tape_sha256=_hash(source), events_sha256=digest_json(arena.events),
            exact_events=True, exact_signed_contacts=True, checked_pairs=len(guard.first),
            clearance=clearance, motor=arena.motor_audit(), score=row["score"],
            signed_contacts=row["signed_contacts"], face_geometry=arena.face_geometry,
        )
        _write(output, receipt)
        print(shot_id, clearance, flush=True)
    finally:
        arena.close()
        cad.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    live = sub.add_parser("run")
    live.add_argument("feeds", type=Path)
    live.add_argument("output", type=Path)
    live.add_argument("--ids", type=int, nargs="+", required=True)
    live.add_argument("--mode", choices=("original", "warm", "cold"), required=True)
    live.add_argument("--checkpoint", type=Path)
    live.add_argument("--spread", choices=("practice", "varied"), default="varied")
    live.add_argument("--offset", type=int, default=91537)
    live.add_argument("--cap", type=int, default=24)
    receipt = sub.add_parser("replay")
    receipt.add_argument("source", type=Path)
    receipt.add_argument("output", type=Path)
    receipt.add_argument("--id", type=int, required=True)
    args = p.parse_args()
    if args.command == "run":
        run(args.feeds, args.output, args.ids, mode=args.mode, checkpoint=args.checkpoint,
            spread=args.spread, offset=args.offset, cap=args.cap)
    else:
        replay(args.source, args.output, args.id)


if __name__ == "__main__":
    main()
