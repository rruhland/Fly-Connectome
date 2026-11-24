"""Retrieve and time prior joint commands during a rally, with an optional dynamics check."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .tt29_rally_physics import MotionPlanner


@dataclass
class Example:
    flight: np.ndarray
    time_to_contact: float
    targets: np.ndarray


class Experience:
    """Save actions that actually landed, including the joint configuration before the stroke."""

    def __init__(self):
        self.examples = []

    def add(self, key, targets, delay):
        self.examples.append(Example(key.copy(), float(delay), targets.copy()))

    def nearest(self, key, limit):
        scale = np.array([0.12, 0.18, 1.2, 1.4, 1.2])
        return sorted(
            self.examples, key=lambda e: np.linalg.norm((e.flight - key) / scale)
        )[:limit]


def context(path, depth):
    if not len(path) or path[0, 2] < depth - 0.05 or path[-1, 2] > depth + 0.05:
        return None
    row = path[np.argmin(abs(path[:, 2] - depth))]
    return row[[1, 3, 4, 5, 6]], row[0]


class RecallPlanner(MotionPlanner):
    """The motor memory is updated only when the point supplies a legal-return reward."""

    def __init__(self, arena, memory, *, seed, verify=True):
        super().__init__(arena, side=1, seed=seed)
        self.memory = memory
        self.verify = verify
        self.pending = None
        self.retrieved = 0
        self.fallback = 0
        self.retrieval_seconds = 0.0
        self.fallback_seconds = 0.0

    def remember(self):
        if self.pending is not None:
            self.memory.add(*self.pending)
            self.pending = None

    def plan(self):
        a = self.arena
        start = perf_counter()
        path = self.flight.path(a.ball, a.velocity, a.spin, 1, a._received_bounce)
        packet = context(path, -1.29)
        if packet is None:
            return super().plan()
        key, delay = packet
        dt = a.control_dt
        inc = self.rates * dt
        q = a.data.ctrl[self.axes].copy()
        lo, hi = a._joint_low[self.axes], a._joint_high[self.axes]
        proposed = []
        for e in self.memory.nearest(key, 6):
            interval = np.arange(max(1, round((delay + 0.65) / dt))) + 1.0
            ix = interval + (e.time_to_contact - delay) / dt
            goals = np.stack(
                [
                    np.interp(ix, np.arange(1, len(e.targets) + 1), col, left=q[j])
                    for j, col in enumerate(e.targets.T)
                ],
                axis=1,
            )
            proposed.append(goals)
        if proposed:
            if self.verify:
                rates, score = self.predict.trajectories(
                    np.stack(proposed), np.array([0.0, 1.32, 1.08, 3.6]), delay + 0.16
                )
                winners = np.flatnonzero(
                    np.isfinite(score[:, 0]) & (score[:, 0] < 1.75)
                )
            else:
                winners = [0]
            if len(winners):
                i = int(
                    winners[0]
                    if not self.verify
                    else winners[np.argmin(score[winners, 0])]
                )
                ref = np.clip(proposed[i], lo, hi)
                tape = (
                    rates[i]
                    if self.verify
                    else np.clip(np.diff(np.vstack([q, ref]), axis=0) / inc, -1.0, 1.0)
                )
                self.pending = (key, ref, delay)
                self.retrieved += 1
                self.retrieval_seconds += perf_counter() - start
                return tape
        self.retrieval_seconds += perf_counter() - start
        start = perf_counter()
        tape = super().plan()
        self.fallback_seconds += perf_counter() - start
        if tape is not None:
            current = q.copy()
            targets = []
            for row in tape:
                current = np.clip(current + inc * row, lo, hi)
                targets.append(current.copy())
            self.pending = (key, np.stack(targets), delay)
            self.fallback += 1
        return tape


def refine_after_bounce(policy, step, flight, q, qdot, commanded):
    """Check the remaining action on a fresh optical estimate after the actual table rebound."""
    twin = policy.twin
    now = step * twin.control_dt
    age = step - policy.first
    if (
        policy.tape is None
        or not policy.incoming
        or age <= 0
        or age >= len(policy.tape)
        or flight is None
        or not flight.bounced
        or now - flight.time > 0.04
        or flight.rms > 2
        or flight.depth_sigma > 0.10
        or getattr(policy, "last_refined", -1000) + 12 > step
    ):
        return
    p, v = flight.advance(now)
    eta = (-1.30 - p[1]) / min(v[1], -0.01)
    if not 0.075 < eta < 0.30 or v[1] > -1.0 or v[2] < -0.3:
        return
    policy.last_refined = step
    policy._refresh(step, flight, q, qdot, commanded)
    twin._received_bounce = True
    low, high = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
    horizon = int(np.ceil((eta + 0.74) / twin.control_dt))
    desired = []
    goal = commanded.copy()
    tail = policy.tape[age:]
    for i in range(horizon):
        if i < len(tail):
            goal = np.clip(
                goal + tail[i] * policy.nominal.rates * twin.control_dt, low, high
            )
        desired.append(goal.copy())
    _, assessment = policy.nominal.predict.trajectories(
        np.array([desired]), np.array([0.0, 1.32, 1.08, 3.6]), eta + 0.12
    )
    if np.isfinite(assessment[0, 0]) and assessment[0, 0] < 1.9:
        return
    new = policy.nominal.plan()
    if new is not None:
        policy.tape = new
        policy.first = step
        policy.corrections = getattr(policy, "corrections", 0) + 1


def run(
    path,
    output,
    ids,
    view,
    pose,
    *,
    checkpoint=None,
    refine=False,
    reset_memory=False,
    no_memory=False,
    robust_contact=False,
    seed_offset=0,
    condition="stock",
    bounce=None,
):
    from .tt4_transfer_trial import _write
    from .tt29_endurance import audit
    from .tt29_partner import FOREHAND_LANES, DEEP_LANES, PartnerArena
    from .tt29_rally_physics import RallyArena, measured_launches
    from .tt29_vision import FlightPractice, OpticalFlight, Rig
    from .tt29_vision_twin import OpticalPlanner

    shots = measured_launches(path)
    settings = dict(
        strike_ready=pose,
        width=384,
        height=288,
        clearance_guard=True,
        fast_search=True,
        max_steps=9000,
    )
    twin = RallyArena(**settings, robust_contact=robust_contact)
    arena = PartnerArena(
        **settings,
        targets=FOREHAND_LANES if pose == "lateral" else DEEP_LANES,
        guard_table=True,
    )
    from .tt29_conditions import CONDITIONS, Delay, apply
    from .tt29_partner import PracticePartner

    conditions = CONDITIONS[condition]
    # The camera rig knows the nominal installation. Perturbations happen afterward.
    rig = Rig.from_arena(arena)
    if condition != "stock":
        apply(arena, conditions)
        arena.partner = PracticePartner(
            arena, targets=arena.partner.targets, guard_table=True
        )
    if bounce is not None:
        from .tt29_bounce import load_checkpoint
        from scipy.spatial.transform import Rotation

        learned = load_checkpoint(bounce)
        extr = np.asarray(learned["registration"])
        rig.wrist = rig.wrist.remount(
            extr[3:6],
            Rotation.from_rotvec(extr[:3]).as_matrix(),
            focal_px=learned["focal_px"],
        )
    nominal = FlightPractice()
    memory = Experience()
    if checkpoint is not None:
        from .tt4_transfer_trial import _read

        for e in _read(checkpoint)["checkpoint"]:
            memory.add(
                np.array(e["flight"]), np.array(e["targets"]), e["time_to_contact"]
            )
    rows = []
    try:
        for point, sid in enumerate(ids):
            if reset_memory:
                memory = Experience()
            arena.launch(shots[sid])
            opponent_seed = sid if seed_offset == 0 else sid + seed_offset + point
            arena.partner.reset(seed=opponent_seed)
            pipeline = Delay(conditions, opponent_seed + 29003)
            twin.reset(seed=sid, server=-1, render=False)
            rig.begin()
            eye = OpticalFlight(rig, nominal, learn=True)
            policy = OpticalPlanner(twin, seed=290110)
            planner = RecallPlanner(twin, memory, seed=290110)
            policy.nominal = planner
            arena.streams = tuple(view.split("+"))
            x = arena.joints.copy()
            prev = x.copy()
            lo, hi = (
                twin._joint_low[policy.actuators],
                twin._joint_high[policy.actuators],
            )
            actions = []
            initial = len(memory.examples)
            cpu = perf_counter()
            for step in range(arena.max_steps):
                q = arena.joints.copy()
                t = step * arena.control_dt
                captured = pipeline.observe(t, arena.render_cameras(), q)
                if captured is not None:
                    eye.observe(*captured)
                dq = (q - prev) / arena.control_dt
                if refine:
                    refine_after_bounce(policy, step, eye.latest, q, dq, x)
                u = policy.command(step, eye.latest, q, dq, x)
                x = np.clip(x + u * planner.rates * arena.control_dt, lo, hi)
                prev = q
                applied = pipeline.command(u)
                _, _, done, cap, info = arena.exchange(applied, np.zeros(5))
                actions.append(applied.tolist())
                if any(
                    e["kind"] == "legal_return" and e["side"] == 1
                    for e in info["events"]
                ):
                    if not no_memory:
                        planner.remember()
                    if arena.legal_returns % 10 == 0:
                        print(
                            "live",
                            sid,
                            arena.legal_returns,
                            arena.rival_returns - 1,
                            flush=True,
                        )
                if done or cap or arena.legal_returns + arena.rival_returns >= 101:
                    break
            stats = audit(arena.events)
            rows.append(
                dict(
                    shot=shots[sid],
                    opponent_seed=opponent_seed,
                    score={k: v for k, v in stats.items() if k != "shots"},
                    initial_examples=initial,
                    final_examples=len(memory.examples),
                    retrieved=planner.retrieved,
                    planned=planner.fallback,
                    corrections=getattr(policy, "corrections", 0),
                    elapsed=perf_counter() - cpu,
                    retrieval_s=planner.retrieval_seconds,
                    search_s=planner.fallback_seconds,
                    actions=actions,
                    events=arena.events.copy(),
                )
            )
            print(
                sid,
                stats["returns"],
                len(memory.examples),
                planner.retrieved,
                flush=True,
            )
            checkpoint = [
                dict(
                    flight=e.flight.tolist(),
                    time_to_contact=float(e.time_to_contact),
                    targets=e.targets.tolist(),
                )
                for e in memory.examples
            ]
            _write(
                output,
                dict(
                    view=view,
                    pose=pose,
                    condition=condition,
                    bounce_checkpoint=str(bounce) if bounce else None,
                    seed_offset=seed_offset,
                    refine=refine,
                    reset_memory=reset_memory,
                    no_memory=no_memory,
                    robust_contact=robust_contact,
                    rows=rows,
                    checkpoint=checkpoint,
                ),
            )
    finally:
        twin.close()
        arena.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--ids", type=int, nargs="+", required=True)
    p.add_argument("--view", default="left+right")
    p.add_argument("--pose", default="lateral")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--reset-memory", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--condition", default="stock")
    p.add_argument("--bounce", type=Path)
    p.add_argument("--robust-contact", action="store_true")
    args = p.parse_args()
    run(
        args.feeds,
        args.output,
        args.ids,
        args.view,
        args.pose,
        checkpoint=args.checkpoint,
        refine=args.refine,
        reset_memory=args.reset_memory,
        no_memory=args.no_memory,
        robust_contact=args.robust_contact,
        seed_offset=args.seed_offset,
        condition=args.condition,
        bounce=args.bounce,
    )
