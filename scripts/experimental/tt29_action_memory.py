"""Retrieve and time prior joint commands during a rally, with an optional dynamics check."""

from dataclasses import dataclass
from time import perf_counter, process_time

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

    def __init__(
        self, arena, memory, *, seed, verify=True, blend=False, recall_only=False
    ):
        super().__init__(arena, side=1, seed=seed)
        self.memory = memory
        self.verify = verify
        self.blend = blend
        self.recall_only = recall_only
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
            return None if self.recall_only else super().plan(path)
        key, delay = packet
        dt = a.control_dt
        inc = self.rates * dt
        q = a.data.ctrl[self.axes].copy()
        lo, hi = a._joint_low[self.axes], a._joint_high[self.axes]
        proposed = []
        neighbors = self.memory.nearest(key, 6)
        for e in neighbors:
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
        if self.blend and len(neighbors) > 1:
            # Interpolate practiced motor outputs along the line joining each pair of observed flights.
            raw = list(proposed)
            scale = np.array([0.12, 0.18, 1.2, 1.4, 1.2])
            for j in range(1, len(neighbors)):
                vector = (neighbors[j].flight - neighbors[0].flight) / scale
                weight = (
                    ((key - neighbors[0].flight) / scale)
                    @ vector
                    / max(1e-6, vector @ vector)
                )
                weight = np.clip(weight, -0.20, 1.10)
                if abs(weight) > 0.05:
                    proposed.append(
                        np.clip(raw[0] + weight * (raw[j] - raw[0]), lo, hi)
                    )
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
        if self.recall_only:
            return None
        start = perf_counter()
        tape = super().plan(path)
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
    blend=False,
    observe_until=0.0,
    recall_only=False,
    verify=True,
    camera_period=1,
    identify_delay=False,
    rated=False,
    fast_partner=False,
    probe_delay=False,
    fit_body=False,
    spread="practice",
    seed_offset=0,
    condition="stock",
    bounce=None,
):
    from .tt4_transfer_trial import _write
    from .tt29_endurance import audit
    from .tt29_partner import PartnerArena
    from .tt29_conditions import cooperative_targets, table_settings
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
        rate_scale=0.8 if rated else 1.0,
        velocity_guard=rated,
    )
    twin = RallyArena(**settings, robust_contact=robust_contact)
    if fast_partner:
        from .tt29_partner_fast import (
            FastPartnerArena,
            FastPracticePartner,
            NativeBallFlight,
        )

    arena_class = FastPartnerArena if fast_partner else PartnerArena
    arena = arena_class(
        **settings,
        targets=cooperative_targets(pose, spread),
        plant=table_settings(condition),
        guard_table=True,
    )
    from .tt29_conditions import CONDITIONS, Delay, apply
    from .tt29_encoder_delay import EncoderDelay
    from .tt29_partner import PracticePartner

    conditions = CONDITIONS[condition]
    # The camera rig knows the nominal installation. Perturbations happen afterward.
    rig = Rig.from_arena(arena)
    if condition not in ("stock", "two_surface", "narrow_table"):
        apply(arena, conditions)
        partner_class = FastPracticePartner if fast_partner else PracticePartner
        arena.partner = partner_class(
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
    delay_ident = EncoderDelay()
    if checkpoint is not None:
        from .tt4_transfer_trial import _read

        for e in _read(checkpoint)["checkpoint"]:
            memory.add(
                np.array(e["flight"]), np.array(e["targets"]), e["time_to_contact"]
            )
    rows = []
    calibration = None
    fixed_lag = None
    try:
        if probe_delay or fit_body:
            from .tt29_encoder_delay import calibrate_delay

            calibration_pipeline = Delay(conditions, 294601)
            fixed_lag, calibration = calibrate_delay(
                arena, twin, calibration_pipeline.command, body=fit_body
            )
            delay_ident.lag = fixed_lag
            print("calibration", fixed_lag, calibration, flush=True)
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
            planner = RecallPlanner(
                twin,
                memory,
                seed=290110,
                blend=blend,
                recall_only=recall_only,
                verify=verify,
            )
            if fast_partner:
                planner.flight = NativeBallFlight(twin)
            policy.nominal = planner
            arena.streams = tuple(view.split("+"))
            x = arena.joints.copy()
            prev = x.copy()
            lo, hi = (
                twin._joint_low[policy.actuators],
                twin._joint_high[policy.actuators],
            )
            actions = []
            # Point reset starts a new time series while retaining the learned delay.
            delay_ident.targets.clear()
            delay_ident.encoders.clear()
            lag_history = []
            initial = len(memory.examples)
            cpu = perf_counter()
            started_cpu = process_time()
            for step in range(arena.max_steps):
                q = arena.joints.copy()
                t = step * arena.control_dt
                captured = (
                    pipeline.observe(t, arena.render_cameras(), q)
                    if step % camera_period == 0
                    else None
                )
                if captured is not None:
                    eye.observe(*captured)
                dq = (q - prev) / arena.control_dt
                if refine:
                    refine_after_bounce(policy, step, eye.latest, q, dq, x)
                sample = eye.latest
                if observe_until and not policy.incoming and sample is not None:
                    position, velocity = sample.advance(t)
                    if (
                        velocity[1] < -1.0
                        and (-1.29 - position[1]) / velocity[1] > observe_until
                    ):
                        sample = None
                ahead = 0
                q_control, dq_control = q, dq
                if identify_delay or probe_delay or fit_body:
                    delay_ident.observe(q, x, fixed=fixed_lag)
                    ahead = delay_ident.lag
                    q_control, delta = delay_ident.predict(q, q - prev)
                    dq_control = delta / arena.control_dt
                    if step % 12 == 0:
                        lag_history.append(ahead)
                u = policy.command(step + ahead, sample, q_control, dq_control, x)
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
                    delay_estimates=lag_history,
                    elapsed=perf_counter() - cpu,
                    process_cpu_s=process_time() - started_cpu,
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
                    blend=blend,
                    observe_until=observe_until,
                    recall_only=recall_only,
                    verify=verify,
                    camera_period=camera_period,
                    identify_delay=identify_delay,
                    rated=rated,
                    fast_partner=fast_partner,
                    calibrated_lag=fixed_lag,
                    delay_probe=calibration,
                    fit_body=fit_body,
                    spread=spread,
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
    p.add_argument("--blend", action="store_true")
    p.add_argument("--observe-until", type=float, default=0.0)
    p.add_argument("--recall-only", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--camera-period", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--fast-partner", action="store_true")
    p.add_argument("--spread", choices=("practice", "varied"), default="practice")
    p.add_argument("--fit-body", action="store_true")
    p.add_argument("--probe-delay", action="store_true")
    p.add_argument(
        "--rated",
        action="store_true",
        help="Screen candidate motor speed and retain its checked follow-through",
    )
    p.add_argument("--identify-delay", action="store_true")
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
        blend=args.blend,
        observe_until=args.observe_until,
        recall_only=args.recall_only,
        verify=not args.no_verify,
        camera_period=args.camera_period,
        identify_delay=args.identify_delay,
        rated=args.rated,
        fast_partner=args.fast_partner,
        probe_delay=args.probe_delay,
        fit_body=args.fit_body,
        spread=args.spread,
        seed_offset=args.seed_offset,
        condition=args.condition,
        bounce=args.bounce,
    )
