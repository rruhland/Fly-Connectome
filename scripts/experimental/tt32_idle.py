"""Diagnose idle arms and try additional contact times after the stock planner fails."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor, tt29_vision_twin as vision
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_rally_physics import mujoco
from .tt29_vision_replay import clearance_final
from .tt31_curriculum import OUTPUT, play_card
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt31_recovery import Recovery


REFERENCE = OUTPUT / "recovery-full.json.gz"
CHECKPOINT = OUTPUT / "ready-staged-checkpoint.json.gz"
EXTRA_DEPTHS = (1.10, 1.16, 1.42, 1.48, 1.54, 1.60)
DEFAULT_IDS = (10898, 3286, 4869, 8557, 13159)


class TraceVision(vision.OpticalPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def command(self, step, flight, angles, rate, target):
        last_try = self.previous_try
        result = super().command(step, flight, angles, rate, target)
        estimate = None
        if flight is not None:
            p, v = flight.advance(step * self.twin.control_dt)
            estimate = dict(captured_s=float(flight.time), pos=p.tolist(), vel=v.tolist(),
                            sampled_pos=flight.position.tolist(), rms_px=float(flight.rms),
                            depth_sigma_m=float(flight.depth_sigma), bounced=bool(flight.bounced))
        self.trace.append(dict(
            tick=step, estimate=estimate, incoming=bool(self.incoming),
            planned=last_try != self.previous_try,
            active=self.tape is not None and 0 <= step - self.first < len(self.tape),
            tape_age=step - self.first if self.tape is not None else None,
            max_rate=float(np.max(np.abs(result))), angles=angles.tolist(),
            commanded=target.tolist(),
        ))
        return result


class TracePlanner(motor.RecallPlanner):
    extend_window = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def candidates(self, windows, target, timeline):
        fit = self.ik.fit
        checks = []
        start = self.arena.data.ctrl[self.axes][:5]

        def observe(center, normal, initial):
            q, pose = fit(center, normal, initial)
            checks.append((q, pose, normal))
            return q, pose

        with patch.object(self.ik, "fit", observe):
            goals, params = super().candidates(windows, target, timeline)
        fits_per_window = len(checks) // len(windows)
        outcomes = dict(geometry=0, face=0, slew=0, targets=0)
        for i, (q, (at, face, *_), normal) in enumerate(checks):
            pose = windows[i // fits_per_window]
            diff = at - pose[1:4]
            if abs(diff @ face + .025) > .045 or np.linalg.norm(np.cross(diff, face)) > .075:
                outcomes["geometry"] += 1
            elif face @ normal < .82:
                outcomes["face"] += 1
            elif np.any(abs(q - start) > self.rates[:5] * (pose[0] + .045) + .09):
                outcomes["slew"] += 1
            else:
                outcomes["targets"] += 1
        if outcomes["targets"] * 5 != len(goals):
            raise ValueError("Observed rejections differ from the stock IK candidate count")
        self.trace[-1]["generated"].append(outcomes)
        return goals, params

    def extra(self, flight):
        window = flight[(flight[:, 3] > .825) & (flight[:, 3] < 1.34) &
                        (flight[:, 0] > .095)]
        if not len(window):
            return None
        candidates = []
        depths = -window[:, 2]
        for depth in EXTRA_DEPTHS:
            if depths[0] <= depth <= depths[-1]:
                candidates.append(window[np.argmin(abs(depths - depth))])
        # Reuse the ordinary impact depths for early, low portions of the bounce.
        low = window[(window[:, 3] <= .89) & (depths >= 1.19) & (depths <= 1.40)]
        if len(low):
            candidates.append(low[len(low) // 2])
        if not candidates:
            return None
        dt = self.arena.control_dt
        last = max(row[0] for row in candidates)
        timeline = dt * (1 + np.arange(int(np.ceil((last + .68) / dt))))
        target = np.array([0., 1.32, 1.08, 3.7])
        self.trace[-1]["additional_windows"] = [row[:7].tolist() for row in candidates]
        goals, _ = self.candidates(candidates, target, timeline)
        if not len(goals):
            return None
        rates, score = self.predict.trajectories(goals, target, last + .08)
        valid = np.flatnonzero(np.isfinite(score[:, 0]))
        if not len(valid):
            return None
        index = valid[np.argmin(score[valid, 0])]
        self.pending = None
        packet = motor.context(flight, -1.29)
        if packet is not None:
            key, delay = packet
            axes = self.axes
            start = self.arena.data.ctrl[axes]
            lo, hi = self.arena._joint_low[axes], self.arena._joint_high[axes]
            targets = []
            for u in rates[index]:
                start = np.clip(start + u * self.rates * dt, lo, hi)
                targets.append(start)
            self.pending = key, np.stack(targets), delay
        self.trace[-1]["additional_impact_s"] = float(score[index, 1])
        return rates[index]

    def plan(self):
        a = self.arena
        before = len(self.decisions)
        recalled = self.retrieved
        self.trace.append(dict(tick=a.steps, bounced=bool(a._received_bounce), generated=[], batches=[]))
        log = self.trace[-1]
        solve = self.predict.trajectories

        def observe(goals, target, cutoff):
            out = solve(goals, target, cutoff)
            cost = out[1][:, 0]
            good = np.isfinite(cost)
            log["batches"].append(dict(
                count=len(cost), finite=int(good.sum()), under_recall_ceiling=int((cost < 1.75).sum()),
                best=float(cost[good].min()) if good.any() else None,
            ))
            return out

        with patch.object(self.predict, "trajectories", observe):
            action = super().plan()
            if len(self.decisions) > before:
                log["base_search"] = self.decisions[-1]
            log["choice"] = "recall" if self.retrieved > recalled else "search" if action is not None else "hold"
            if action is None and self.extend_window:
                path = self.flight.path(a.ball, a.velocity, a.spin, 1, a._received_bounce)
                action = self.extra(path)
                if action is not None:
                    log["choice"] = "wider-window"
        return action


def physical_replay(row, *, clearance):
    card = row["placements"]["card"]
    a = replay_arena([card])
    cad = replay_arena([card], detail="cad") if clearance else None
    guard = ArmClearance(cad.model, tolerance=0.) if clearance else None
    shadow = mujoco.MjData(cad.model) if clearance else None
    near = []
    hook = a._aerodynamics
    before = None

    def monitor():
        nonlocal before
        if a._last_hitter == -1 and a._received_bounce:
            current = np.r_[a.data.time, a.ball]
            if before is not None and before[2] > -1.10 >= current[2]:
                frac = (-1.10 - before[2]) / (current[2] - before[2])
                near.append((before + frac * (current - before)).tolist())
            before = current
        else:
            before = None
        hook()

    try:
        a.launch(row["shot"])
        a.partner.reset(seed=card["seed"])
        if clearance:
            a.sample_clearance(True)
            a._cad_reader = guard, shadow
        with patch.object(a, "_aerodynamics", monitor):
            for command in row["actions"]:
                a.exchange(command, np.zeros(5))
        event_hash = digest_json(a.events)
        if event_hash != digest_json(row["events"]) or (
            digest_json(a.point_placements()) != digest_json(row["placements"])
        ):
            raise ValueError("Physical replay disagrees with recorded events or placements")
        result = None
        if clearance:
            result = clearance_final(a, guard, shadow)
            if result["samples"] != round(a.data.time / a.model.opt.timestep) + 1:
                raise ValueError("Clearance replay omitted a physics state")
        return dict(receive_crossings=near, clearance=result, events_sha256=event_hash, motor=a.motor_audit())
    finally:
        a.close()
        if cad is not None:
            cad.close()


def evaluate(feeds, output, ids, *, widen, reference, checkpoint, optical=None):
    archived = _read(reference)
    optical = checkpoint if optical is None else optical
    movement_hash = _hash(checkpoint)
    optical_hash = _hash(optical)
    result = dict(reference_sha256=_hash(reference), mode="wider-window" if widen else "trace",
                  ready=archived["ready"], movement_sha256=movement_hash, optical_sha256=optical_hash,
                  opening_side=1, ids=ids, cards=[], rows=[])
    references = {row["shot"]["id"]: row for row in archived["rows"]}

    class Planner(TracePlanner):
        extend_window = widen

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            plans.append(self.trace)

    class Policy(TraceVision):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optics.append(self.trace)

    with centered(archived["ready"]), TemporaryDirectory(prefix="tt32-idle-") as temp:
        point = Path(temp) / "point.json.gz"
        for sid in ids:
            row = references[sid]
            if (row["initial_checkpoint_sha256"] != movement_hash
                    or row["initial_optical_checkpoint_sha256"] != optical_hash):
                raise ValueError("Trace checkpoint does not match the archive")
            plans, optics = [], []
            with (patch.object(motor, "RecallPlanner", Planner),
                  patch.object(motor, "RallyObjective", Recovery),
                  patch.object(vision, "OpticalPlanner", Policy)):
                fresh, _ = play_card(feeds, row["placements"]["card"], checkpoint, optical,
                                     point, practice=False, cap=16)
            if not widen and any(digest_json(row[k]) != digest_json(fresh[k])
                                 for k in ("events", "actions", "placements")):
                raise ValueError(f"Tracing changed archived point {sid}")
            fresh["replay"] = physical_replay(fresh, clearance=widen)
            fresh["plan_trace"], fresh["optical_trace"] = plans[0], optics[0]
            result["cards"].append(fresh["placements"]["card"])
            result["rows"].append(fresh)
            _write(output, result)
            print(sid, row["score"]["learner"], fresh["score"]["learner"],
                  [p["choice"] for p in plans[0]], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--optical", type=Path)
    p.add_argument("--widen", action="store_true")
    a = p.parse_args()
    evaluate(a.feeds, a.output, a.ids, widen=a.widen, reference=a.reference,
             checkpoint=a.checkpoint, optical=a.optical)
