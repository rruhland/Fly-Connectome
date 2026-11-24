"""Rephase and translate a planned stroke with later RGB and encoder observations."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory
from .tt4_transfer_trial import _read, _write
from .tt29_action_memory import RecallPlanner
from .tt29_vision_twin import OpticalPlanner


class SelectedRollout:
    """Read the contact prediction for the selected tape in an isolated twin."""

    def __init__(self, solver):
        self.solver = solver
        self.batches = []

    def trajectories(self, goals, target, cutoff):
        rates, scores = self.solver.trajectories(goals, target, cutoff)
        self.batches.append((rates, scores, target))
        return rates, scores

    def selected(self, tape):
        for rates, scores, target in reversed(self.batches):
            if np.shares_memory(rates, tape):
                row = (tape.ctypes.data - rates.ctypes.data) // rates.strides[0]
                return scores[row], target
        return None


@dataclass
class ContactPlan:
    time: float
    position: np.ndarray
    target: np.ndarray


class AnchoredRecall(RecallPlanner):
    def plan(self):
        self.anchor = None
        self.edited_rollout = None
        arena = self.arena
        self.initial_target = arena.data.ctrl[self.axes].copy()
        prediction = self.predict
        tracked = SelectedRollout(prediction)
        self.predict = tracked
        try:
            tape = super().plan()
        finally:
            self.predict = prediction
        if tape is not None and (chosen := tracked.selected(tape)) is not None:
            scores, target = chosen
            path = self.flight.path(
                arena.ball, arena.velocity, arena.spin, 1, arena._received_bounce
            )
            if len(path) and path[0, 0] <= scores[1] <= path[-1, 0]:
                position = np.array(
                    [np.interp(scores[1], path[:, 0], axis) for axis in path[:, 1:4].T]
                )
                self.anchor = ContactPlan(
                    arena.steps * arena.control_dt + scores[1], position, target
                )
        return tape


def contact_shift(path, anchor, now):
    """Compare the old and new flight at the same court depth."""
    y = anchor.position[1]
    if len(path) < 2 or not path[-1, 2] < y < path[0, 2]:
        return None
    updated = np.array([np.interp(y, path[::-1, 2], col[::-1]) for col in path.T[:4]])
    delay = float(updated[0] + now - anchor.time)
    return float(updated[0]), delay, updated[1:4] - anchor.position


def edited_goals(base, commanded, offset, translation, reach_time, dt):
    """Shift the held stroke and introduce a smooth, local joint offset."""
    clock = np.arange(1, len(base) + 1)
    shifted = np.stack(
        [
            np.interp(clock - offset / dt, np.r_[0, clock], col)
            for col in np.vstack((commanded, base)).T
        ],
        axis=1,
    )
    elapsed = clock * dt
    ramp = np.clip(elapsed / min(0.065, reach_time), 0.0, 1.0)
    release = np.clip((reach_time + 0.16 - elapsed) / 0.08, 0.0, 1.0)
    shifted[:, :5] += (ramp * release)[:, None] * translation
    return shifted


class ContactEditor(OpticalPlanner):
    """Use later optical estimates to edit the tape around the planned impact."""

    def __init__(self, twin, *, seed, mode):
        super().__init__(twin, seed=seed)
        self.mode = mode
        self.log = []
        self.corrections = 0
        self.last_read = -1.0

    def command(self, step, flight, angles, rate, target):
        original = super().command(step, flight, angles, rate, target)
        planner = self.nominal
        anchor = planner.anchor if self.tape is not None else None
        dt = self.twin.control_dt
        now = step * dt
        age = step - self.first
        if (
            anchor is None
            or age <= 0
            or not self.incoming
            or flight is None
            or not 0.035 < anchor.time - now < 0.16
            or not flight.bounced
            or not self.last_read < flight.time
            or now - flight.time > 0.044
            or flight.rms >= 1.4
            or flight.depth_sigma >= 0.075
            or flight.frames < 4
        ):
            return original
        self.last_read = flight.time
        was_bounced = self.bounced
        self._refresh(step, flight, angles, rate, target)
        self.bounced = was_bounced
        path = planner.flight.path(
            self.twin.ball, self.twin.velocity, self.twin.spin, 1, True
        )
        change = contact_shift(path, anchor, now)
        if change is None:
            return original
        reach_time, offset, displacement = change
        if (
            not 0.032 < reach_time < 0.18
            or abs(offset) > 0.065
            or np.linalg.norm(displacement) > 0.095
        ):
            return original
        low, high = (
            self.twin._joint_low[self.actuators],
            self.twin._joint_high[self.actuators],
        )
        increment = dt * planner.rates
        base = np.clip(
            target + np.cumsum(self.tape[age:], axis=0) * increment, low, high
        )
        if len(base) * dt < reach_time + 0.45:
            return original
        hit_goals = base[min(len(base) - 1, int((anchor.time - now) / dt)), :5]
        _, _, linear, angular = planner.ik.pose(hit_goals, jacobian=True)
        adjustment = np.linalg.lstsq(
            np.r_[linear, 0.17 * angular, 0.035 * np.eye(5)],
            np.r_[displacement, np.zeros(8)],
            rcond=None,
        )[0]
        adjustment *= min(1.0, 0.18 / max(np.max(np.abs(adjustment)), 1e-8))
        phase = float(np.clip(offset, -0.036, 0.036))
        edits = [(0.0, 0.0), (phase, 0.0), (0.0, 1.0), (phase, 0.60), (phase, 1.0)]
        proposals = np.clip(
            np.stack(
                [
                    edited_goals(base, target, shift, gain * adjustment, reach_time, dt)
                    for shift, gain in edits
                ]
            ),
            low,
            high,
        )
        rates, checked = planner.predict.trajectories(
            proposals, anchor.target, reach_time + 0.10
        )
        scores = np.where(np.isfinite(checked[:, 0]), checked[:, 0], np.inf)
        winner = int(scores.argmin())
        accept = bool(
            winner != 0 and scores[winner] < 1.8 and scores[winner] + 0.16 < scores[0]
        )
        self.log.append(
            {
                "step": step,
                "time_to_contact": reach_time,
                "phase_s": offset,
                "offset_m": displacement.tolist(),
                "scores": [
                    float(value) if np.isfinite(value) else None for value in scores
                ],
                "proposed": winner,
                "accepted": accept,
                "applied": accept and self.mode == "edit",
            }
        )
        if not accept or self.mode != "edit":
            return original
        self.corrections += 1
        edited_tail = rates[winner]
        self.tape = np.concatenate((self.tape[:age], edited_tail))
        planner.edited_rollout = (self.tape, edited_tail)
        if planner.pending is not None:
            key, _, delay = planner.pending
            targets = np.clip(
                planner.initial_target + np.cumsum(self.tape, axis=0) * increment,
                low,
                high,
            )
            planner.pending = (key, targets, delay)
        selected_time = float(checked[winner, 1])
        position = np.array(
            [np.interp(selected_time, path[:, 0], axis) for axis in path[:, 1:4].T]
        )
        planner.anchor = ContactPlan(now + selected_time, position, anchor.target)
        return self.tape[age]


def run(
    feeds,
    output,
    ids,
    checkpoint,
    *,
    mode,
    seed_offset,
    return_target,
    optical_checkpoint=None,
):
    from . import tt29_vision_twin

    policies = []

    def make_policy(twin, *, seed):
        policy = ContactEditor(twin, seed=seed, mode=mode)
        policies.append(policy)
        return policy

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(tt29_action_memory, "RecallPlanner", AnchoredRecall)
        )
        stack.enter_context(
            patch.object(tt29_vision_twin, "OpticalPlanner", make_policy)
        )
        if optical_checkpoint is not None:
            from . import tt29_optical_outcomes

            class EditedOpticalOutcomes(tt29_optical_outcomes.OpticalOutcomes):
                def command(self, tape):
                    # Concatenation makes an independent command array. The
                    # original tail view identifies the checked rollout row.
                    edited = getattr(self.planner, "edited_rollout", None)
                    if edited is not None and tape is edited[0]:
                        tape = edited[1]
                    super().command(tape)

            stack.enter_context(
                patch.object(
                    tt29_optical_outcomes, "OpticalOutcomes", EditedOpticalOutcomes
                )
            )
        tt29_action_memory.run(
            feeds,
            output,
            ids,
            "left+right",
            "lateral",
            checkpoint=checkpoint,
            seed_offset=seed_offset,
            return_target=return_target,
            camera_period=2,
            rated=True,
            motor_envelope=True,
            fast_partner=True,
            cached_stereo=True,
            robust_contact=True,
            rally_objective=True,
            recent=True,
            blend=True,
            net_rescue=True,
            spread="varied",
            **(
                {
                    "optical_outcomes": "learn",
                    "optical_checkpoint": optical_checkpoint,
                    "freeze_optical_outcomes": True,
                }
                if optical_checkpoint is not None
                else {}
            ),
        )
    receipt = _read(output)
    receipt["motor_correction"] = mode
    for row, policy in zip(receipt["rows"], policies, strict=True):
        row["motor_checks"] = policy.log
    _write(output, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--optical-checkpoint",
        type=Path,
        help="Apply a frozen optical outcome residual checkpoint",
    )
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("observe", "edit"), default="edit")
    parser.add_argument("--seed-offset", type=int, default=62470)
    parser.add_argument("--return-target", type=int, default=100)
    args = parser.parse_args()
    run(
        args.feeds,
        args.output,
        args.ids,
        args.checkpoint,
        mode=args.mode,
        seed_offset=args.seed_offset,
        return_target=args.return_target,
        optical_checkpoint=args.optical_checkpoint,
    )
