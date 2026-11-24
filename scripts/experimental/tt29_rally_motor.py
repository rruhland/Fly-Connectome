"""Check signed 100-return torque and motion against a nominal linear motor envelope."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt25_arm_replay import NOMINAL_NO_LOAD
from .tt29_partner import PartnerArena
from .tt29_rally_physics import OUTPUT, event_metrics


class MotorTrials(PartnerArena):
    """Read paired speed and effort at each physics step, optionally derating motoring torque."""

    def __init__(self, *, stall, derate=False, **settings):
        self.stall = stall
        self.derate = derate
        self.angular = []
        self.effort = []
        super().__init__(**settings)
        self.drives = self._arm_ids[1][2][:5]
        self.dofs = self._near_dofs[:5]
        if not np.all(self.model.actuator_gear[self.drives, 0] == 1):
            raise ValueError("Expected direct-drive torque units at the modeled joint")
        self.nominal_limits = self.model.actuator_forcerange[self.drives].copy()

    def _aerodynamics(self):
        super()._aerodynamics()
        self.angular.append(self.data.qvel[self.dofs].copy())
        if self.derate:
            scaled = self.angular[-1] / NOMINAL_NO_LOAD
            forward = self.stall * np.clip(1.0 - scaled, 0.0, 1.0)
            reverse = self.stall * np.clip(1.0 + scaled, 0.0, 1.0)
            self.model.actuator_forcerange[self.drives] = np.column_stack((-reverse, forward))

    def _contacts(self, incoming):
        self.effort.append(self.data.actuator_force[self.drives].copy())
        if self.derate:
            self.model.actuator_forcerange[self.drives] = self.nominal_limits
        super()._contacts(incoming)


def _arena(record, stall, derate=False):
    configuration = record["configuration"].copy()
    lanes = configuration.pop("partner_lanes")
    fast = configuration.pop("partner_fast", False)
    arena = MotorTrials(stall=stall, derate=derate, targets=lanes, **configuration)
    if fast:
        from .tt29_partner_fast import FastPracticePartner

        arena.partner = FastPracticePartner(arena, targets=lanes, guard_table=arena.partner.guard_table)
    return arena


def _record(document, shot_id):
    if isinstance(document, list):
        return max((r for r in document if shot_id is None or r["launch"]["id"] == shot_id),
                   key=lambda r: r["returns"])
    if "configuration" in document:
        if shot_id is not None and document["launch"]["id"] != shot_id:
            raise ValueError("The requested launch is absent")
        return document

    from .tt29_conditions import cooperative_targets

    if document.get("condition", "stock") != "stock":
        raise ValueError("Camera action replay currently expects the stock condition")
    row = max((r for r in document["rows"] if shot_id is None or r["shot"]["id"] == shot_id),
              key=lambda r: r["score"]["returns"])
    width, height = document.get("size", (384, 288))
    lane = document.get("spread", "practice")
    pose = document["pose"]
    configuration = dict(width=width, height=height, strike_ready=pose, guard_table=True,
                         partner_fast=document.get("fast_partner", False),
                         motor_envelope=document.get("motor_envelope", False),
                         partner_lanes=cooperative_targets(pose, lane))
    return dict(configuration=configuration, launch=row["shot"], opponent_seed=row["opponent_seed"],
                returns=row["score"]["returns"], events=row["events"], actions=[(u, [0.] * 5) for u in row["actions"]])


def envelope(angular, effort, stall, tick):
    speed, torque = np.asarray(angular), np.asarray(effort)
    if torque.shape != speed.shape:
        raise ValueError("Every actuator force requires a matching pre-step joint velocity")
    voltage = np.abs(torque / stall + speed / NOMINAL_NO_LOAD)
    excess = voltage > 1.000001
    power = speed * torque
    motoring = power > 1e-5
    braking = power < -1e-5
    possible = stall * np.maximum(0.0, 1.0 - np.abs(speed) / NOMINAL_NO_LOAD)
    positive = np.where(motoring, power, 0.0)
    extra = np.where(motoring, np.maximum(0.0, np.abs(torque) - possible) * np.abs(speed), 0.0)
    return dict(
        integrated_steps=len(speed), joint_samples=speed.size,
        per_joint_peak_actual_speed_rad_s=np.max(np.abs(speed), axis=0).tolist(),
        per_joint_peak_voltage_proxy=np.max(voltage, axis=0).tolist(),
        joint_samples_above_voltage_proxy=int(excess.sum()),
        steps_with_any_excess=int(excess.any(axis=1).sum()),
        per_joint_motoring_samples=motoring.sum(axis=0).tolist(),
        per_joint_motoring_excess=(excess & motoring).sum(axis=0).tolist(),
        per_joint_braking_samples=braking.sum(axis=0).tolist(),
        per_joint_braking_excess=(excess & braking).sum(axis=0).tolist(),
        motoring_work_j=float(positive.sum() * tick),
        excess_motoring_work_j=float(extra.sum() * tick),
    )


def measure(record, stall, derate=False):
    arena = _arena(record, stall, derate)
    arena.launch(record["launch"])
    if "opponent_seed" in record:
        arena.partner.reset(seed=record["opponent_seed"])
    try:
        for near, far in record["actions"]:
            _, _, done, cap, _ = arena.exchange(near, far)
            if done or cap:
                break
            if derate and arena.data.time > 8.0:
                break
        if not derate and json.dumps(arena.events) != json.dumps(record["events"]):
            raise ValueError("Unchanged motor audit replay changed recorded events")
        return dict(
            receipt_returns=record["returns"], replay_returns=len(event_metrics(arena.events)),
            simulated_seconds=arena.data.time, reason=arena.reason,
            stall_limit_nm=stall, no_load_rad_s=float(NOMINAL_NO_LOAD),
            derated_replay=derate,
            **envelope(arena.angular, arena.effort, stall, arena.model.opt.timestep),
        )
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--stall", type=float, default=3.35)
    parser.add_argument("--id", type=int)
    parser.add_argument("--derate", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT / "rally-physics-motor-envelope.json")
    args = parser.parse_args()
    if args.stall <= 0:
        parser.error("stall torque must be positive")
    row = _record(_read(args.receipt), args.id)
    result = measure(row, args.stall, args.derate)
    _write(args.output, dict(receipt_sha256=_hash(args.receipt), **result))
    print(result, flush=True)


if __name__ == "__main__":
    main()
