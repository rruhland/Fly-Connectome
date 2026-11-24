"""Prospective fixed-phase control for the successful timing retries."""
import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import tt29_vision_twin
from .tt29_action_memory import run
from .tt4_transfer_trial import _read, _write


class LeadPlanner(tt29_vision_twin.OpticalPlanner):
    def __init__(self, twin, *, seed, ticks):
        super().__init__(twin, seed=seed)
        self.lead = ticks * twin.control_dt

    def _refresh(self, step, flight, angles, rate, target):
        # Seed the isolated dynamics trials ahead along the measured flight.
        # Observation acceptance and the live body continue on their own clocks.
        forecast = replace(flight, time=flight.time - self.lead)
        super()._refresh(step, forecast, angles, rate, target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("movement_checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lead", type=float, default=1.0)
    p.add_argument("--ids", type=int, nargs="+", default=[7271, 7345, 11292, 4187, 7345, 11292])
    p.add_argument("--seed-offset", type=int, default=62470)
    args = p.parse_args()
    def make(twin, *, seed):
        return LeadPlanner(twin, seed=seed, ticks=args.lead)
    with patch.object(tt29_vision_twin, "OpticalPlanner", make):
        run(args.feeds, args.output, args.ids, "left+right", "lateral", checkpoint=args.movement_checkpoint,
            robust_contact=True, rally_objective=True, recent=True, blend=True, rated=True,
            motor_envelope=True, fast_partner=True, cached_stereo=True, camera_period=2,
            net_rescue=True, spread="varied", return_target=32, seed_offset=args.seed_offset)

    record = _read(args.output)
    record["phase_lead_ticks"] = args.lead
    _write(args.output, record)
