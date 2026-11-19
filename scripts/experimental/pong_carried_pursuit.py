"""Carry only the visual pursuit estimate between Pong points."""

from dataclasses import dataclass
import math

from pong_risk_aware_placement import RiskAwarePlacement
from run_pong_score_placement import build_controller


PURSUIT_RETENTION = .88


@dataclass(frozen=True)
class PursuitMemory:
    pursuit_floor: float = 0.

    def __post_init__(self):
        if not math.isfinite(self.pursuit_floor) or not 0. <= self.pursuit_floor <= 1.:
            raise ValueError('bounded public pursuit evidence required')

    def to_state(self):
        return dict(pursuit_floor=self.pursuit_floor)


class CarriedPursuitPlacement(RiskAwarePlacement):
    """Retain a decaying opposing-paddle pursuit estimate from public images."""

    def __init__(self, timing, placement, memory, *, forecast_center=False):
        super().__init__(timing, placement, forecast_center=forecast_center)
        self.pursuit_floor = memory.pursuit_floor
        self.memory = memory

    def observe(self, **transition):
        super().observe(**transition)
        if transition['terminated'] or transition['truncated']:
            self.memory = PursuitMemory(self.pursuit_floor * PURSUIT_RETENTION)


def build_carried_pursuit_controller(policy_seed, motor_state, timing_state, score_policy,
                                     exploration=0., *, memory=None, timing_updates=False,
                                     forecast_center=False):
    standard = build_controller(policy_seed, motor_state, timing_state, score_policy, exploration,
                                timing_updates=timing_updates, forecast_center=forecast_center)
    return CarriedPursuitPlacement(standard.timing, standard.placement,
                                  PursuitMemory() if memory is None else memory,
                                  forecast_center=forecast_center)
