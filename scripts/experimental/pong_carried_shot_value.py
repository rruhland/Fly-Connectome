"""Remember visible opponent pursuit and score outcomes of shots taken under lag."""
from dataclasses import asdict, dataclass
import math

from pong_carried_pursuit import CarriedPursuitPlacement, PursuitMemory
from run_pong_score_placement import build_controller


@dataclass(frozen=True)
class ShotMemory:
    pursuit_floor: float = 0.
    delayed_win_estimate: float = .6
    skipped_points: int = 0

    def __post_init__(self):
        PursuitMemory(self.pursuit_floor)
        if (not math.isfinite(self.delayed_win_estimate) or not 0 <= self.delayed_win_estimate <= 1
                or type(self.skipped_points) is not int or self.skipped_points < 0):
            raise ValueError('invalid carried shot outcomes')

    def to_state(self):
        return asdict(self)


class CarriedShotPlacement(CarriedPursuitPlacement):
    def __init__(self, timing, placement, memory, *, forecast_center=False):
        super().__init__(timing, placement, PursuitMemory(memory.pursuit_floor),
                         forecast_center=forecast_center)
        self.shot_memory = memory
        self.considered_delayed_shot = False
        self.attempted_delayed_shot = False

    def _reason(self, context):
        reason = super()._reason(context)
        if reason or not self.timing.delay_commands:
            return reason
        self.considered_delayed_shot = True
        # When delayed attacks stop producing scores, spend several opportunities
        # defending before sampling the learned shot again.
        if self.shot_memory.delayed_win_estimate < .4 and self.shot_memory.skipped_points < 5:
            return 'recent_delayed_shot_outcomes'
        self.attempted_delayed_shot = True
        return None

    def observe(self, **transition):
        super().observe(**transition)
        if not (transition['terminated'] or transition['truncated']):
            return
        estimate = self.shot_memory.delayed_win_estimate
        skipped = self.shot_memory.skipped_points
        if self.attempted_delayed_shot:
            estimate = .5 * (estimate + int(transition['reward'] > 0))
            skipped = 0
        elif self.considered_delayed_shot:
            skipped = min(5, skipped + 1)
        self.shot_memory = ShotMemory(self.memory.pursuit_floor, estimate, skipped)


def build_carried_shot_controller(policy_seed, motor_state, timing_state, score_policy,
                                  exploration=0., *, memory=None, timing_updates=False,
                                  forecast_center=False):
    standard = build_controller(policy_seed, motor_state, timing_state, score_policy, exploration,
                                timing_updates=timing_updates, forecast_center=forecast_center)
    return CarriedShotPlacement(standard.timing, standard.placement,
                                ShotMemory() if memory is None else memory,
                                forecast_center=forecast_center)
