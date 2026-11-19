"""Apply learned Pong shot placement while the visual approach supports it."""

from collections import deque
import math
from statistics import median

from pong_score_placement import ScorePlacementController


IMAGE_SAMPLE = 1 / 64
MIN_PURSUIT_SECONDS = .10
PURSUIT_LIMIT = .43


class RiskAwarePlacement(ScorePlacementController):
    """Retain center interception if a learned edge shot has low visual support."""

    def __init__(self, timing, placement, *, forecast_center=False):
        super().__init__(timing, placement, forecast_center=forecast_center)
        self.entities = ()
        self.opponent_id = None
        self.opponent_track = deque(maxlen=9)
        self.pursuit_floor = 0.
        self.ball_speeds = deque(maxlen=13)
        self.forecast = deque(maxlen=5)
        self.forecast_approach = 0
        self.cancelled = {}

    def choose(self, entities, baseline_action):
        self.entities = entities
        return super().choose(entities, baseline_action)

    def _observe_pursuit(self):
        binding = self.timing.inherited.bound_ids
        if binding is None:
            self.opponent_track.clear()
            return
        axis = self.timing.axis
        observed = {item['id']: item['normalized_position'] for item in self.entities if item['observed']}
        roles = [identity for identity in observed if identity not in binding]
        if len(roles) != 1 or not all(identity in observed for identity in binding):
            self.opponent_track.clear()
            return
        identity = roles[0]
        current = observed[identity]
        cross = current[1 - axis]
        own, ball = (observed[identity] for identity in binding)
        if abs(cross - own[1 - axis]) < .30 or abs(cross - ball[1 - axis]) < .05:
            self.opponent_track.clear()
            return
        now = self.timing.seconds
        history = self.opponent_track
        if (identity != self.opponent_id or (history and (
                now - history[-1][0] > .06 or abs(cross - history[0][2]) > IMAGE_SAMPLE))):
            history.clear()
        self.opponent_id = identity
        history.append((now, current[axis], cross))
        elapsed = now - history[0][0]
        if elapsed >= MIN_PURSUIT_SECONDS:
            lower_speed = (abs(current[axis] - history[0][1]) - IMAGE_SAMPLE) / elapsed
            if lower_speed <= 1.:
                self.pursuit_floor = max(self.pursuit_floor, lower_speed)
        ball_track = self.timing.inherited.history
        if len(ball_track) >= 5:
            past, recent = ball_track[0], ball_track[-1]
            interval = recent[0] - past[0]
            if interval > 0:
                speed = math.hypot(recent[1] - past[1], recent[2] - past[2]) / interval
                if .3 <= speed <= .85:
                    self.ball_speeds.append(speed)

    def _reason(self, context):
        ball_speed = median(self.ball_speeds) if self.ball_speeds else .55
        if self.pursuit_floor > max(PURSUIT_LIMIT, .80 * ball_speed):
            return 'pursuit'
        if not self.timing.delay_commands:
            return None
        remaining = context.seconds - self.timing.delay_commands * self.timing.camera_seconds
        belief = self.timing.belief
        if remaining < .48 and (belief.probabilities[belief.confirmed] < .65
                                or self.timing.candidate_dwell):
            return 'uncertain_actuation'
        if (len(self.forecast) >= 4 and remaining < .55
                and max(self.forecast) - min(self.forecast) > 4 * IMAGE_SAMPLE):
            return 'unstable_crossing'
        return None

    def _offset(self, context):
        self._observe_pursuit()
        planned = super()._offset(context)
        if context is None:
            return planned
        if self.forecast_approach != self.approach:
            self.forecast_approach = self.approach
            self.forecast.clear()
        self.forecast.append(context.intercept)
        if self.approach in self.cancelled:
            return 0.
        if planned and self.placement.choices[-1].arm != 0:
            reason = self._reason(context)
            if reason is not None:
                self.cancelled[self.approach] = reason
                return 0.
        return planned

    def observe(self, **transition):
        terminal = transition['terminated'] or transition['truncated']
        choices = self.placement.choices
        if terminal and choices and choices[-1].approach in self.cancelled:
            # The guarded interception did not execute the assigned shot.
            self.timing.observe(**transition)
            self.placement.done = True
        else:
            super().observe(**transition)


def build_risk_aware_controller(policy_seed, motor_state, timing_state, score_policy,
                                exploration=0., *, timing_updates=False, forecast_center=False):
    """Opt into identical risk monitoring for trained, exploration, and center arms."""
    from run_pong_score_placement import build_controller

    standard = build_controller(policy_seed, motor_state, timing_state, score_policy, exploration,
                                timing_updates=timing_updates, forecast_center=forecast_center)
    return RiskAwarePlacement(standard.timing, standard.placement, forecast_center=forecast_center)
