"""Choose a reachable shot using public motor and opponent motion forecasts."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from pong_rapid_motor_recovery import (
    CHANGE_THRESHOLD, INVERSE_ORDERS, POSITION_LIMIT, RESPONSE_SPEEDS, SPEED_LIMIT,
    motor_dynamics,
)
from pong_timing_controller import reflect


@dataclass(frozen=True)
class ShotGeometry:
    own_plane: float
    opposing_plane: float
    boundary: float
    ball_speed: float
    angle_per_offset: float
    chase_gain: float
    opponent_speeds: tuple


OFFSETS = np.array([0., .055, -.055, .075, -.075, .087, -.087, .096, -.096])


def motor_bounds(timing, duration, observed_position):
    """Project extreme admissible future drives with the pending request queue."""
    belief = timing.belief
    if belief.states is None or belief.probabilities[belief.confirmed] < CHANGE_THRESHOLD:
        return None
    model = belief.confirmed
    state = np.tile(belief.states[model], (2, 1))
    state[:, 0] = observed_position
    queued = list(timing.requests)[-timing.delay_commands:] if timing.delay_commands else []
    queued = [belief.order[1]] * (timing.delay_commands - len(queued)) + queued

    def advance(seconds, drive):
        transition, response = motor_dynamics(seconds)
        state[:] = state @ transition.T + drive * response
        state[:, 1] = np.clip(state[:, 1], -SPEED_LIMIT, SPEED_LIMIT)
        state[abs(state[:, 0]) > POSITION_LIMIT, 1] = 0.
        np.clip(state[:, 0], -POSITION_LIMIT, POSITION_LIMIT, out=state[:, 0])

    for request in queued:
        step = min(duration, timing.camera_seconds)
        advance(step, RESPONSE_SPEEDS[INVERSE_ORDERS[model, request]])
        duration -= step
        if duration <= 0:
            return tuple(state[:, 0])
    while duration > 1e-8:
        step = min(.045, duration)
        advance(step, RESPONSE_SPEEDS[[2, 0], None])
        duration -= step
    return tuple(state[:, 0])


class ReachablePlacement:
    """Retain visual interception when a contact placement has weak support."""

    def __init__(self, timing, geometry, *, aimed=True):
        self.timing = timing
        self.geometry = geometry
        self.aimed = aimed
        self.latest = None
        self.opponent_history = deque(maxlen=9)
        self.opponent_speed_seen = 0.
        self.arrival_at = None
        self.approach = 0
        self.offset = None
        self.arrival_estimates = deque(maxlen=6)
        self.shots = []
        self.safety_reductions = 0
        timing.contact_offset = self._offset

    def choose(self, entities, baseline_action):
        self.latest = entities
        return self.timing.choose(entities, baseline_action)

    def observe(self, **transition):
        return self.timing.observe(**transition)

    def _visual_motion(self):
        binding = self.timing.inherited.bound_ids
        history = self.timing.inherited.history
        if binding is None or len(history) < 4:
            return None
        observed = [item for item in self.latest if item['observed'] and item['id'] not in binding]
        if len(observed) != 1:
            return None
        axis = self.timing.axis
        opponent = observed[0]
        opposite = opponent['normalized_position'][1 - axis]
        if abs(opposite - self.geometry.opposing_plane) > .08:
            return None
        y = opponent['normalized_position'][axis]
        now = self.timing.seconds
        if self.opponent_history and now - self.opponent_history[-1][0] > .12:
            self.opponent_history.clear()
        self.opponent_history.append((now, y))
        v = 0.
        first_time, first_y = self.opponent_history[0]
        if now - first_time > .09:
            v = (y - first_y) / (now - first_time)
            if .10 < abs(v) < .85 and abs(y) < .37:
                self.opponent_speed_seen = max(abs(v), self.opponent_speed_seen)
        return y, v

    def _offset(self, context):
        if context is None:
            return 0.
        now = self.timing.seconds
        arrival = now + context.seconds
        if (self.arrival_at is not None and arrival - self.arrival_at > .70
                and now > self.arrival_at - .20):
            self.approach += 1
            self.offset = None
            self.arrival_estimates.clear()
        self.arrival_at = arrival
        opponent = self._visual_motion()
        if context.seconds > 1.10:
            return 0.
        history = self.timing.inherited.history
        if len(history) < 4:
            return 0.
        first, last = history[0], history[-1]
        elapsed = last[0] - first[0]
        if elapsed <= 0:
            return 0.
        vertical, horizontal = (last[1] - first[1]) / elapsed, (last[2] - first[2]) / elapsed
        if horizontal >= -.12:
            return 0.
        duration = max(0., (self.geometry.own_plane - last[2]) / horizontal - (now - last[0]))
        intercept = float(reflect(last[1] + (duration + now - last[0]) * vertical, self.geometry.boundary))
        self.arrival_estimates.append(intercept)
        target = intercept
        if self.aimed and (self.offset is not None or duration >= .30):
            bounds = motor_bounds(self.timing, duration, context.paddle)
            if bounds is not None:
                lower, upper = bounds
                uncertainty = float(np.ptp(self.arrival_estimates))
                allowed = .098 - min(.036, .45 * uncertainty)
                if self.offset is None:
                    # Choose once per approach after both the shot and its
                    # paddle destination have enough visual support.
                    prepared = min(.022, max(0., .08 * duration))
                    legal = (max(lower + prepared, intercept - allowed),
                             min(upper - prepared, intercept + allowed))
                    choice = self._select_placement(intercept, duration, legal, last[1], vertical, opponent)
                    if choice is not None:
                        self.offset, value = choice
                        self.shots.append((now, duration, intercept, self.offset, value))
                if self.offset is not None:
                    intended = intercept + float(np.clip(self.offset, -allowed, allowed))
                    target = float(np.clip(intended, min(lower, intercept), max(upper, intercept)))
                    self.safety_reductions += int(abs(target - intended) > .004)
        target = float(np.clip(target, context.lower, context.upper))
        return target - context.intercept or 1e-9

    def _select_placement(self, intercept, duration, legal, ball_y, vertical, opponent):
        if opponent is None:
            return None
        lower, upper = legal
        centers = intercept + OFFSETS
        candidates = OFFSETS[(centers >= lower) & (centers <= upper)]
        if not np.any(candidates != 0):
            return None
        values = self._clearance(candidates, intercept, duration, ball_y, vertical, opponent)
        best = int(values.argmax())
        return float(candidates[best]), float(values[best])

    def _clearance(self, offsets, intercept, duration, ball_y, ball_velocity, opponent):
        geometry = self.geometry
        caps, position, velocity = self._opponent_at_crossing(
            ball_y, ball_velocity, duration, opponent, columns=len(offsets))
        step = .045
        angles = -offsets * geometry.angle_per_offset
        speed_y = geometry.ball_speed * np.sin(angles)
        crossing = (geometry.opposing_plane - geometry.own_plane) / (geometry.ball_speed * np.cos(angles))
        for elapsed in np.arange(0., crossing.max(), step):
            ball = reflect(intercept + elapsed * speed_y, geometry.boundary)
            active = np.clip(crossing - elapsed, 0., step)
            desired = np.clip(geometry.chase_gain * (ball - position), -caps, caps)
            velocity += (active / .13) * (desired - velocity)
            position = np.clip(position + active * velocity, -POSITION_LIMIT, POSITION_LIMIT)
        incoming = reflect(intercept + crossing * speed_y, geometry.boundary)
        return np.min(abs(position - incoming), axis=0) - .06 * abs(offsets)

    def _opponent_at_crossing(self, ball_y, ball_velocity, duration, opponent, *, columns=1):
        geometry = self.geometry
        opponent_y, opponent_velocity = opponent
        slow, fast = geometry.opponent_speeds
        speed = np.clip(self.opponent_speed_seen * 1.06, slow, fast)
        # Forecast a family of plausible speeds and prefer shots whose
        # clearance survives the remaining ambiguity about the opponent.
        caps = np.array([speed, min(max(speed, slow) + .13, fast)])[:, None]
        position = np.full((len(caps), columns), opponent_y)
        velocity = np.full_like(position, opponent_velocity)

        def advance(target, elapsed):
            desired = np.clip(geometry.chase_gain * (target - position), -caps, caps)
            velocity[:] += min(1., elapsed / .13) * (desired - velocity)
            position[:] = np.clip(position + elapsed * velocity, -POSITION_LIMIT, POSITION_LIMIT)

        step = .045
        for elapsed in np.arange(0., duration, step):
            advance(reflect(ball_y + elapsed * ball_velocity, geometry.boundary), min(step, duration - elapsed))
        return caps, position, velocity


class GuardedScorePlacement(ReachablePlacement):
    """Select a reachable placement using the unchanged frozen score ranking."""

    def __init__(self, timing, geometry, placement, *, aimed=True):
        if placement.policy.updates_enabled or placement.exploration:
            raise ValueError('frozen score placement required')
        super().__init__(timing, geometry, aimed=aimed)
        self.placement = placement

    def _select_placement(self, intercept, duration, legal, ball_y, vertical, opponent):
        if legal[0] > legal[1]:
            return None
        belief = self.timing.belief
        if belief.probabilities[belief.confirmed] < .85 or self.timing.candidate is not None:
            return None
        offset = self.placement.offset(self.approach, intercept, duration, legal)
        choices = self.placement.choices
        if choices and choices[-1].approach == self.approach:
            return offset, None
        return None


class OutboundReplyPlacement(GuardedScorePlacement):
    """Use a predicted opponent rebound to move during an outbound flight."""

    def __init__(self, timing, geometry, placement):
        super().__init__(timing, geometry, placement)
        self.reply_moves = 0

    def _offset(self, context):
        if context is not None:
            return super()._offset(context)
        if (not self.timing.delay_commands or self.arrival_at is None
                or self.timing.seconds < self.arrival_at):
            return 0.
        history = self.timing.inherited.history
        if len(history) < 5:
            return 0.
        first, last = history[0], history[-1]
        duration = last[0] - first[0]
        stale = self.timing.seconds - last[0]
        if duration <= 0. or stale > .065:
            return 0.
        vertical, horizontal = (last[1] - first[1]) / duration, (last[2] - first[2]) / duration
        if not .20 < horizontal < .75:
            return 0.
        geometry = self.geometry
        through_paddle = (geometry.opposing_plane - last[2]) / horizontal - stale
        if not .085 <= through_paddle <= .62:
            return 0.
        opponent = self._visual_motion()
        if opponent is None:
            return 0.
        current_ball = reflect(last[1] + stale * vertical, geometry.boundary)
        _, paddle, _ = self._opponent_at_crossing(current_ball, vertical, through_paddle, opponent)
        ball_contact = reflect(current_ball + through_paddle * vertical, geometry.boundary)
        gap = ball_contact - paddle[:, 0]
        if np.any(abs(gap) > .103):
            return 0.
        angle = np.clip(gap * geometry.angle_per_offset, -.91, .91)
        reply = reflect(ball_contact + (geometry.opposing_plane - geometry.own_plane) * np.tan(angle),
                        geometry.boundary)
        if np.ptp(reply) > .24:
            return 0.
        target = .70 * float(np.mean(reply)) + .30 * current_ball
        target = float(np.clip(target, -POSITION_LIMIT, POSITION_LIMIT))
        self.reply_moves += 1
        lead = self.timing.delay_commands * self.timing.camera_seconds
        return target - current_ball - vertical * lead
