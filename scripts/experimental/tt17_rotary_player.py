"""Concentrate the learned predictive controller on the later collision deadline."""

import numpy as np

from .tt6_bounded import _clipped_plan
from .tt6_continuous import PredictiveServo
from .tt15_rotary_player import LaterReplyPlayer


class DeadlineServo(PredictiveServo):
    later_strike = False
    weight_deadline = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline_frames = 0
        self.last_weighted = False

    def command(self, goal, lead, pitch):
        self.last_weighted = False
        if (not self.weight_deadline or not self.later_strike or self.intercept is None
                or self.intercept <= self.observer.delay + .75):
            return super().command(goal, lead, pitch)
        # A command's future weight reflects whether it can reach the collision
        # in time. Later pose targets remain as a weak continuation objective.
        interval = np.arange(1, self.horizon + 1) - min(self.intercept, self.horizon)
        initial = self.scale
        self.scale = initial * (.48 + 1.60 * np.exp(-.50 * (interval / 1.30)**2))
        self.deadline_frames += 1
        self.last_weighted = True
        try:
            return super().command(goal, lead, pitch)
        finally:
            self.scale = initial


class DeadlinePlayer(LaterReplyPlayer):
    reply_servo = DeadlineServo

    def __init__(self, *args, mode, **kwargs):
        self.weight_deadline = mode == 'deadline'
        super().__init__(*args, mode=mode, **kwargs)

    def reset(self):
        super().reset()
        self.receiving_servo.weight_deadline = self.weight_deadline

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        if self.receive_from is not None:
            self.trace[-1]['deadline_weighted'] = self.receiving_servo.last_weighted
        return command

    def finish(self, reward):
        return dict(super().finish(reward), deadline_frames=self.receiving_servo.deadline_frames)


def measured_pace(trace, contact):
    views = [(row['tick'], row['blade'][1]) for row in trace
             if contact - 3. <= row['tick'] <= contact + 1.25 and row['blade'] is not None]
    if len(views) < 3:
        return None
    tick, depth = np.asarray(views).T
    if tick[-1] < contact - 1.5 or max(np.diff(tick)) > 2.:
        return None
    centered = tick - tick.mean()
    pace = centered @ (depth - depth.mean()) / (centered @ centered)
    residual = depth - depth.mean() - centered * pace
    if not .018 < pace < .10 or max(np.abs(residual)) > .030:
        return None
    return float(pace)


def replay_first_pace(trace, receive_from):
    """Infer when the live player would have measured its first blade swing."""
    previous = -100.
    for index, frame in enumerate(trace):
        tick, horizon = frame['tick'], frame['horizon']
        if receive_from is not None and tick >= receive_from and horizon is not None and -1.5 < horizon < 2.5:
            previous = tick + horizon
        if -50. < previous < tick - 3.:
            return measured_pace(trace[:index + 1], previous)
    return None


class ForwardServo(PredictiveServo):
    """Reuse an achieved camera-measured pace when the reply plan falls short."""

    later_strike = False
    reference_pace = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed_frames = 0
        self.last_weighted = False

    def command(self, goal, lead, pitch):
        self.last_weighted = False
        return super().command(goal, lead, pitch)

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if (not self.later_strike or self.reference_pace is None or self.intercept is None
                or self.intercept <= self.observer.delay + .75):
            return
        motion = np.diff(self.fit.pose(states)[:, 1])
        response = np.einsum('n,tmn->tmn', jac[1], np.diff(effect, axis=0)).reshape(len(motion), -1)
        plan = np.linalg.solve(normal, rhs).reshape(self.moves, self.fit.action_size)
        feasible = _clipped_plan(plan, self.requested)
        planned = motion + response @ feasible.ravel()
        times = np.arange(2, self.horizon + 1) - .5
        close = np.abs(times - self.intercept) <= 1.75
        short = close & (planned < self.reference_pace - .006)
        if not np.any(short):
            return
        weight = 2.8 * np.exp(-.50 * ((times[short] - self.intercept) / 1.30)**2)
        design = response[short] * weight[:, None]
        normal += design.T @ design
        rhs += design.T @ ((self.reference_pace - motion[short]) * weight)
        self.speed_frames += 1
        self.last_weighted = True


class MeasuredSpeedPlayer(LaterReplyPlayer):
    """Calibrate the second strike's speed from the observed first swing."""

    reply_servo = ForwardServo

    def reset(self):
        super().reset()
        self.reference_ready = False
        self.first_pace = None

    def _stroke_goals(self, estimate):
        goals = super()._stroke_goals(estimate)
        self.receiving_servo.reference_pace = self.first_pace if self.subsequent else None
        return goals

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        tick = self.scene.tick
        if not self.reference_ready and -50. < self.last_attempt < tick - 3.:
            self.first_pace = measured_pace(self.trace, self.last_attempt)
            self.reference_ready = True
        if self.receive_from is not None:
            self.trace[-1]['speed_weighted'] = self.receiving_servo.last_weighted
        return command

    def finish(self, reward):
        return dict(super().finish(reward), deadline_frames=0, speed_frames=self.receiving_servo.speed_frames,
                    first_optical_pace=self.first_pace)
