"""Use optical blade readiness and the approaching ball to end rotary recovery."""

import numpy as np

from .tt6_continuous import PredictiveServo
from .tt8_bounded import BoundedServo
from .tt14_rotary_player import FlightCorrectedPlayer


def reply_time(frame, ready_depth):
    if frame['horizon'] is not None:
        return frame['horizon']
    if frame['estimate'] is None:
        return None
    position, velocity = frame['estimate']
    if velocity[1] < -.025 and position[1] > ready_depth:
        return (position[1] - ready_depth) / -velocity[1]
    return None


class OpticalRecoveryPlayer(FlightCorrectedPlayer):
    """Recover actuator travel early, then turn control over to the next strike."""

    def reset(self):
        super().reset()
        self.previous_blade = None

    def update(self, frames, preceding=None):
        inherited = super().update(frames, preceding)
        if self.receive_from is None:
            return inherited
        frame = self.trace[-1]
        tick, horizon = frame['tick'], frame['horizon']
        if horizon is not None and -1.5 < horizon < 2.5:
            self.last_attempt = tick + horizon
        blade = None if frame['blade'] is None else np.asarray(frame['blade'])
        motion = None if blade is None or self.previous_blade is None else float(np.linalg.norm(blade - self.previous_blade))
        self.previous_blade = blade
        if not 5. <= tick - self.last_attempt <= 24.:
            return inherited

        servo = self.receiving_servo
        pending = float(np.max(np.abs(servo.requested)))
        ready = servo.fit.ready
        error = None if blade is None else float(np.linalg.norm(blade - ready))
        remaining = reply_time(frame, ready[1])
        reserve = servo.horizon + servo.observer.delay + np.ceil(pending / .90)
        if remaining is not None and remaining <= reserve:
            return inherited
        # With a live plan, keep recovering only when stereo still shows a
        # substantial displacement and there is time to complete the stroke.
        if horizon is not None and (error is None or error < .085):
            return inherited
        if error is not None and error <= .035 and motion is not None and motion <= .017 and pending <= .32:
            output = np.zeros_like(inherited)
        else:
            cap = .72 if error is None else float(np.clip(.45 + 3.2 * error, .45, .94))
            output = -np.clip(servo.requested, -cap, cap)
        frame['recovery'] = dict(error=error, reply_frames=remaining, reserve=float(reserve),
                                 command=output.tolist())
        return output


class ReplyServo(BoundedServo):
    """Bound every planned motor rate only near a subsequent strike."""

    later_strike = False

    def _solve_plan(self, normal, rhs):
        if self.later_strike:
            return super()._solve_plan(normal, rhs)
        return PredictiveServo._solve_plan(self, normal, rhs)


class FeedbackServo(PredictiveServo):
    """Apply coherent optical pose residuals when new commands can reach the shot."""

    later_strike = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_sighting = None
        self.coherent_sighting = None
        self.feedback_frames = 0

    def observe(self, actor, pitch):
        super().observe(actor, pitch)
        pose = None if actor is None or not actor.observed else np.asarray(actor.position)
        coherent = (pose is not None and self.last_sighting is not None
                    and np.linalg.norm(pose - self.last_sighting) < .15)
        self.coherent_sighting = pose if coherent else None
        self.last_sighting = pose

    def command(self, goal, lead, pitch):
        if not self.later_strike or self.coherent_sighting is None:
            return super().command(goal, lead, pitch)
        usable_time = self.intercept - self.observer.delay - 1.
        bias = self.coherent_sighting - self.pose
        if usable_time <= 0. or not .02 < np.linalg.norm(bias) < .28:
            return super().command(goal, lead, pitch)
        self.feedback_frames += 1
        original = self.correction[:3].copy()
        self.correction[:3] += min(.75, usable_time / 5.) * np.clip(bias, -.16, .16)
        try:
            return super().command(goal, lead, pitch)
        finally:
            self.correction[:3] = original


class LaterReplyPlayer(FlightCorrectedPlayer):
    """Keep blind neutral recovery and change only a subsequent approach."""

    def __init__(self, *args, mode, **kwargs):
        super().__init__(*args, mode='recenter', **kwargs)

    def reset(self):
        super().reset()
        self.receiving_servo = self.reply_servo(self.receiving_model, self.timing_memory, learn=self.adjust)
        self.subsequent = False

    def _stroke_goals(self, estimate):
        goals = super()._stroke_goals(estimate)
        remaining = goals[-1]
        if remaining is None or self.receive_from is None:
            self.subsequent = False
        elif -50. < self.last_attempt <= self.scene.tick - 12.:
            self.subsequent = True
        servo = self.receiving_servo
        servo.later_strike = self.subsequent and -1.5 < remaining <= servo.horizon + 1.
        return goals


class BoundedReplyPlayer(LaterReplyPlayer):
    reply_servo = ReplyServo

    def finish(self, reward):
        details = super().finish(reward)
        evidence = self.receiving_servo.plan_evidence
        details['bounded_replies'] = {key: evidence[key] for key in ('frames', 'infeasible_frames',
                                                                   'altered_first_moves', 'solver_warning_frames')}
        return details


class FeedbackReplyPlayer(LaterReplyPlayer):
    reply_servo = FeedbackServo

    def finish(self, reward):
        return {**super().finish(reward), 'timed_feedback_frames': self.receiving_servo.feedback_frames}
