"""Change the follow-up stroke after recognizing a serve and rival reply in RGB."""

import numpy as np

from .tt5_player import _prediction
from .tt7_contact import LiftTiming
from .tt8_service_drive import DrivePlayer, ServiceDriveServo


class RecoveryServo(ServiceDriveServo):
    follow_up_velocity = None

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if self.follow_up_velocity is None or self.drive is not None:
            return super()._strike_objective(normal, rhs, states, effect, jac, targets)
        if self.intercept is None or not -.5 < self.intercept < self.horizon + 1.:
            return
        displacement = np.diff(self.fit.pose(states)[:, 1:3], axis=0)
        response = np.einsum('an,tmn->tamn', jac[1:3], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weights = 2.8 * np.exp(-.5 * ((times - self.intercept) / 1.25)**2)
        motion = (response * weights[:, None, None, None]).reshape(-1, rhs.size)
        residual = ((self.follow_up_velocity - displacement) * weights[:, None]).ravel()
        normal += motion.T @ motion
        rhs += motion.T @ residual


FOLLOW_UP_MODES = ('velocity', 'lift', 'both', 'loft', 'coast', 'coast-loft', 'early-lift')


class RecoveryPlayer(DrivePlayer):
    def __init__(self, *args, follow_up, **kwargs):
        if follow_up not in FOLLOW_UP_MODES:
            raise ValueError(f'unknown recovery policy {follow_up}')
        self.follow_up = follow_up
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return RecoveryServo(self.adapter, self.timing_memory, learn=self.adjust)

    def reset(self):
        super().reset()
        self.far_sighting = None
        self.reply = None
        self.last_estimate = None
        self.coast_frames = 0
        self.shaped_frames = 0

    def _observe_reply(self):
        if self.serve_plan is None or self.reply is not None:
            return
        tick = self.scene.tick
        ball = self.scene.tracker.ball
        if tick <= self.serve_plan[0] + 3 or ball is None or not ball.observed:
            return
        depth = ball.position[1]
        if self.far_sighting is not None and .0 < depth < self.far_sighting - .14:
            self.reply = tick
        elif depth > .48:
            self.far_sighting = depth if self.far_sighting is None else max(self.far_sighting, depth)

    def _strike_pitch(self, position, incoming):
        pitch = super()._strike_pitch(position, incoming)
        if self.reply is not None and self.follow_up in ('both', 'loft', 'coast-loft'):
            pitch = min(.72, pitch + .12)
        return pitch

    def _stroke_goals(self, estimate):
        self._observe_reply()
        self.servo.follow_up_velocity = None
        tick = self.scene.tick
        coast = self.reply is not None and self.follow_up in ('coast', 'coast-loft', 'early-lift')
        if estimate is not None:
            self.last_estimate = (tick, estimate)
        if coast and estimate is None and self.last_estimate is not None and self.plan is not None:
            then, state = self.last_estimate
            if 0 < tick - then <= 4 and -2.5 < self.plan[0] - tick < 5.:
                position, velocity = state
                for _ in range(tick - then):
                    position, velocity, _ = _prediction(position, velocity, 1., self._surface_ratio())
                if velocity[1] < -.025 and -1.82 < position[1] < -.78:
                    estimate = position, velocity
                    self.coast_frames += 1
        goals = super()._stroke_goals(estimate)
        if self.reply is not None and self.follow_up in ('velocity', 'lift', 'both') and goals[-1] is not None:
            self.servo.follow_up_velocity = np.array([.072, .012] if self.follow_up != 'lift' else [.049, .023])
        goal, lead, pitch, horizon = goals
        if self.reply is not None and self.follow_up == 'early-lift' and horizon is not None and -2. < horizon < 6.5:
            phase = LiftTiming(1.)
            remaining = horizon - self.servo.observer.delay
            goal = goal + (0., 0., phase.offset(remaining))
            lead = lead + (0., 0., phase.offset(remaining - 1.))
            self.shaped_frames += 1
        return goal, lead, pitch, horizon

    def finish(self, reward):
        return dict(super().finish(reward), optical_reply=self.reply,
                    follow_up_coast_frames=self.coast_frames, follow_up_shaped_frames=self.shaped_frames)
