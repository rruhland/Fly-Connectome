"""Adapt service acceleration from first-person departures of the tossed ball."""

from dataclasses import dataclass

import numpy as np

from .tt6_continuous import PredictiveServo
from .tt7_flight import FlightPlayer, outgoing_speed
from .tt8_service_player import falling_crossing


@dataclass
class ServiceDrive:
    rate: float = .132
    samples: int = 0

    def observe(self, speed, progress, reward):
        """Correct persistent short service flights without using bounce labels."""
        if speed is not None:
            error = float(np.clip(.122 - speed, -.027, .034))
            # A slow-looking departure which still wins provides little evidence
            # that further acceleration will help.
            gain = .22 if reward > 0 else .55
            self.rate = float(np.clip(self.rate + gain * error, .105, .184))
            self.samples += 1
        elif progress > .13 and progress < .92 and reward < 0:
            self.rate = min(.184, self.rate + .008)


class ServiceDriveServo(PredictiveServo):
    drive = None

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if self.drive is None or self.intercept is None:
            return
        if not 0. < self.intercept < self.horizon + 1.:
            return
        # Resolve the velocity from two adjacent poses so settling and queued
        # commands contribute to the same time window as newly requested ones.
        motion = np.diff(self.fit.pose(states)[:, 1])
        response = np.einsum('n,tmn->tmn', jac[1], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weight = 2.8 * np.exp(-.5 * ((times - self.intercept) / 1.25)**2)
        rows = (response * weight[:, None, None]).reshape(len(times), -1)
        normal += rows.T @ rows
        rhs += rows.T @ ((self.drive - motion) * weight)


class DrivePlayer(FlightPlayer):
    """Keep a toss-specific stroke through the velocity window near impact."""

    def __init__(self, *args, drive, learn=True, **kwargs):
        self.drive = drive
        self.learn_drive = learn
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return ServiceDriveServo(self.adapter, self.timing_memory, learn=self.adjust)

    def reset(self):
        super().reset()
        self.toss = []
        self.serve_plan = None
        self.serve_rate = self.drive.rate

    def _stroke_goals(self, estimate):
        tick = self.scene.tick
        seen = self.scene.tracker.ball
        self.servo.drive = None
        if (tick <= 18 and seen is not None and seen.observed and seen.position[1] < -.98
                and (self.serve_plan is None or tick < self.serve_plan[0] - 2)):
            self.toss.append((tick, seen.position))
            if (len(self.toss) >= 3 and max(row[1][2] for row in self.toss) > self.toss[0][1][2] + .05
                    and np.ptp([row[1][1] for row in self.toss]) < .11):
                crossing = falling_crossing(self.toss[-9:], .45)
                if crossing is not None and tick + 1 <= crossing[0] <= tick + 22:
                    self.serve_plan = crossing
        if self.serve_plan is None or tick > self.serve_plan[0] + 3:
            return super()._stroke_goals(estimate)

        deadline, xy = self.serve_plan
        horizon = deadline - tick
        pitch = .19
        centre = np.array([*xy, .45]) - .024 * np.array([0., np.cos(pitch), np.sin(pitch)])
        goal = centre.copy()
        goal[1] -= np.clip(self.serve_rate * horizon, -.107, .185)
        lead = centre.copy()
        lead[1] -= np.clip(self.serve_rate * (horizon - 1), -.107, .185)
        self.servo.intercept = horizon
        self.servo.stroke = self.serve_rate
        self.servo.drive = self.serve_rate
        return goal, lead, pitch, horizon

    def finish(self, reward):
        details = super().finish(reward)
        details['serve_drive'] = None
        if self.serve_plan is None:
            return details
        tick, xy = self.serve_plan
        exit_fit = outgoing_speed(self.trace, tick)
        speed = None if exit_fit is None else float(exit_fit[0][0])
        progress = max((row['ball'][1] - xy[1] for row in self.trace
                        if row['ball'] is not None and tick < row['tick'] <= tick + 20), default=0.)
        if self.learn_drive:
            self.drive.observe(speed, progress, reward)
        details['serve_drive'] = dict(tick=float(tick), rate=self.serve_rate, next_rate=self.drive.rate,
                                      optical_speed=speed, progress=float(progress))
        return details
