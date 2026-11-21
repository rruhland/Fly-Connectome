"""Infer a receiving stroke correction from camera-observed departure arcs."""

from dataclasses import dataclass

import numpy as np

from .tt5_player import BALL_RADIUS, DRAG, GRAVITY, NET_CLEARANCE, TABLE_END
from .tt7_flight import nominal_exit, outgoing_speed
from .tt8_service_drive import ServiceDriveServo
from .tt13_rotary_player import RotaryReturnPlayer


FORWARD_MODES = ('pace', 'pace-relative')


def elevation(position, forward, rise, depth):
    """Evaluate the arc at a depth, using drag along the depth direction."""
    rate = DRAG * forward
    time = np.expm1(DRAG * (depth - position[1])) / rate
    decay = np.log1p(rate * time) / rate
    return position[2] + (rise + GRAVITY / (2. * rate)) * decay - GRAVITY / (2. * rate) * time - GRAVITY * time**2 / 4.


def usable_arc(position, forward, rise):
    return ((elevation(position, forward, rise, 0.) >= NET_CLEARANCE + .012)
            & (elevation(position, forward, rise, .22) >= BALL_RADIUS)
            & (elevation(position, forward, rise, TABLE_END - .22) <= BALL_RADIUS))


def increments(position, speed):
    grid = np.linspace(0., .12, 121)
    choices = [usable_arc(position, *(speed[:, None] + axis[:, None] * grid)) for axis in np.eye(2)]
    return [float(grid[np.argmax(ok)]) if ok.any() else None for ok in choices]


@dataclass(frozen=True)
class ReturnCorrection:
    pace: float
    pitch: float


def learn(traces):
    """Estimate the typical required change in departure from optical short contacts."""
    samples = []
    for trace, contact in traces:
        if contact is None:
            continue
        fit = outgoing_speed(trace, contact)
        if fit is None:
            continue
        speed, position, sightings = fit
        # Keep the camera's earlier contact prediction as the reference, even
        # when the departing ball is only reacquired closer to the net.
        adjustment = increments(position, speed)
        samples.append(dict(tick=contact, position=position.tolist(), departure=speed.tolist(),
                            sightings=list(sightings), correction=adjustment,
                            below_clearance=bool(elevation(position, *speed, 0.) < NET_CLEARANCE)))
    short = [sample for sample in samples if sample['below_clearance']]
    if len(short) < 3:
        raise ValueError('at least three optically observed short flights are required')
    feasible = [[sample['correction'][axis] for sample in short if sample['correction'][axis] is not None]
                for axis in range(2)]
    if not all(feasible):
        raise ValueError('the observed short flights yielded no feasible correction along one axis')
    typical = [np.median(offsets) for offsets in feasible]
    # Relate a proposed change in outgoing velocity to a change in the
    # receiving actuator objective, using the standard collision prior.
    incoming = np.array([-.075, .02])
    angle, pace, restitution = .32, .049, .80
    base = nominal_exit(incoming, angle, pace, restitution)
    delta = .01
    forward_gain = (nominal_exit(incoming, angle, pace + delta, restitution)[0] - base[0]) / delta
    pitch_gain = (nominal_exit(incoming, angle + delta, pace, restitution)[1] - base[1]) / delta
    correction = ReturnCorrection(pace=float(np.clip(pace + typical[0] / forward_gain, pace, .092)),
                                  pitch=float(np.clip(typical[1] / pitch_gain, 0., .15)))
    return correction, samples


class FlightCorrectedPlayer(RotaryReturnPlayer):
    """Apply optical calibration exclusively to the receiving stroke."""

    def __init__(self, *args, correction, mode, **kwargs):
        self.correction = correction
        self.mode = mode
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        if self.mode in FORWARD_MODES:
            self.receiving_servo = ServiceDriveServo(self.receiving_model, self.timing_memory, learn=self.adjust)
        self.corrected_frames = 0
        self.last_attempt = -100.

    def _stroke_goals(self, estimate):
        goal, lead, pitch, remaining = super()._stroke_goals(estimate)
        if self.receive_from is None:
            return goal, lead, pitch, remaining
        if self.mode in FORWARD_MODES:
            pace = self.correction.pace
            if self.mode == 'pace-relative':
                pace = min(.112, self.stroke_speed + pace - self.response.default_speed)
            self.servo.drive = pace if remaining is not None else None
        if remaining is not None:
            self.corrected_frames += 1
        return goal, lead, pitch, remaining

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        if self.mode != 'recenter' or self.receive_from is None:
            return command
        frame = self.trace[-1]
        tick, remaining = frame['tick'], frame['horizon']
        if remaining is not None and -1.5 < remaining < 2.5:
            self.last_attempt = tick + remaining
        if remaining is None and 5. <= tick - self.last_attempt <= 24.:
            # A return can leave the blade displaced even as the modeled pose
            # reads ready. Back away from the previous requested actuator
            # excursion before asking for an unrelated next contact.
            command = -np.clip(self.receiving_servo.requested, -.72, .72)
            if np.any(command):
                frame['neutral_request'] = command.tolist()
                self.corrected_frames += 1
        return command

    def _strike_pitch(self, position, incoming):
        pitch = super()._strike_pitch(position, incoming)
        if self.mode != 'loft' or self.receive_from is None:
            return pitch
        chosen = min(.72, pitch + self.correction.pitch)
        self.prediction = incoming[1:].copy(), chosen, self.stroke_speed
        return chosen

    def finish(self, reward):
        details = super().finish(reward)
        details['flight_correction'] = dict(mode=self.mode, shaped_frames=self.corrected_frames)
        return details
