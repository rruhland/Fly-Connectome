"""A recurring RGB-triggered joint pulse with no collision candidate search."""

import numpy as np

from .tt19_factory import FactoryModel, HOME, RATE
from .tt19_flight import FlightLaw, optical_start
from .tt19_shoot import COURT
from .tt20_curriculum import CLAMP_SHIFT


STRIKE_DEPTH = -.835
REFERENCE_HEIGHT = .17


def approach(law, point, speed):
    """Estimate when one observed flight reaches the existing motor stroke."""
    if speed[1] >= -.024 or not STRIKE_DEPTH < point[1] < .24:
        return None
    positions, _ = law.trajectory(point, speed, 22, bounds=COURT)
    found = np.flatnonzero(positions[1:, 1] < STRIKE_DEPTH)
    if not len(found):
        return None
    i = int(found[0])
    a, b = positions[i:i+2]
    share = (STRIKE_DEPTH - a[1]) / (b[1] - a[1])
    return i+share, a + share*(b-a)


def relative_lateral(ready_x, seen, velocity, time, coefficients):
    """Scale relative displacement and lateral motion by the supplied gains."""
    displacement, motion = coefficients
    shift = displacement * (seen[0]-ready_x) + motion * time * velocity[0]
    return float(np.clip(shift, -.105, .105))


def lift_face(normal, amount):
    horizontal = np.linalg.norm(normal[:2])
    pitch = np.arctan2(normal[2], horizontal) + amount
    return np.r_[normal[:2]*np.cos(pitch)/horizontal, np.sin(pitch)]


class RecurrentServo:
    """Prepare two joint keyframes. The incoming visual phase starts each pulse."""

    def __init__(self, original, coefficients, *, lead, amplitude, loft=0., height_gain=.88, continuous_phase=False):
        self.original = np.asarray(original, dtype=float)
        self.coefficients = coefficients
        self.lead, self.amplitude = lead, amplitude
        self.law = FlightLaw()
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        self.neutral = HOME + RATE[:5]*self.original[:5]
        self.peak = self.neutral + amplitude * RATE[:5]*self.original[5:10]
        self.keyframes = tuple(self.factory.pose(q)[:2] for q in (self.neutral, self.peak))
        if loft:
            self.keyframes = tuple((p, lift_face(n, loft)) for p, n in self.keyframes)
            self.neutral, self.peak = (self.factory.solve(p, n, q)
                                       for (p, n), q in zip(self.keyframes, (self.neutral, self.peak), strict=True))
        self.height_gain = height_gain
        self.continuous_phase = continuous_phase
        self.rise, self.fall = self.original[11], 5.
        self.reset()

    def reset(self):
        self.history = []
        self.joints = HOME.copy()
        self.ready, self.swing = self.neutral.copy(), self.peak.copy()
        self.displacement = np.zeros(3)
        self.last_update = -10
        self.launch = None
        self.armed = True
        self.pulses = []

    def translate(self, shift, tick):
        if tick-self.last_update < 2 or np.linalg.norm(shift-self.displacement) < .011:
            return
        a, b = (self.factory.solve(pos+shift, normal, initial)
                for (pos, normal), initial in zip(self.keyframes, (self.ready, self.swing), strict=True))
        self.ready, self.swing = a, b
        self.displacement, self.last_update = shift, tick

    def command(self, tick, position):
        self.history.append((tick, position))
        seen = optical_start(self.history, self.law.gravity)
        if seen is not None:
            point, velocity = seen
            if self.launch is not None and tick-self.launch >= 5 and point[1] > .065 and velocity[1] > .023:
                self.launch = None
                self.armed = True
            upcoming = approach(self.law, point, velocity) if self.armed else None
            if upcoming is not None:
                time, intercept = upcoming
                if .065 < intercept[2] < .34:
                    dx = relative_lateral(self.keyframes[0][0][0], point, velocity, time, self.coefficients)
                    dz = float(np.clip(self.height_gain*(intercept[2]-REFERENCE_HEIGHT), -.076, .076))
                    self.translate(np.array((dx, 0., dz)), tick)
                    if time <= self.lead:
                        self.armed = False
                        self.launch = tick + time - self.lead if self.continuous_phase else tick
                        self.pulses.append(dict(tick=tick, time_to_plane=float(time), incoming=point.tolist(),
                                                 velocity=velocity.tolist(), intercept=intercept.tolist(),
                                                 applied_shift=self.displacement.tolist(),
                                                 ready=self.ready.tolist(), swing=self.swing.tolist(),
                                                 phase_origin=float(self.launch)))
        target = self.ready
        if self.launch is not None:
            age = tick-self.launch
            phase = min(1., age/self.rise, max(0., 1.-(age-self.rise)/self.fall))
            target = target + phase*(self.swing-target)
        rates = np.clip((target-self.joints)/RATE[:5], -1., 1.)
        self.joints += rates*RATE[:5]
        return np.r_[rates, 0.]
