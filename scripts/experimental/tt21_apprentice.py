"""Retarget physically demonstrated SO-101 reply styles using only native RGB."""

import numpy as np

from .tt4_arena import mujoco
from .tt19_factory import CONTACT_OFFSET, FactoryModel, moving_joint_rate
from .tt19_flight import optical_start
from .tt19_model_rollout import observed_ball
from .tt19_shoot import COURT, HORIZON
from .tt20_curriculum import CLAMP_SHIFT
from .tt21_body_state import StateInitializedDrySimulator


GENERIC = ((.20, .020), (.45, .035), (.70, .050))


def style_from_motor(twin, commands, contact):
    """Extract blade posture and speed in a factory twin from requested rates."""
    twin.reset()
    centers = []
    for tick, command in enumerate(commands[:contact]):
        twin.advance(command)
        if tick >= contact - 4:
            mujoco.mj_kinematics(twin.model, twin.data)
            centers.append(twin.data.geom('learner_racket').xpos.copy())
    blade = twin.data.geom('learner_racket')
    normal = blade.xmat.reshape(3, 3)[:, 2].copy()
    normal *= 1. if normal[1] >= 0. else -1.
    pitch = np.arctan2(normal[2], np.linalg.norm(normal[:2]))
    forward = np.median(np.diff(np.asarray(centers), axis=0)[:, 1])
    return [float(np.clip(pitch, .08, .92)), float(np.clip(forward, .008, .064))]


class Apprentice:
    """Use the same visual and nominal-body guard for any small style library."""

    def __init__(self, law):
        self.law = law
        self.sim = StateInitializedDrySimulator(law)
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT

    def reset(self, styles):
        self.styles = styles
        self.sim.twin.reset()
        self.history = []
        self.outgoing = False
        self.attempts = []
        self.applied = None
        self.tape = []
        self.target = None

    def observe(self, tick, point, *, opening):
        self.history.append((tick, point))
        self.outgoing |= bool(opening and point is not None and point[1] > .30)

    def advance(self, command):
        self.sim.twin.advance(command)

    def _goal_tapes(self, state, estimate):
        points, speeds = self.law.trajectory(*estimate, HORIZON, bounds=COURT)
        incoming = np.flatnonzero((points[:, 1] < -.74) & (points[:, 1] > -1.14)
                                  & (np.abs(points[:, 0]) < .42) & (points[:, 2] > .043)
                                  & (points[:, 2] < .46) & (speeds[:, 1] < -.028))
        incoming = incoming[(6 <= incoming) & (incoming <= HORIZON - 1)]
        if not len(incoming):
            return [], [], []
        ticks = np.unique(incoming[np.round(np.linspace(0, len(incoming)-1, min(3, len(incoming)))).astype(int)])
        twin = self.sim.twin
        reference = state.targets[twin.axes[:5]]
        speed_limit = twin.inc[:5] * twin.micro
        candidates, durations, metadata = [], [], []
        clock = np.arange(HORIZON)
        for impact in ticks:
            center, velocity = points[impact], speeds[impact]
            yaw = np.clip(.34 * velocity[0] / max(.014, -velocity[1]) - .35 * center[0], -.24, .24)
            for style, (pitch, pace) in enumerate(self.styles):
                normal = np.array((np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)))
                target = center - CONTACT_OFFSET * normal
                posture = self.factory.solve(target, normal, reference)
                predicted, face, _, _ = self.factory.pose(posture)
                mismatch = np.linalg.norm(predicted - target)
                if mismatch > .115 or face @ normal < .56:
                    continue
                rate = moving_joint_rate(self.factory, posture, pace)
                for lead in (1.6, 3.6):
                    goals = posture + np.clip(clock - impact + lead, -4.4, 4.4)[:, None] * rate
                    request = reference.copy()
                    tape = []
                    for goal in goals:
                        action = np.clip((goal - request)/speed_limit, -1., 1.)
                        tape.append(action)
                        request = np.clip(request + action*speed_limit, self.factory.low, self.factory.high)
                    candidates.append(tape)
                    durations.append(min(HORIZON, int(impact) + 3))
                    metadata.append(dict(style=style, expected_contact=int(impact), lead=lead,
                                         optical_target=center.tolist(), model_alignment_m=float(mismatch)))
        return candidates, durations, metadata

    def _plan(self, estimate):
        state = self.sim.twin.state()
        options, duration, metadata = self._goal_tapes(state, estimate)
        if not options:
            return None, dict(candidates=0, predicted_legal=0)
        observation = observed_ball(self.history, estimate, self.law)
        self.sim.current = state
        try:
            rows = self.sim.predict(np.asarray(options), np.asarray(duration), observation,
                                    np.zeros((len(options), 9)))
        finally:
            self.sim.current = None
        legal = [i for i, row in enumerate(rows) if row['legal']]
        summary = dict(candidates=len(rows), predicted_legal=len(legal))
        if not legal:
            return None, summary
        best = min(legal, key=lambda i: rows[i]['cost'])
        detail = dict(**metadata[best], **rows[best])
        return options[best][:duration[best]], dict(**summary, **detail)

    def command(self, tick, default, *, enabled):
        if self.tape:
            return np.r_[self.tape.pop(0), 0.]
        if self.applied is not None:
            twin = self.sim.twin
            progress = self.target - twin.data.ctrl[twin.axes[:5]]
            return np.r_[np.clip(progress/(twin.inc[:5] * twin.micro), -1., 1.), 0.]
        if not enabled or not self.outgoing or len(self.attempts) >= 4:
            return default
        if self.attempts and tick - self.attempts[-1]['tick'] < 3:
            return default
        estimate = optical_start(self.history, self.law.gravity)
        if estimate is None or estimate[1][1] >= -.032 or not -.57 < estimate[0][1] < .63:
            return default
        tape, detail = self._plan(estimate)
        self.attempts.append(dict(tick=tick, **detail))
        if tape is None:
            return default
        self.applied = tick
        self.tape = tape
        twin = self.sim.twin
        self.target = twin.data.ctrl[twin.axes[:5]].copy()
        return np.r_[self.tape.pop(0), 0.]

    def close(self):
        self.sim.close()
