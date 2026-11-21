"""Optical rally controller with a prepared blade and a fresh physical search per shot."""

from itertools import product

import numpy as np

from .tt19_factory import CONTACT_OFFSET, FactoryModel, HOME, RATE, moving_joint_rate
from .tt19_flight import optical_start
from .tt19_model_rollout import BallObservation, observed_ball
from .tt19_shoot import COURT, HORIZON
from .tt20_curriculum import CLAMP_SHIFT
from .tt21_body_state import StateInitializedDrySimulator


DEPTHS = (-.815, -.865, -.92, -.975, -1.02)
PITCHES = (.25, .39, .53, .67, .81, .94)
PACES = (.010, .025, .043)
LEADS = (.7, 2.2)
ROBUST = np.array(((-.005, .006, -.004, -.025, -.045, .07, -26., 8., 0.),
                   (.005, -.006, .004, .025, .045, -.07, 26., -8., 0.)))


def face(pitch, yaw):
    return np.array((np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)))


def incoming_crossings(law, point, velocity):
    points, speeds = law.trajectory(point, velocity, HORIZON, bounds=COURT)
    for depth in DEPTHS:
        hits = np.flatnonzero((points[:-1, 1] >= depth) & (points[1:, 1] < depth))
        if len(hits):
            i = int(hits[0])
            fraction = (depth-points[i, 1]) / (points[i+1, 1]-points[i, 1])
            center = (1.-fraction)*points[i] + fraction*points[i+1]
            speed = (1.-fraction)*speeds[i] + fraction*speeds[i+1]
            if 1.4 < i+fraction < HORIZON-2 and .056 < center[2] < .45 and abs(center[0]) < .33:
                yield i+fraction, center, speed


def rates_for(goals, target, low, high):
    drive = np.broadcast_to(target, goals.shape[:1] + target.shape).copy()
    sequence = []
    for desired in np.moveaxis(goals, 1, 0):
        rate = np.clip((np.clip(desired, low, high) - drive) / RATE[:5], -1., 1.)
        drive += rate * RATE[:5]
        sequence.append(rate)
    return np.stack(sequence, axis=1)


def cooperation_cost(result):
    if not result['hit']:
        return result['cost']
    x, y = result['landing']
    margin = result['net']
    return float(result['cost'] + 2.2*abs(x) + 1.05*abs(y-.57)
                 + (.8 if margin is None else 1.15*abs(margin-.285)))


def visual_return(history, shot):
    """Find a court-height change on the outgoing optical arc of an issued stroke."""
    impact = shot['tick'] + shot['prediction']['eta']
    reads = [(t, p) for t, p in history if p is not None and impact-4 <= t <= impact+36]
    for i in range(len(reads)-2):
        (t, a), (u, b), (w, c) = reads[i:i+3]
        if (a[1] < -.43 and max(u-t, w-u) <= 3 and .022 < (b[1]-a[1])/(u-t) < .22
                and .022 < (c[1]-b[1])/(w-u) < .22):
            break
    else:
        return None
    net = None
    for j in range(i+1, len(reads)-1):
        (t, a), (u, b) = reads[j-1:j+1]
        w, c = reads[j+1]
        if max(u-t, w-u) > 3 or b[1]-a[1] < -.02:
            break
        if a[1] <= 0. < b[1]:
            part = -a[1] / (b[1]-a[1])
            net = float(a[2] + part*(b[2]-a[2]))
        if -.43 < b[1] < .81 and b[2] < .093 and a[2]-b[2] > .007 and c[2]-b[2] > .007:
            return dict(landing=[float(b[0]), float(b[1])], net=net, observed_tick=w)
    return None


class ReturnMemory:
    """Correct the predicted landing with earlier own RGB-visible shot outcomes."""

    def __init__(self, evidence=(), *, guard_empty=False):
        self.evidence = list(evidence)
        self.guard_empty = guard_empty

    def bias(self):
        limits = np.array((.060, .21, .060))
        if not self.evidence:
            return np.zeros(3)
        residual = np.array([row['residual'] for row in self.evidence[-18:]], dtype=float)
        valid = np.isfinite(residual)
        return np.sum(np.where(valid, np.clip(residual, -limits, limits), 0.), axis=0) / (2+valid.sum(axis=0))

    def observe(self, shot, reading):
        nominal = shot['prediction']
        if nominal['landing'] is None:
            return
        residual = (np.asarray(reading['landing']) - nominal['landing']).tolist()
        residual.append(None if reading['net'] is None or nominal['net'] is None else reading['net']-nominal['net'])
        self.evidence.append(dict(shot=shot['tick'], optical=reading, predicted=nominal, residual=residual))

    def rank(self, result):
        if (not self.evidence and not self.guard_empty) or not result['hit']:
            return cooperation_cost(result)
        dx, dy, dz = self.bias()
        x, y = result['landing']
        x, y = x+dx, y+dy
        net = result['net']
        net = None if net is None else net+dz
        # Stay well inside the opponent's table. The measured residual often
        # calls for a deeper or better-cleared nominal stroke.
        return float(result['cost'] + 2.2*abs(x) + 1.05*abs(y-.57)
                     + 5.*max(.08-y, y-.775, 0.)
                     + (.8 if net is None else 1.15*abs(net-.285) + 3.*max(.208-net, 0.)))


class RallyPlayer:
    def __init__(self, law, *, pitch_bias=0., pace_scale=1.):
        self.law, self.pitch_bias, self.pace_scale = law, pitch_bias, pace_scale
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        self.sim = StateInitializedDrySimulator(law)
        self.neutral = self.factory.solve(np.array((0., -.911, .166)), face(.48, 0.), HOME)
        self.memory = None
        self.update_memory = False
        self.late_refine = False
        self.reset()

    def reset(self):
        self.sim.twin.reset()
        self.sim.current = None
        self.history = []
        self.plan = None
        self.shots = []
        self.next_plan = 0
        self.outbound = True

    def close(self):
        self.sim.close()

    def rally_yaw(self, center, speed):
        return float(np.clip(-7.*speed[0] - .45*center[0], -.42, .42))

    def lateral_contact(self, estimate, center):
        return center[0]

    def goal_tapes(self, estimate, target):
        goals, parameters = [], []
        timeline = np.arange(HORIZON)
        for eta, center, speed in incoming_crossings(self.law, *estimate):
            yaw = self.rally_yaw(center, speed)
            for pitch in PITCHES:
                normal = face(pitch + self.pitch_bias, yaw)
                anchor = center - CONTACT_OFFSET * normal
                anchor[0] += self.lateral_contact(estimate, center) - center[0]
                pose = self.factory.solve(anchor, normal, target)
                recovered, facing, _, _ = self.factory.pose(pose)
                error = np.linalg.norm(recovered-anchor)
                if error > .057 or facing @ normal < .77:
                    continue
                for pace, lead in product(PACES, LEADS):
                    rate = moving_joint_rate(self.factory, pose, pace*self.pace_scale)
                    phase = np.clip(timeline - eta + lead, -3.1, 3.4)
                    goals.append(pose + phase[:, None]*rate)
                    parameters.append(dict(pitch=pitch + self.pitch_bias, yaw=yaw, pace=pace*self.pace_scale,
                                           eta=eta, depth=float(center[1]), anchor=center.tolist(), lead=lead,
                                           alignment=float(error)))
        if not goals:
            return np.empty((0, HORIZON, 5)), []
        return rates_for(np.asarray(goals), target, self.factory.low, self.factory.high), parameters

    def search(self, estimate, observation, target):
        tapes, params = self.goal_tapes(estimate, target)
        if not len(tapes):
            return None
        lengths = np.clip(np.ceil([p['eta'] for p in params]).astype(int)+3, 3, HORIZON)
        self.sim.current = self.sim.twin.state()
        trials = self.sim.predict(tapes, lengths, observation, np.zeros((len(tapes), 9)))
        rank = cooperation_cost if self.memory is None else self.memory.rank
        score = np.array([rank(row) for row in trials])
        picks = np.argsort(score)[:min(8, len(score))]
        if not trials[picks[0]]['hit']:
            return None
        variants = self.sim.predict(np.repeat(tapes[picks], 2, axis=0), np.repeat(lengths[picks], 2),
                                    observation, np.tile(ROBUST, (len(picks), 1)))
        robust = np.asarray([rank(row) for row in variants]).reshape((-1, 2))
        chosen = int(picks[np.argmin(.65*score[picks] + .25*robust.mean(axis=1) + .1*robust.max(axis=1))])
        row = trials[chosen]
        if not row['legal'] and estimate[0][1] > -.64:
            return None
        return tapes[chosen, :lengths[chosen]], dict(params[chosen], prediction=row, candidate_count=len(tapes))

    def refine_shot(self, tick, estimate, current):
        shot = self.shots[-1]
        position, velocity = estimate
        eta = shot['prediction']['eta']
        if ('late_check' in shot or eta is None or not -.75 < position[1] < -.49 or velocity[2] < .008
                or velocity[1] > -.025 or not 2 <= shot['tick']+eta-tick <= 12):
            return
        observation = observed_ball(self.history, estimate, self.law)
        if not observation.bounced:
            return
        start, tape = self.plan
        left = tape[tick-start:]
        continuation = np.zeros((1, HORIZON, 5))
        continuation[0, :len(left)] = left
        scenarios = np.r_[np.zeros((1, 9)), ROBUST]
        self.sim.current = self.sim.twin.state()
        old = self.sim.predict(np.repeat(continuation, len(scenarios), axis=0), np.full(len(scenarios), len(left)),
                               observation, scenarios)
        legal = sum(row['legal'] for row in old)
        checked = dict(tick=tick, legal_predictions=legal)
        shot['late_check'] = checked
        if legal >= 2:
            return
        chosen = self.search(estimate, observation, current)
        if chosen is None or not chosen[1]['prediction']['legal']:
            return
        tape, details = chosen
        self.plan = (tick, tape)
        self.shots[-1] = dict(tick=tick, **details, late_check=checked, replaced_plan=shot)

    def command(self, tick, position, *, teacher=None):
        """The ordinary call receives only reconstructed RGB and the public clock."""
        self.history.append((tick, position))
        if self.update_memory and self.shots:
            previous = self.shots[-1]
            if 'optical_outcome' not in previous and previous['prediction']['eta'] is not None:
                reading = visual_return(self.history, previous)
                if reading is not None:
                    self.memory.observe(previous, reading)
                    previous['optical_outcome'] = reading
        if position is not None and position[1] > .07:
            self.outbound = True
        estimate = optical_start(self.history, self.law.gravity)
        if teacher is not None:
            estimate = (teacher.center, teacher.velocity/30.)
        current = self.sim.twin.data.ctrl[self.sim.twin.axes[:5]]
        action = np.r_[np.clip((self.neutral-current)/RATE[:5], -1., 1.), 0.]
        if self.plan is not None and tick >= self.plan[0]+len(self.plan[1]):
            self.plan = None
        if self.late_refine and self.plan is not None and estimate is not None and teacher is None:
            self.refine_shot(tick, estimate, current)
        if (self.plan is None and estimate is not None and tick >= self.next_plan
                and self.outbound and -.84 < estimate[0][1] < -.30 and estimate[1][1] < -.025):
            observation = observed_ball(self.history, estimate, self.law) if teacher is None else teacher
            chosen = self.search(estimate, observation, current)
            self.next_plan = tick + 3
            if chosen is not None:
                tape, note = chosen
                self.plan = (tick, tape)
                self.shots.append(dict(tick=tick, **note))
                self.outbound = False
        if self.plan is not None:
            start, tape = self.plan
            action = np.r_[tape[tick-start], 0.]
        self.sim.twin.advance(action)
        return action
