"""Select recurring SO-101 motor strokes from real branch-practice outcomes."""

from itertools import product

import numpy as np

from .tt19_factory import CONTACT_OFFSET, FactoryModel, HOME, RATE, moving_joint_rate
from .tt19_flight import FlightLaw, optical_start
from .tt19_shoot import COURT, HORIZON
from .tt20_curriculum import CLAMP_SHIFT
from .tt22_rally import face, rates_for
from .tt22_servo import relative_lateral

DEPTHS = (-.815, -.865, -.92, -.975)
PITCHES = (.25, .39, .53, .67, .81, .94)
PACES = (.010, .025, .043)
LEADS = (.7, 2.2)
MOTORS = tuple(product(DEPTHS, PITCHES, PACES, LEADS))
FEATURE_SCALE = np.array((.095, .095, .038, .013, .021, .028, .075, .065, .073, 3.))
LATERAL_FRACTION = .24


class MotorCue:
    """Estimate an incoming trajectory and track own commands at the public rate."""

    def __init__(self, coefficients, lateral_fraction=LATERAL_FRACTION):
        self.coefficients = coefficients
        self.lateral_fraction = lateral_fraction
        self.law = FlightLaw()
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        self.neutral = self.factory.solve(np.array((0., -.911, .166)), face(.48, 0.), HOME)
        self.ready_x = self.factory.pose(self.neutral)[0][0]
        self.reset()

    def reset(self):
        self.history = []
        self.target = HOME.copy()
        self.armed = True
        self.previous = 0

    def observe(self, tick, position):
        self.history.append((tick, position))
        estimate = optical_start(self.history, self.law.gravity)
        if estimate is None:
            return None
        point, speed = estimate
        if point[1] > .07 and speed[1] > .020:
            self.armed = True
        if self.armed and -.84 < point[1] < -.30 and speed[1] < -.025:
            return point, speed
        return None

    def consume(self):
        self.armed = False
        self.previous += 1

    def advance(self, command):
        self.target = np.clip(self.target + np.asarray(command[:5])*RATE[:5], self.factory.low, self.factory.high)

    def rest(self):
        return np.r_[np.clip((self.neutral-self.target)/RATE[:5], -1., 1.), 0.]

    def proposals(self, position, velocity):
        """Convert a fixed action grid to motor tapes with inverse kinematics only."""
        path, speed = self.law.trajectory(position, velocity, HORIZON, bounds=COURT)
        blade = self.factory.pose(self.target)[0]
        goals, indices, lengths, alignments, offsets, reference = [], [], [], [], [], None
        timeline = np.arange(HORIZON)
        for depth_index, depth in enumerate(DEPTHS):
            hits = np.flatnonzero((path[:-1, 1] >= depth) & (path[1:, 1] < depth))
            if not len(hits):
                continue
            i = int(hits[0])
            part = (depth-path[i, 1])/(path[i+1, 1]-path[i, 1])
            time = i+part
            center = path[i] + part*(path[i+1]-path[i])
            approach = speed[i] + part*(speed[i+1]-speed[i])
            if depth == -.865:
                reference = (time, center)
            if not (1.4 < time < HORIZON-2 and .056 < center[2] < .45 and abs(center[0]) < .33):
                continue
            yaw = float(np.clip(-7.*approach[0] - .45*center[0], -.42, .42))
            # Blend the signed source relation into aim at the sensed blade corridor.
            source_shift = relative_lateral(self.ready_x, position, velocity, time, self.coefficients)
            requested = self.ready_x + source_shift - center[0]
            correction = requested if self.lateral_fraction == 1. else self.lateral_fraction*np.clip(requested, -.065, .065)
            for pitch_index, pitch in enumerate(PITCHES):
                normal = face(pitch, yaw)
                anchor = center - CONTACT_OFFSET*normal
                anchor[0] += correction
                pose = self.factory.solve(anchor, normal, self.target)
                recovered, facing, _, _ = self.factory.pose(pose)
                error = np.linalg.norm(recovered-anchor)
                if error > .057 or facing @ normal < .77:
                    continue
                for pace_index, pace in enumerate(PACES):
                    rate = moving_joint_rate(self.factory, pose, pace)
                    for lead_index, lead in enumerate(LEADS):
                        phase = np.clip(timeline-time+lead, -3.1, 3.4)
                        goals.append(pose+phase[:, None]*rate)
                        indices.append(np.ravel_multi_index((depth_index, pitch_index, pace_index, lead_index),
                                                            (len(DEPTHS), len(PITCHES), len(PACES), len(LEADS))))
                        lengths.append(min(HORIZON, int(np.ceil(time))+3))
                        alignments.append(error)
                        offsets.append((source_shift, correction))
        if not goals:
            return None
        time, center = (0., position) if reference is None else reference
        features = np.r_[position[0], center[0], center[2], velocity, blade-center, time]
        tapes = rates_for(np.asarray(goals), self.target, self.factory.low, self.factory.high)
        return features, np.asarray(indices), tapes, lengths, np.asarray(alignments), offsets


class OutcomeScorer:
    """Kernel posterior of strict outcomes, with a matched context-free ablation."""

    def __init__(self, corpus, *, contextual=True):
        self.contextual = contextual
        self.context = np.asarray([row['features'] for row in corpus])/FEATURE_SCALE
        self.later = np.array([row['leg'] > 0 for row in corpus])
        self.available = np.zeros((len(corpus), len(MOTORS)))
        self.reward = np.zeros_like(self.available)
        for j, row in enumerate(corpus):
            for trial in row['outcomes']:
                motor, legal, _, landing = trial
                self.available[j, motor] = 1.
                # A softer placement credit applies only after the strict legal bounce.
                if legal:
                    margin = max(0., 1.-abs(landing[0])/.34-abs(landing[1]-.59)/.48)
                    self.reward[j, motor] = 1. + .30*margin
        self.priors = []
        for stage in (False, True):
            selected = self.later == stage
            counts = self.available[selected].sum(axis=0)
            self.priors.append((self.reward[selected].sum(axis=0)+.2)/(counts+2.))

    def ranking(self, features, later, indices, errors):
        prior = self.priors[int(later)][indices]
        score = prior.copy()
        if self.contextual and later:
            distance = np.sum((self.context-features/FEATURE_SCALE)**2, axis=1)
            nearest = self.later == later
            if np.any(nearest):
                distance -= distance[nearest].min()
                weights = np.where(nearest, np.exp(-.5*np.maximum(distance, 0.)), 0.)
                support = weights @ self.available[:, indices]
                observed = weights @ self.reward[:, indices]
                score = (observed+1.6*prior)/(support+1.6)
        return score - .28*errors


class BranchMotor:
    """One visual decision per incoming leg. No physics engine or rollout at play time."""

    def __init__(self, coefficients, scorer, lateral_fraction=LATERAL_FRACTION):
        self.cue = MotorCue(coefficients, lateral_fraction)
        self.scorer = scorer
        self.reset()

    def reset(self):
        self.cue.reset()
        self.tape = ()
        self.start = 0
        self.shots = []

    def command(self, tick, position):
        estimate = self.cue.observe(tick, position)
        if tick-self.start >= len(self.tape) and estimate is not None:
            proposals = self.cue.proposals(*estimate)
            if proposals is not None:
                features, indices, tapes, lengths, errors, offsets = proposals
                later = self.cue.previous > 0
                score = self.scorer.ranking(features, later, indices, errors)
                index = int(np.argmax(score))
                self.start = tick
                self.tape = tapes[index, :lengths[index]]
                source_shift, correction = offsets[index]
                self.shots.append(dict(tick=tick, leg=self.cue.previous, motor=int(indices[index]),
                                       score=float(score[index]), features=features.tolist(),
                                       source_shift=float(source_shift), aim_correction=float(correction),
                                       lateral_travel_limit=abs(source_shift) >= .105))
                self.cue.consume()
        offset = tick-self.start
        command = self.cue.rest() if offset >= len(self.tape) else np.r_[self.tape[offset], 0.]
        self.cue.advance(command)
        return command
