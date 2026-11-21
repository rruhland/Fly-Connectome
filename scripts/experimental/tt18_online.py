"""Acquire a camera body model through the commands of scored table play."""

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tt6_body_control import MOTOR_BUDGET, POSE_WEIGHT, CameraBody, _features, _response_bank
from .tt6_memory import _serializable_body
from .tt8_service_drive import ServiceDriveServo
from .tt8_service_search import optical_credit
from .tt9_precontact_registration import RegistrationPlayer


def carry_body(body, transform):
    """Express a related arm's quadratic pose fit in its public port coordinates."""
    basis = np.eye(body.action_size)
    positive = body.pose(basis @ transform.T)
    negative = body.pose(-basis @ transform.T)
    left, right = np.triu_indices(body.action_size)
    cross = body.pose((basis[left] + basis[right]) @ transform.T) - positive[left] - positive[right] + body.home
    cross[left == right] *= .5
    strongest = np.argmax(abs(transform), axis=0)
    return replace(body, coefficients=np.vstack((body.home, .5 * (positive - negative), cross)),
                   lags=body.lags[strongest], kernels=body.kernels[strongest])


def blank_body(width):
    """Use a generic table-ready pose and settling times, with zero motor gains."""
    curves, delays = _response_bank()
    selected = np.argmin(abs(delays - .50))
    coefficients = np.zeros((1 + width + width * (width + 1) // 2, 4))
    coefficients[0] = (0., -1.48, .22, .25)
    return CameraBody(coefficients, np.full(width, delays[selected]), np.tile(curves[selected], (width, 1)),
                      0., 0, 0, False)


def restore_body(payload):
    arrays = {name: np.asarray(payload[name]) for name in ('coefficients', 'lags', 'kernels')}
    return CameraBody(**dict(payload, **arrays))


class BodyLearning:
    """Regularize a causal command-to-pose fit towards the carried body map."""

    def __init__(self, prior, *, enabled, seed, directed=True):
        self.prior = self.model = prior
        self.enabled = enabled
        self.directed = directed
        self.rng = np.random.default_rng(seed)
        self.width = prior.action_size
        size = len(prior.coefficients)
        self.penalties = np.r_[.65, np.full(self.width, .8), np.full(size - self.width - 1, 18.)]
        self.gram = np.repeat(np.diag(self.penalties)[None], 2, axis=0)
        self.cross = self.penalties[:, None] * prior.coefficients
        self.sightings = self.pitches = self.updates = 0
        self.uncertainty = np.full(self.width, 1 / self.penalties[1])
        self.move = np.zeros(self.width)
        self.interval = 0

    def begin_point(self):
        self.move.fill(0.)
        self.interval = 0

    def observe(self, state, pose):
        """Score the prediction made before fitting the current optical sample."""
        predicted = self.model.pose(state)
        error = float(np.linalg.norm(predicted[:3] - pose[:3]))
        if not self.enabled:
            return error
        feature = _features(state)
        # Long excursions near the workspace boundary are less compatible with a
        # local quadratic map than nearby observations of the same direction.
        weight = 1 / np.sqrt(1 + (error / .095)**2)
        moment = weight * np.outer(feature, feature)
        self.gram[0] += moment
        self.cross[:, :3] += weight * feature[:, None] * pose[:3]
        self.sightings += 1
        if np.isfinite(pose[3]):
            pitch_weight = 1 / np.sqrt(1 + ((predicted[3] - pose[3]) / .34)**2)
            self.gram[1] += pitch_weight * np.outer(feature, feature)
            self.cross[:, 3] += pitch_weight * feature * pose[3]
            self.pitches += 1
        if self.sightings % 6 == 0:
            self.model = self.fit(self.cross)
            marginal = np.linalg.inv(self.gram[0])
            self.uncertainty = marginal.diagonal()[1:1 + self.width].copy()
            self.updates += 1
        return error

    def fit(self, cross):
        coefficients = np.column_stack((np.linalg.solve(self.gram[0], cross[:, :3]),
                                        np.linalg.solve(self.gram[1], cross[:, 3])))
        active = np.max(abs(coefficients[1:1 + self.width, 3])) >= .035
        return replace(self.model, coefficients=coefficients, samples=self.sightings, pitch_samples=self.pitches,
                       pitch_controllable=bool(active))

    def explore(self, command, position, deadline):
        """Choose a small rate correction from uncertainty in the current play."""
        if not self.enabled or not self.directed:
            return command
        rank = np.linalg.svd(self.model.jacobian(position)[:3], compute_uv=False)[-1]
        confidence = np.clip(rank / .045, 0., 1.)
        far_from_strike = deadline is None or deadline > 8 or deadline < -3
        strength = min(.36, .14 * np.sqrt(np.max(self.uncertainty))) + .33 * (1 - confidence)
        if not far_from_strike:
            strength *= .10 + .40 * (1 - confidence)
        if self.interval == 0:
            weights = self.uncertainty * np.maximum(.1, 1. - np.abs(command))
            index = self.rng.choice(self.width, p=weights / weights.sum())
            self.move.fill(0.)
            self.move[index] = self.rng.choice((-1., 1.))
        self.interval = (self.interval + 1) % 3
        lower = np.maximum(-1., -MOTOR_BUDGET - position)
        upper = np.minimum(1., MOTOR_BUDGET - position)
        return np.clip(command + strength * self.move, lower, upper)

    def state(self):
        return dict(enabled=self.enabled, directed=self.directed,
                    prior=_serializable_body(self.prior), model=_serializable_body(self.model),
                    gram=self.gram.tolist(), cross=self.cross.tolist(), sightings=self.sightings,
                    pitches=self.pitches, updates=self.updates, uncertainty=self.uncertainty.tolist(),
                    rng=self.rng.bit_generator.state)

    @classmethod
    def restore(cls, payload):
        learner = cls(restore_body(payload['prior']), enabled=payload['enabled'], seed=0,
                      directed=payload['directed'])
        learner.model = restore_body(payload['model'])
        for name in ('gram', 'cross', 'uncertainty'):
            setattr(learner, name, np.asarray(payload[name]))
        for name in ('sightings', 'pitches', 'updates'):
            setattr(learner, name, payload[name])
        learner.rng.bit_generator.state = payload['rng']
        return learner


class LearningServo(ServiceDriveServo):
    def __init__(self, body, timing, *, learn):
        super().__init__(body.model, timing, learn=learn)
        self.body = body
        self.body.begin_point()
        self.last_error = None
        self.last_dither = 0.

    def observe(self, actor, pitch):
        super().observe(actor, pitch)
        self.last_error = None
        if actor is None or not actor.observed:
            return
        pose = np.r_[actor.position, np.nan if pitch is None else pitch]
        previous = self.body.model
        self.last_error = self.body.observe(self.effective, pose)
        if previous is not self.body.model:
            self.fit = self.body.model
            new_prediction = self.fit.pose(self.effective)
            # A model update already explains some of the observed deviation.
            self.correction = np.clip(np.nan_to_num(pose - new_prediction), -.07, .07)

    def command(self, goal, lead, pitch):
        planned = super().command(goal, lead, pitch)
        action = self.body.explore(planned, self.requested, self.intercept)
        self.last_dither = float(np.linalg.norm(action - planned, ord=1))
        return action


class OnlineBodyPlayer(RegistrationPlayer):
    def _make_servo(self):
        return LearningServo(self.body_learning, self.timing_memory, learn=self.adjust)

    def update(self, frames, preceding=None):
        action = super().update(frames, preceding)
        self.adapter = self.servo.fit
        self.trace[-1].update(body_prediction_error=self.servo.last_error, dither=self.servo.last_dither)
        return action


def signed_basis(source, observed):
    """Match axes by the optical displacement each column predicts."""
    width = source.action_size
    home = np.zeros(width)
    reference = POSE_WEIGHT[:, None] * source.jacobian(home)
    current = POSE_WEIGHT[:, None] * observed.jacobian(home)
    reference /= np.maximum(.025, np.linalg.norm(reference, axis=0))
    current /= np.maximum(.025, np.linalg.norm(current, axis=0))
    agreement = reference.T @ current
    old, new = linear_sum_assignment(-np.abs(agreement))
    basis = np.zeros((width, width))
    basis[old, new] = np.where(agreement[old, new] < 0., -1., 1.)
    return basis


class RecallLearning(BodyLearning):
    """Propose a different carried basis and verify it on the next scored point."""

    def __init__(self, prior, *, enabled, seed, directed=True):
        super().__init__(prior, enabled=enabled, seed=seed, directed=directed)
        self.source = prior
        self.pending = None
        self.pending_model = None
        self.reference_model = None
        self.paired_errors = []
        self.decision = None

    def begin_point(self):
        super().begin_point()
        self.reference_model = self.model
        self.paired_errors = []

    def observe(self, state, pose):
        if self.pending_model is not None:
            target = pose[:3]
            self.paired_errors.append([float(np.linalg.norm(model.pose(state)[:3] - target))
                                       for model in (self.reference_model, self.pending_model)])
        return super().observe(state, pose)

    def _shifted(self, proposed):
        return self.cross + self.penalties[:, None] * (proposed.coefficients - self.prior.coefficients)

    def end_point(self):
        self.decision = dict(adopted=False, pair_count=len(self.paired_errors))
        if self.pending is not None and len(self.paired_errors) >= 12:
            before, after = np.median(self.paired_errors, axis=0)
            self.decision.update(before=float(before), after=float(after))
            if after + .003 < .93 * before:
                self.cross = self._shifted(self.pending)
                self.prior = self.pending
                self.model = self.fit(self.cross)
                self.decision['adopted'] = True
        self.pending = self.pending_model = None
        if self.sightings >= 20:
            basis = signed_basis(self.source, self.model)
            proposed = carry_body(self.source, basis)
            # Input features retain their original public-port settling curves.
            proposed = replace(proposed, lags=self.model.lags, kernels=self.model.kernels)
            if not np.allclose(self.prior.coefficients, proposed.coefficients):
                self.pending = proposed
                self.pending_model = self.fit(self._shifted(proposed))
                self.decision['proposed_basis'] = basis.tolist()

    def state(self):
        return dict(super().state(), source=_serializable_body(self.source),
                    pending=None if self.pending is None else _serializable_body(self.pending),
                    pending_model=None if self.pending_model is None else _serializable_body(self.pending_model))

    @classmethod
    def restore(cls, payload):
        learner = super().restore(payload)
        learner.source = restore_body(payload['source'])
        for name in ('pending', 'pending_model'):
            setattr(learner, name, None if payload[name] is None else restore_body(payload[name]))
        return learner


SERVE_OFFSETS = (
    (0., 0., 0.),
    (-.045, 0., 0.),
    (.045, 0., 0.),
    (0., -.09, 0.),
    (0., .10, 0.),
    (0., 0., -.026),
)


class ServeChoices:
    """Choose a variation using stereo launch evidence and the point reward."""

    def __init__(self, observations=()):
        self.observations = list(observations)

    def choose(self):
        counts = np.zeros(len(SERVE_OFFSETS))
        totals = np.zeros_like(counts)
        for row in self.observations:
            counts[row['choice']] += 1
            totals[row['choice']] += row['score']
        baseline = np.array([.21, .13, .13, .13, .13, .13])
        means = (totals + 1.4 * baseline) / (1.4 + counts)
        bonus = .19 * np.sqrt(np.log(2 + counts.sum()) / (1. + counts))
        return int(np.argmax(means + bonus))


class OnlineServePlayer(OnlineBodyPlayer):
    """Keep the learned arm controller and test serve variations inside games."""

    def reset(self):
        super().reset()
        self.serve_height = .45
        self.serve_pitch = .19
        self.serve_choice = None

    def _stroke_goals(self, estimate):
        if self.serve_choice is None and self.serve_plan is not None:
            self.serve_choice = self.serve_choices.choose()
            height, pitch, drive = SERVE_OFFSETS[self.serve_choice]
            self.serve_height += height
            self.serve_pitch += pitch
            self.serve_rate += drive
            if height:
                self.serve_plan = None
        return super()._stroke_goals(estimate)

    def finish(self, reward):
        outcome = super().finish(reward)
        outcome['serve_exploration'] = None
        if self.serve_choice is not None:
            credit = dict(score=.18 * max(0., reward), reward=reward)
            if self.serve_plan is not None:
                deadline, xy = self.serve_plan
                credit = optical_credit(self.trace, deadline, xy[1], reward)
            observed = dict(choice=self.serve_choice, **credit)
            self.serve_choices.observations.append(observed)
            outcome['serve_exploration'] = observed
        return outcome
