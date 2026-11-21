"""Camera-only control on a four-axis table, with portable Pong and mini priors."""

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .tt3_visual import RoleModel, _SceneHistory, _fit_ranking, _scaled


ACTIONS = 4
ZERO = np.zeros(ACTIONS)
POWERS = np.array([
    [term.count(axis) for axis in range(ACTIONS)]
    for degree in range(4)
    for term in combinations_with_replacement(range(ACTIONS), degree)
])
DEGREES = POWERS.sum(axis=1)
DERIVATIVE_POWERS = np.maximum(POWERS[:, None, :] - np.eye(ACTIONS, dtype=int), 0)
ARM_RIDGE = np.diag(np.choose(DEGREES, [.015, .025, .18, .80]))


def _basis(coordinates):
    return np.power(np.asarray(coordinates)[..., None, :], POWERS).prod(axis=-1)


def _bound(value, maximum=1.):
    return value * min(1., maximum / max(np.abs(value).max(), 1e-8))


@dataclass(frozen=True)
class CalibrationSample:
    """Actual commands and anonymous positions reconstructed from both eyes."""

    action: np.ndarray
    positions: dict
    appearances: dict


def _body_responses(samples):
    """Collect candidate components and their response to one probe tape."""
    commands = np.array([sample.action for sample in samples])
    coordinates = np.cumsum(commands, axis=0)
    identities = {identity for sample in samples for identity in sample.positions}
    candidates = []
    for identity in identities:
        indices = [i for i, sample in enumerate(samples) if identity in sample.positions]
        if len(indices) < max(8, .45 * len(samples)) or indices[0] > 2:
            continue
        t = np.array(indices, dtype=float)
        times = (t - t.mean()) / max(1., np.ptp(t))
        motions = np.array([samples[i].positions[identity] for i in indices])
        # A ballistic or camera-drift trajectory can be well fit by time alone.
        # Regress the remaining motion on signed action coordinates.
        drift, _ = np.linalg.qr(np.column_stack((np.ones(len(t)), times, times**2)))
        motion = motions - drift @ (drift.T @ motions)
        independent = coordinates[indices] - drift @ (drift.T @ coordinates[indices])
        variance = np.square(motion).sum()
        coefficient = np.linalg.solve(independent.T @ independent + .3 * np.eye(ACTIONS),
                                      independent.T @ motion)
        predicted = independent @ coefficient
        contrast = (variance - np.square(motion - predicted).sum()) / max(variance, .02)
        # Depth from stereo is less certain on far objects. A bound on motion
        # per command downweights chance fits to poorly triangulated tracks.
        score = contrast * min(1., .15 / max(1e-8, np.linalg.norm(coefficient)))
        if score > .015:
            hues = [samples[i].appearances[identity] for i in indices
                    if identity in samples[i].appearances]
            candidates.append((score, commands, np.array(indices), motions, hues))
    return candidates


class ArmAdapter:
    """A shared RGB and signed-action fit with no knowledge of the motor wiring."""

    def __init__(self, trajectories):
        by_trajectory = [_body_responses(rows) for rows in trajectories]
        groups = []
        for rows in by_trajectory:
            for candidate in rows:
                anchor = candidate[3][0]
                group = [max((row for row in other if np.linalg.norm(row[3][0] - anchor) < .23),
                             key=lambda row: row[0], default=None) for other in by_trajectory]
                group = [row for row in group if row is not None]
                if len(group) >= 2:
                    groups.append(group)
        accepted = max(groups, key=self._agreement, default=[])
        if len(accepted) < 2:
            raise ValueError('fewer than two probe points identified an action-responsive track')
        starts = np.stack([positions[0] for _, _, _, positions, _ in accepted])
        center = np.median(starts, axis=0)
        accepted = [row for row in accepted if np.linalg.norm(row[3][0] - center) < .24]
        if len(accepted) < 2:
            raise ValueError('probe points disagree on the ready pose')
        self.ready = np.median(np.stack([row[3][0] for row in accepted]), axis=0)
        self.colors = Counter(hue for row in accepted for hue in row[-1])
        self.hue = self.colors.most_common(1)[0][0] if self.colors else None
        self.lower = np.quantile(np.concatenate([q for _, q in self._coordinates(accepted, .4)]), .005, axis=0)
        self.upper = np.quantile(np.concatenate([q for _, q in self._coordinates(accepted, .4)]), .995, axis=0)
        self.fit, self.lag, self.error = self._fit(accepted)
        self.accepted = len(accepted)
        self.samples = sum(len(row[3]) for row in accepted)
        self.rank = int(np.linalg.matrix_rank(self.jacobian(ZERO)))
        direction = np.linalg.svd(self.jacobian(ZERO))[2][-1]
        self.wrist_direction = direction * np.sign(direction[np.argmax(np.abs(direction))])

    @classmethod
    def _agreement(cls, group):
        poses, coordinates = zip(*cls._coordinates(group, .62), strict=True)
        actions = np.concatenate([q - q.mean(axis=0) for q in coordinates])
        outcomes = np.concatenate([p - p.mean(axis=0) for p in poses])
        response = np.linalg.solve(actions.T @ actions + .10 * np.eye(ACTIONS), actions.T @ outcomes)
        residual = outcomes - actions @ response
        prediction = 1. - np.square(residual).sum() / max(.03, np.square(outcomes).sum())
        singular = np.linalg.svd(response, compute_uv=False)
        # A repeated response on several independent axes distinguishes the
        # controllable component from a fast uncommanded track.
        return len(group) * max(0., prediction) * min(.40, singular[0]) * min(.30, singular[1])

    @staticmethod
    def _coordinates(accepted, lag):
        for _, actions, indices, positions, _ in accepted:
            requested = np.cumsum(actions, axis=0) / 4.
            effective = np.zeros_like(requested)
            for i in range(1, len(requested)):
                effective[i] = lag * effective[i - 1] + (1. - lag) * requested[i]
            yield positions, effective[indices]

    def _fit(self, accepted):
        best = None
        for lag in (0., .25, .45, .60, .72, .81):
            poses, coordinates = zip(*self._coordinates(accepted, lag), strict=True)
            design = _basis(np.concatenate(coordinates))
            observed = np.concatenate(poses) - self.ready
            weights = np.ones(len(design))
            for _ in range(3):
                weighted = design * weights[:, None]
                coefficients = np.linalg.solve(
                    design.T @ weighted + ARM_RIDGE, weighted.T @ observed
                )
                residual = np.linalg.norm(design @ coefficients - observed, axis=1)
                weights = 1. / (1. + np.square(residual / .045))
            error = float(np.median(residual))
            if best is None or error < best[-1]:
                best = coefficients, lag, error
        return best

    def pose(self, coordinates):
        return self.ready + _basis(coordinates) @ self.fit

    def jacobian(self, coordinates):
        slopes = np.power(coordinates, DERIVATIVE_POWERS).prod(axis=-1) * POWERS
        return self.fit.T @ slopes

    def stats(self):
        return dict(ready=self.ready.tolist(), lag=self.lag, median_fit_error=self.error,
                    accepted_points=self.accepted, motor_samples=self.samples, rank=self.rank,
                    ready_jacobian=self.jacobian(ZERO).tolist(), actor_hue=self.hue,
                    wrist_direction=self.wrist_direction.tolist(),
                    command_coordinates_low=self.lower.tolist(),
                    command_coordinates_high=self.upper.tolist())

    def command(self, requested, effective, position, goal, *, wrist=0.):
        """Solve a bounded visual position correction using the learned response."""
        basis = self.jacobian(effective)
        signal = self.pose(effective) if position is None else position
        residual = goal - signal
        inverse = np.linalg.solve(basis @ basis.T + .035 * np.eye(3), basis)
        change = inverse.T @ residual
        # Suppress drift in directions the camera did not identify.
        shadow = wrist * self.wrist_direction - requested
        change += .35 * (shadow - inverse.T @ (basis @ shadow))
        lead = 1. - self.lag
        action = _bound(4. * (change - .48 * (requested - effective)) / max(.58, lead))
        lower = np.minimum(-.65, self.lower - .15)
        upper = np.maximum(.65, self.upper + .15)
        bounded = np.clip(requested + action / 4., lower, upper)
        return np.clip(4. * (bounded - requested), -1., 1.)


def calibration_commands(seed, horizon):
    """Return a nonadaptive, repeatable tape with paired signed perturbations."""
    rng = np.random.default_rng(seed)
    commands = []
    # Zero-mean oscillations exercise all axes; the return leg provides an
    # independent check on the relation between color-track drift and actions.
    blocks = int(np.ceil(horizon / 4)) + 1
    position = np.zeros(ACTIONS)
    for k in range(blocks):
        goal = np.zeros(ACTIONS) if k % 3 == 2 else rng.uniform(-1., 1., ACTIONS) * 3.2
        action = np.clip((goal - position) / 4., -1., 1.)
        commands.extend([action] * 4)
        position += 4. * action
    return np.array(commands[:horizon])


def _native_numeric(state):
    return (np.asarray(state['coefficients'], dtype=float),
            np.asarray(state['totals'], dtype=float),
            np.asarray(state['counts'], dtype=int))


def _delay_state(state=None, *, reverse=False):
    if state is None:
        return (np.array([-3.8, 0., 3.8]) / 2., np.zeros(3), np.zeros(3), np.zeros(3))
    order = slice(None, None, -1) if reverse else slice(None)
    return (np.asarray(state['delays']) / 2., np.asarray(state['uses'], dtype=float)[order].copy(),
            np.asarray(state['continuations'], dtype=float)[order].copy(),
            np.asarray(state['public_scores'], dtype=float)[order].copy())


class TransferPrior:
    """Numeric, timing and visual evidence the same policy may bring to the table."""

    def __init__(self, condition, native, seed, *, mini=None):
        self.rng = np.random.default_rng(seed)
        self.clock = None
        self.span_memory = None
        self.bounce_values = []
        if condition == 'fresh':
            scratch = np.stack([(self.rng.permutation(4) + self.rng.random(4)) / 2. - 1.
                                for _ in range(3)], axis=1) * [.11, 1.4, 1.4]
            self.numeric = scratch, np.zeros(4), np.zeros(4, dtype=int)
            self.roles = RoleModel.untrained()
            self.delay = _delay_state()
        elif condition == 'pong_only':
            self.numeric = _native_numeric(native['numeric'])
            self.roles = RoleModel(**native['roles'])
            self.clock = native['clock']
            self.delay = _delay_state()
        else:
            if mini is None:
                raise ValueError(f'{condition} requires a mini checkpoint')
            self.numeric = _native_numeric(mini['memory']['shared'])
            self.roles = RoleModel(**mini['roles'])
            self.clock = mini['clock']
            self.delay = _delay_state(mini['timing'], reverse=condition == 'mini_shuffled_timing')
            # Old locations and accelerations depend on the old camera and time
            # units. Only a dimensionless estimate of bounce speed survives.
            surfaces = [ratios for axis, sign, ratios in mini['surfaces'] if axis == 2 and sign == 1]
            ratios = [ratio for rows in surfaces for _, ratio in rows]
            if ratios:
                self.bounce_values = [float(np.median(ratios))] * 3

    @property
    def restitution(self):
        return float(np.median(self.bounce_values)) if self.bounce_values else .70

    def choose(self):
        if self.span_memory is None:
            _, scores, counts = self.numeric
            means = np.divide(scores, np.maximum(1, counts))
            selection = int(self.rng.choice(np.flatnonzero(np.isclose(means, means.max()))))
        else:
            selection = self.span_memory.choose(self.rng)
        delays, uses, continuations, totals = self.delay
        unseen = np.flatnonzero(uses == 0)
        if len(unseen):
            distance = np.abs(delays[unseen])
            index = int(self.rng.choice(unseen[np.isclose(distance, distance.min())]))
        else:
            estimates = (continuations + .20 * totals) / (uses + 2.)
            indices = np.flatnonzero(np.isclose(estimates, estimates.max()))
            index = int(indices[np.argmin(np.abs(delays[indices]))])
        return selection, index

    def observe(self, option, delay, reward, *, opportunity, outgoing):
        if not opportunity:
            return
        if self.span_memory is None:
            _, scores, counts = self.numeric
            scores[option] += reward
            counts[option] += 1
        else:
            self.span_memory.observe(option, reward)
        _, uses, continuations, totals = self.delay
        uses[delay] += 1
        continuations[delay] += int(outgoing)
        totals[delay] += reward

    def state(self):
        coefficients, totals, counts = self.numeric
        delays, uses, continuations, score = self.delay
        return dict(numeric=dict(coefficients=coefficients.tolist(), totals=totals.tolist(),
                                 counts=counts.tolist()),
                    timing=dict(delays=delays.tolist(), uses=uses.tolist(),
                                continuations=continuations.tolist(), public_scores=score.tolist()),
                    restitution=self.restitution, bounce_evidence=len(self.bounce_values),
                    role_weights=self.roles.relevance_weights)


class Flight:
    """Estimate a ballistic segment and visible rebounds from 3D camera samples."""

    def __init__(self):
        self.history = deque(maxlen=10)
        self.identity = None
        self.previous = None
        self.acceleration = -.009
        self.gravity = deque(maxlen=40)

    def update(self, track, tick, restitution_samples):
        if track is None or not track.observed:
            if self.previous is not None and tick - self.previous[0] <= 5:
                time, position, velocity = self.previous
                duration = tick - time
                position = position + velocity * duration
                position[2] += .5 * self.acceleration * duration**2
                velocity = velocity.copy()
                velocity[2] += self.acceleration * duration
                return position, velocity
            return None
        point = np.array(track.position)
        if track.id != self.identity or (self.history and tick - self.history[-1][0] > 2):
            self.history.clear()
            self.previous = None
            self.identity = track.id
        if self.previous is not None:
            old_tick, pos, speed = self.previous
            span = tick - old_tick
            predicted = pos + speed * span + np.array([0., 0., .5 * self.acceleration * span**2])
            residual = point - predicted
            bounce = residual[2] > .044 and speed[2] < -.04 and min(point[2], pos[2]) < .17
            direction_change = abs(residual[1]) > .10 and np.sign(point[1] - pos[1]) != np.sign(speed[1])
            if bounce or direction_change:
                if bounce and point[2] > pos[2]:
                    height = pos[2] - .020
                    to_surface = np.clip(height / max(.03, -speed[2]), 0., span - .15)
                    apparent = (point[2] - .020) / max(.025, -speed[2] * (span - to_surface))
                    if .3 < apparent < 1.2:
                        restitution_samples.append(float(apparent))
                self.history.clear()
                if direction_change:
                    self.history.append((old_tick, pos))
                else:
                    incoming = -speed[2] * (np.median(restitution_samples) if restitution_samples else .75)
                    seed = point - np.array([speed[0], speed[1], incoming])
                    self.history.extend(((tick - 2, 2 * seed - point), (tick - 1, seed)))
        self.history.append((tick, point))
        if len(self.history) < 3:
            self.previous = None
            return None
        times = np.array([t - tick for t, _ in self.history], dtype=float)
        values = np.array([p for _, p in self.history])
        linear = np.column_stack((np.ones(len(times)), times))
        if len(times) >= 6 and tick % 2 == 0:
            quad = np.column_stack((linear, .5 * times**2))
            fit = np.linalg.lstsq(quad, values, rcond=None)[0]
            inferred = float(fit[2, 2])
            if -.026 < inferred < -.002 and np.max(np.abs(quad @ fit - values)) < .075:
                self.gravity.append(inferred)
                # A broad physical prior stabilizes short and noisy camera segments.
                self.acceleration = float(np.median([-.009] * 5 + list(self.gravity)))
        values[:, 2] -= .5 * self.acceleration * times**2
        fitted, velocity = np.linalg.lstsq(linear, values, rcond=None)[0]
        self.previous = (tick, fitted, velocity)
        return fitted, velocity

    def project(self, position, velocity, horizon, restitution):
        strike = position + velocity * horizon
        g = self.acceleration
        strike[2] += .5 * g * horizon**2
        tabletop = .025
        discriminant = velocity[2]**2 - 2 * g * (position[2] - tabletop)
        if discriminant >= 0.:
            t = (-velocity[2] - np.sqrt(discriminant)) / g
            if 0 < t < horizon and abs(position[1] + t * velocity[1]) < 1.38:
                rising = restitution * np.sqrt(discriminant)
                strike[2] = tabletop + rising * (horizon - t) + .5 * g * (horizon - t)**2
        return strike


class TransferPlayer:
    """Inputs are calibrated RGB, the preceding requested action, and point reward."""

    def __init__(self, adapter, prior, tracker_factory, *, stroke=(0., .050)):
        self.adapter = adapter
        self.prior = prior
        self.tracker_factory = tracker_factory
        self.stroke = stroke
        self.role_rows = []
        self.gravity = -.009
        self.reset()

    def reset(self):
        self.scene = _SceneHistory()
        self.scene.tracker = self.tracker_factory()
        self.recent = deque(maxlen=9)
        self.requested = np.zeros(ACTIONS)
        self.effective = np.zeros(ACTIONS)
        self.flight = Flight()
        self.flight.acceleration = self.gravity
        self.option, self.delay = self.prior.choose()
        self.actor_id = None
        self.ball_id = None
        self.ball_hue = None
        self.opportunity = False
        self.outgoing = False
        self.stats = dict(actor_frames=0, relevant_frames=0, approach_frames=0,
                          predicted_returns=0, attributed_rewards=0)
        self.reward = 0.

    def _actor(self, tracks):
        expected = self.adapter.pose(self.effective)
        distances = [np.linalg.norm(track.position - expected) for track in tracks]
        hues = self.scene.tracker.appearances
        preferred = self.adapter.hue
        options = [i for i, track in enumerate(tracks)
                   if distances[i] < .34 and track.observed and track.id != self.ball_id
                   and (preferred is None or self._hue_distance(hues.get(track.id), preferred) <= 1)]
        if not options:
            return None
        best = min(options, key=lambda i: distances[i] - .05 * (tracks[i].id == self.actor_id))
        actor = tracks[best]
        self.actor_id = actor.id
        self.stats['actor_frames'] += 1
        return actor

    @staticmethod
    def _hue_distance(first, second):
        return 6 if first is None or second is None else min((first - second) % 12, (second - first) % 12)

    def _candidates(self, tracks, actor):
        choices = [track for track in tracks if track.observed and track.age >= 3
                   and track.id != self.actor_id and -.20 < track.position[2] < 1.7]
        if not choices:
            return (), None
        rows = _scaled([self.scene.motion(track, actor) for track in choices])
        return choices, rows

    def _relevant(self, tracks, actor):
        candidates, rows = self._candidates(tracks, actor)
        if not candidates:
            return None
        if self.role_rows:
            weights = _fit_ranking(self.role_rows, prior=self.prior.roles.relevance_weights,
                                   prior_count=self.prior.roles.reward_examples)
        else:
            weights = self.prior.roles.relevance_weights
        velocities = np.array([track.velocity for track in candidates])
        scores = rows @ weights
        # Shared physics-independent cue: a cross-court actor travels. This
        # bootstrap also lets the fully fresh learner gather reward evidence.
        scores += np.clip(np.abs(velocities[:, 1]) / .10, 0., 1.) * .42
        scores += .25 * np.array([track.id == self.ball_id for track in candidates])
        hues = self.scene.tracker.appearances
        if self.ball_hue is not None:
            scores += .38 * np.array([self._hue_distance(hues.get(track.id), self.ball_hue) <= 1
                                       for track in candidates])
        winner = int(scores.argmax())
        if scores[winner] < .06:
            return None
        ball = candidates[winner]
        if np.linalg.norm(ball.velocity) < .015 and ball.id != self.ball_id:
            return None
        self.ball_id = ball.id
        if abs(ball.velocity[1]) > .055 and ball.age >= 7:
            self.ball_hue = hues.get(ball.id)
        self.stats['relevant_frames'] += 1
        return ball

    def _goal(self, estimate):
        ready = self.adapter.ready
        if estimate is None:
            return ready, None
        position, speed = estimate
        gap = position[1] - ready[1]
        if speed[1] > .045 and -.20 < gap < .40 and self.opportunity:
            self.outgoing = True
        if speed[1] >= -.025 or gap < -.40:
            return ready, None
        self.stats['approach_frames'] += 1
        delay = self.prior.delay[0][self.delay]
        # Court-facing equilibrium and a moment beyond it are generic ways to
        # intercept a ball after the bounce and to move through the hit.
        meeting = ready[1] - .12 + np.clip(delay * speed[1], -.085, .085)
        horizon = (meeting - position[1]) / speed[1]
        if not -3. < horizon < 33.:
            return ready, None
        self.opportunity = True
        strike = self.flight.project(position, speed, max(0., horizon), self.prior.restitution)
        _, displacement, extrapolation = self.prior.numeric[0][self.option]
        aim = ready + displacement * (position - ready) + extrapolation * (strike - position)
        aim[1] = meeting + self.stroke[1] * np.clip(3. - horizon, -3., 4.)
        aim[0] = np.clip(aim[0], ready[0] - .75, ready[0] + .75)
        aim[1] = np.clip(aim[1], ready[1] - .29, ready[1] + .19)
        aim[2] = np.clip(aim[2], .23 if position[1] > -.91 else .10, ready[2] + .50)
        return aim, horizon

    def update(self, frames, preceding=None):
        if preceding is not None:
            self.requested += preceding / 4.
            lag = self.adapter.lag
            self.effective = lag * self.effective + (1. - lag) * self.requested
        tracks = self.scene.update(frames, preceding)
        self.recent.append((self.scene.tick, tracks, self.scene.tracker.support))
        actor = self._actor(tracks)
        ball = self._relevant(tracks, actor)
        estimate = self.flight.update(ball, self.scene.tick, self.prior.bounce_values)
        self.prior.bounce_values[:] = self.prior.bounce_values[-40:]
        goal, horizon = self._goal(estimate)
        apparent = None if actor is None else np.asarray(actor.position)
        return self.adapter.command(self.requested, self.effective, apparent, goal,
                                    wrist=self.stroke[0] if horizon is not None else 0.)

    def finish(self, reward):
        self.prior.observe(self.option, self.delay, reward,
                           opportunity=self.opportunity, outgoing=self.outgoing)
        self.gravity = self.flight.acceleration
        if reward:
            self._attribute_reward()
        self.stats['predicted_returns'] = int(self.outgoing)
        return dict(**self.stats, option=self.option,
                    delay_frames=float(self.prior.delay[0][self.delay]),
                    opportunity=self.opportunity, gravity=self.gravity)

    def _attribute_reward(self):
        now = self.scene.tick
        histories = {}
        for tick, tracks, support in self.recent:
            for track in tracks:
                if track.observed and track.id in support:
                    histories.setdefault(track.id, []).append((tick, track, support[track.id][0]))
        paths = [path for path in histories.values() if len(path) >= 3]
        if len(paths) < 2:
            return
        final_tracks = [path[-1][1] for path in paths]
        features = []
        for path in paths:
            final_tick, track, last_area = path[-1]
            points = np.array([item.position for _, item, _ in path]) / 4.
            velocities = np.array([item.velocity for _, item, _ in path]) / 4.
            areas = [area for _, _, area in path]
            edge = np.max(np.abs(points) / np.array([.30, .54, .35]), axis=1)
            nearest = min((np.linalg.norm(np.asarray(track.position) - candidate.position) / 4.
                           for candidate in final_tracks if candidate.id != track.id), default=1.)
            features.append((np.linalg.norm(velocities[-1]), np.linalg.norm(points[-1] - points[0]),
                             np.linalg.norm(velocities[-1] - velocities[0]), np.log1p(last_area),
                             np.log(last_area / max(1., np.median(areas))), 1. - len(path) / len(self.recent),
                             (now - final_tick) / len(self.recent), edge[-1], edge[-1] - edge[0], nearest))
        rows = np.array(features)
        clock = self.prior.clock
        if clock is None:
            # Learn from the component whose apparent world position moved
            # furthest before a real public score, leaving weights cold at entry.
            score = rows[:, 1] * (.25 + np.maximum(0., rows[:, 8]))
        else:
            standardized = (rows - np.asarray(clock['center'])) / np.asarray(clock['scale'])
            score = np.clip(standardized, -8., 8.) @ np.asarray(clock['coefficients'])[1:]
        candidates = [track for track in final_tracks if track.id != self.actor_id]
        if len(candidates) < 2:
            return
        chosen = max(candidates, key=lambda track: score[final_tracks.index(track)])
        rows = _scaled([self.scene.motion(track) for track in candidates])
        selected = candidates.index(chosen)
        self.role_rows.extend(rows[selected] - np.delete(rows, selected, axis=0))
        self.role_rows = self.role_rows[-256:]
        self.stats['attributed_rewards'] += 1
