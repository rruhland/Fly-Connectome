"""Reward-selected relations and a visual motor model shared across games."""

from collections import deque
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


RELATIONS = ('project', 'pursue', 'hold', 'recede',
             'project_positive', 'project_negative')


class ReturnMemory:
    """Remember the observed public utility of each candidate relation."""

    def __init__(self):
        self.scores = np.zeros(len(RELATIONS))
        self.counts = np.zeros(len(RELATIONS), dtype=int)

    @property
    def means(self):
        return self.scores / np.maximum(1, self.counts)

    def choose(self, rng, explore=0.):
        if explore and rng.random() < explore:
            return int(rng.choice(len(RELATIONS)))
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen):
            return int(rng.choice(unseen))
        means = self.means
        return int(rng.choice(np.flatnonzero(np.isclose(means, means.max()))))

    def observe(self, option, reward):
        self.scores[option] += float(np.clip(reward, -1, 1))
        self.counts[option] += 1

    def state(self):
        return dict(relations=list(RELATIONS), scores=self.scores.tolist(),
                    counts=self.counts.tolist())

    def relation_name(self, option):
        return RELATIONS[option]

    @classmethod
    def restore(cls, state):
        if tuple(state.get('relations', RELATIONS)) != RELATIONS:
            raise ValueError('unknown candidate relations in state')
        memory = cls()
        memory.scores[:] = state['scores']
        memory.counts[:] = state['counts']
        return memory

    def permuted(self, permutation):
        """Return the same observations assigned to different hypotheses."""
        order = np.asarray(permutation)
        if sorted(order) != list(range(len(RELATIONS))):
            raise ValueError('a permutation of the candidate relations is required')
        memory = ReturnMemory()
        memory.scores[:] = self.scores[order]
        memory.counts[:] = self.counts[order]
        return memory


def approach_normal(position, velocity, ready):
    """Find the visual direction with both a large gap and substantial motion."""
    leverage = np.abs(velocity * (position - ready))
    direction = np.zeros(len(ready))
    direction[int(leverage.argmax())] = 1.
    return direction


def _crossing_time(gap, speed, acceleration):
    if abs(speed) < 1e-6 or speed * gap >= 0.:
        return None
    linear_time = -gap / speed
    discriminant = speed * speed - 2 * acceleration * gap
    if abs(acceleration * linear_time) < .04 * abs(speed) or discriminant <= 0:
        return linear_time
    rate = speed + np.copysign(np.sqrt(discriminant), speed)
    return -2 * gap / rate


def _reflect(position, bounds):
    low, high = map(np.asarray, bounds)
    bounded = np.isfinite(low) & np.isfinite(high) & (high > low)
    span = high[bounded] - low[bounded]
    phase = (position[bounded] - low[bounded]) % (2 * span)
    position[bounded] = low[bounded] + np.minimum(phase, 2 * span - phase)


def relation_goal(option, actor_position, mover_position, mover_velocity,
                  ready_position, *, acceleration=None, normal=None,
                  motion_bounds=None, response_axis=None, offset=.03):
    """Evaluate one spatial hypothesis on unlabeled positions and velocities."""
    name = RELATIONS[option] if isinstance(option, (int, np.integer)) else option
    if name not in RELATIONS:
        raise ValueError('unknown candidate relation')
    ready = np.asarray(ready_position, dtype=float)
    if name == 'hold' or mover_position is None or mover_velocity is None:
        return ready.copy()
    position = np.asarray(mover_position, dtype=float)
    velocity = np.asarray(mover_velocity, dtype=float)
    acceleration = np.zeros_like(velocity) if acceleration is None else np.asarray(acceleration)
    normal = approach_normal(position, velocity, ready) if normal is None else np.asarray(normal)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    gap = np.dot(position - ready, normal)
    time = _crossing_time(gap, np.dot(velocity, normal), np.dot(acceleration, normal))
    if time is None:
        return ready.copy()
    time = min(time, 120.)
    goal = position.copy() if name == 'pursue' else position + time * (velocity + .5 * time * acceleration)
    if motion_bounds is not None:
        _reflect(goal, motion_bounds)
    goal -= np.dot(goal - ready, normal) * normal
    if name == 'recede':
        goal = ready - (goal - ready)
    if name in ('project_positive', 'project_negative') and response_axis is not None:
        direction = response_axis - np.dot(response_axis, normal) * normal
        extent = np.linalg.norm(direction)
        if extent > 1e-8:
            goal += (offset if name == 'project_positive' else -offset) * direction / extent
    return goal


class MotorModel:
    """Regression from legal action and visual position to visual displacement."""

    def __init__(self, dimensions=3, actions=3, capacity=384):
        self.dimensions = dimensions
        self.actions = actions
        self.capacity = capacity
        self.positions = np.empty((capacity, dimensions))
        self.commands = np.empty((capacity, actions))
        self.responses = np.empty((capacity, dimensions))
        self.samples = 0

    def add(self, position, action, displacement):
        if np.linalg.norm(displacement) > .065 or not np.any(action):
            return
        index = self.samples % self.capacity
        self.positions[index] = position
        self.commands[index] = action
        self.responses[index] = displacement
        self.samples += 1

    def matrix(self, position):
        count = min(self.capacity, self.samples)
        commands = self.commands[:count]
        positions = self.positions[:count]
        responses = self.responses[:count]
        delta = positions - position
        weights = 1. / (1. + (delta * delta).sum(axis=1) / .045**2)
        weighted = commands * weights[:, None]
        return np.linalg.solve(commands.T @ weighted + .22 * np.eye(self.actions),
                               weighted.T @ responses).T

    def command(self, position, goal, *, jacobian=None):
        jacobian = self.matrix(position) if jacobian is None else jacobian
        correction = goal - position
        step = .22 * np.where(np.abs(correction) < .003, 0., correction)
        action = np.linalg.lstsq(jacobian, step, rcond=.13)[0]
        # Refine the projected least-squares solution when joints saturate.
        action = np.clip(action, -1., 1.)
        norm_squared = np.square(jacobian).sum()
        if norm_squared > 1e-9:
            for _ in range(5):
                action = np.clip(action + jacobian.T @ (step - jacobian @ action)
                                 / norm_squared, -1., 1.)
        return action

    def state(self):
        count = min(self.capacity, self.samples)
        return dict(dimensions=self.dimensions, actions=self.actions, capacity=self.capacity,
                    samples=self.samples, positions=self.positions[:count].tolist(),
                    commands=self.commands[:count].tolist(),
                    responses=self.responses[:count].tolist())

    @classmethod
    def restore(cls, state):
        motor = cls(state['dimensions'], state['actions'], state['capacity'])
        motor.samples = state['samples']
        count = min(motor.capacity, motor.samples)
        if count:
            motor.positions[:count] = state['positions']
            motor.commands[:count] = state['commands']
            motor.responses[:count] = state['responses']
        return motor


def _polynomial_geometry(samples):
    time = np.arange(1 - samples, 1, dtype=float)
    linear = np.column_stack((np.ones(samples), time))
    quadratic = np.column_stack((linear, .5 * time * time))
    return linear, quadratic, np.linalg.pinv(linear), np.linalg.pinv(quadratic)


FITS = {samples: _polynomial_geometry(samples) for samples in range(3, 11)}


class FlightHistory:
    """Fit a short trajectory and restart the fit on a visible deflection."""

    def __init__(self):
        self.identity = None
        self.positions = deque(maxlen=max(FITS))
        self.frame = None

    def update(self, track, frame):
        if track is None:
            return None
        position = np.asarray(track.position)
        reported_velocity = np.asarray(track.velocity)
        if not track.observed:
            return position, reported_velocity, np.zeros_like(position)
        if track.id != self.identity or self.frame != frame - 1:
            self.positions.clear()
        self.identity, self.frame = track.id, frame
        if len(self.positions) >= 2:
            previous = self.positions[-1] - self.positions[-2]
            difference = position - self.positions[-1] - previous
            if np.linalg.norm(difference) > max(.006, .58 * np.linalg.norm(previous)):
                self.positions.clear()
        self.positions.append(position)
        count = len(self.positions)
        if count < 3:
            velocity = position - self.positions[-2] if count == 2 else reported_velocity
            return position, velocity, np.zeros_like(position)
        samples = np.asarray(self.positions)
        linear, quadratic, linear_inverse, quadratic_inverse = FITS[count]
        intercept, velocity = linear_inverse @ samples
        acceleration = np.zeros_like(position)
        if count >= 6:
            coefficients = quadratic_inverse @ samples
            simple_error = np.square(samples - linear @ np.stack((intercept, velocity))).sum(axis=0)
            curve_error = np.square(samples - quadratic @ coefficients).sum(axis=0)
            curved = (curve_error < .40 * simple_error) & (np.abs(coefficients[2]) < .004)
            intercept[curved], velocity[curved], acceleration[curved] = coefficients[:, curved]
        return intercept, velocity, acceleration


@dataclass(frozen=True)
class _VisualStep:
    action: np.ndarray | None
    positions: dict


def _default_probes():
    half = np.array(((1., 1., 1.), (-1., 1., -1.), (-1., -1., 1.), (1., -1., -1.)))
    return np.repeat(np.concatenate((half, -half)), 2, axis=0)


class TableLearner:
    """Use reward-selected relations with a newly identified visual action map."""

    def __init__(self, role_model, memory=None, *, seed=0, probe_frames=16,
                 explore=0., probes=None, learn_roles=False, stop_probing=None):
        try:
            from .tt3_visual import RoleBinder
        except ImportError:
            from tt3_visual import RoleBinder

        self.binder = RoleBinder(role_model, learn=learn_roles)
        self.learn_roles = learn_roles
        self.memory = ReturnMemory() if memory is None else memory
        self.motor = MotorModel()
        self.rng = np.random.default_rng(seed)
        self.explore = explore
        self.probes = _default_probes() if probes is None else np.asarray(probes, dtype=float)
        if self.probes.ndim != 2 or self.probes.shape[1] != 3 or not len(self.probes):
            raise ValueError('at least one legal three-rate probe is required')
        if not np.isfinite(self.probes).all() or (np.abs(self.probes) > 1.).any():
            raise ValueError('probe rates must be finite and within [-1, 1]')
        self.probe_frames = probe_frames
        self.stop_probing = stop_probing
        self.option = None

    def reset(self, option=None):
        self.binder.reset()
        self.flight = FlightHistory()
        self.option = self.memory.choose(self.rng, self.explore) if option is None else option
        self.history = deque(maxlen=max(32, self.probe_frames + 2))
        self.starting_motor = self.motor.state()
        self.frame = self.bound_frames = self.probe_count = 0
        self.probe_limit = self.probe_frames
        self.first_binding = None
        self.ready = self.normal = self.action = self.identity = self.goal = None
        self.reward = 0.

    def _update_motor(self, controlled):
        identity = controlled.id
        if identity != self.identity:
            if self.identity is not None:
                self.motor = MotorModel.restore(self.starting_motor)
            self.identity = identity
            recorded = list(self.history)
            reset_age = self.frame + 1 - len(recorded)
            initial = next(((index + reset_age, row.positions[identity])
                            for index, row in enumerate(recorded) if identity in row.positions), None)
            if self.ready is None or (initial is not None and initial[0] < self.probe_frames):
                self.ready = np.asarray(controlled.position) if initial is None else initial[1].copy()
                self.normal = None
        else:
            recorded = list(self.history)[-2:]
        for previous, current in zip(recorded, recorded[1:]):
            start, end = previous.positions.get(identity), current.positions.get(identity)
            if start is not None and end is not None and current.action is not None:
                self.motor.add(start, current.action, end - start)

    def act(self, frames):
        binding = self.binder.update(frames, self.action)
        tracks = tuple(binding.tracks) + (binding.controlled, binding.relevant)
        positions = {track.id: np.asarray(track.position) for track in tracks
                     if track is not None and track.observed}
        self.history.append(_VisualStep(self.action, positions))
        controlled = binding.controlled
        if controlled is not None:
            self._update_motor(controlled)
        flight = self.flight.update(binding.relevant, self.frame)
        if controlled is not None and binding.relevant is not None:
            self.bound_frames += 1
            if self.first_binding is None:
                self.first_binding = self.frame
        if (self.frame < self.probe_limit and controlled is not None and self.stop_probing
                and self.stop_probing(self, binding)):
            self.probe_limit = self.frame
        if self.frame < self.probe_limit or self.ready is None:
            action = self.probes[self.probe_count % len(self.probes)].copy()
            self.probe_count += 1
        elif controlled is None:
            action = np.zeros(3)
        else:
            position = np.asarray(controlled.position)
            jacobian = self.motor.matrix(position)
            if flight is None:
                if self.goal is None or self.frame - (self.flight.frame or 0) > 3:
                    self.goal = self.ready.copy()
            else:
                mover, velocity, acceleration = flight
                if self.normal is None and np.linalg.norm(velocity) > 1e-4:
                    self.normal = approach_normal(mover, velocity, self.ready)
                axis = jacobian[:, 0]
                goal = getattr(self.memory, 'goal', relation_goal)
                self.goal = goal(self.option, position, mover, velocity, self.ready,
                                 acceleration=acceleration, normal=self.normal,
                                 response_axis=axis)
                if self.goal is None:
                    self.goal = self.ready.copy()
            action = self.motor.command(position, self.goal, jacobian=jacobian)
        self.action = action
        self.frame += 1
        return action

    def observe(self, reward, *, terminal_frames=None):
        self.reward += float(reward)
        if self.learn_roles and terminal_frames is not None:
            self.binder.update(terminal_frames, self.action)
            self.binder.observe_reward(self.reward)

    def finish(self):
        self.memory.observe(self.option, self.reward)
        name = getattr(self.memory, 'relation_name', str)(self.option)
        return dict(option=name, option_index=int(self.option), public_reward=self.reward,
                    frames=self.frame, bound_frames=self.bound_frames,
                    first_binding=self.first_binding, probes=self.probe_count,
                    motor_samples=self.motor.samples)


def play_point(arena, learner, *, option=None, reward_map=None, capture=None):
    """Run a point and attach evaluator statistics after action selection ends."""
    learner.reset(option)
    frames = arena.reset()
    events = []
    commands = sha256()
    done = False
    while not done:
        action = learner.act(frames)
        commands.update(np.asarray(action, dtype='<f8').tobytes())
        frames, reward, done, info = arena.step(action)
        public_reward = reward if reward_map is None else reward_map(reward, done)
        learner.observe(public_reward, terminal_frames=frames if done else None)
        if capture is not None:
            capture(frames, action, public_reward)
        events.extend(info.get('events', ()))
    result = learner.finish()
    result.update({name: int(info[name]) for name in ('contacts', 'legal_returns', 'rival_returns',
                                                      'points_for', 'points_against') if name in info})
    result.update(events=events, action_sha256=commands.hexdigest())
    return result
