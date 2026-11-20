"""Fresh numerical learner for the same public role adapter used in transfer."""

from collections import deque
from dataclasses import dataclass
import math
import random

from pong_role_transfer import PIXEL, RoleAdapter


ACTIONS = (0, 1, 2)
RELATIONS = ('toward', 'away')
MOTOR_HISTORY = 16


@dataclass
class _MotorSamples:
    count: int = 0
    prior_mean: float = 0.
    response_mean: float = 0.
    prior_variation: float = 0.
    response_variation: float = 0.
    covariance: float = 0.

    def add(self, prior, response):
        self.count += 1
        prior_residual = prior - self.prior_mean
        response_residual = response - self.response_mean
        self.prior_mean += prior_residual / self.count
        self.response_mean += response_residual / self.count
        self.prior_variation += prior_residual * (prior - self.prior_mean)
        self.response_variation += response_residual * (response - self.response_mean)
        self.covariance += prior_residual * (response - self.response_mean)


class _VisualMotor:
    """Fit one shared velocity persistence and one impulse per action online."""

    def __init__(self):
        self.bank = [[_MotorSamples() for _ in ACTIONS] for _ in range(MOTOR_HISTORY)]

    def add(self, history, prior, response):
        for samples, action in zip(self.bank, reversed(history)):
            samples[action].add(prior, response)

    @staticmethod
    def _fit(samples):
        variation = sum(sample.prior_variation for sample in samples)
        covariance = sum(sample.covariance for sample in samples)
        inertia = max(-1., min(1., covariance / variation)) if variation > 0 else 0.
        impulses = [s.response_mean - inertia * s.prior_mean for s in samples]
        return inertia, impulses

    def estimate(self):
        best = -1.
        estimate = (0, *self._fit(self.bank[0]))
        for lag, samples in enumerate(self.bank):
            if min(s.count for s in samples) < 2:
                continue
            count = sum(s.count for s in samples)
            inertia, impulses = self._fit(samples)
            mean = sum(s.count * u for s, u in zip(samples, impulses)) / count
            signal = sum(s.count * (u - mean)**2 for s, u in zip(samples, impulses))
            noise = sum(s.response_variation - 2 * inertia * s.covariance
                        + inertia**2 * s.prior_variation for s in samples)
            # Discount estimates with few differentiated requests. Additional
            # lag is useful only when it explains a clearer camera response.
            quality = signal / (signal + max(1e-12, noise)) * count / (count + 2 * (lag + 1))
            if quality > best:
                best = quality
                estimate = (lag, inertia, impulses)
        return estimate

    def choose(self, rng, along, target, speed, elapsed, history):
        samples = self.bank[0]
        smallest_count = min(sample.count for sample in samples)
        lag, inertia, impulses = self.estimate()
        if smallest_count == 0 or max(impulses) == min(impulses):
            return rng.choice([a for a in ACTIONS if samples[a].count == smallest_count]), True

        for action in list(history)[-lag:] if lag else ():
            speed = inertia * speed + impulses[action]
            along = max(-.5, min(.5, along + elapsed * speed))

        next_velocity = [inertia * speed + impulse for impulse in impulses]
        # Choose the request whose next position is closest to the target.
        # When clipping makes requests equivalent, prefer less residual speed.
        def cost(action):
            position = max(-.5, min(.5, along + elapsed * next_velocity[action]))
            return abs(position - target), abs(next_velocity[action])

        costs = [cost(action) for action in ACTIONS]
        return rng.choice([a for a in ACTIONS if costs[a] == min(costs)]), False

    def state(self):
        lag, inertia, impulses = self.estimate()
        return {
            'samples_per_action': [sample.count for sample in self.bank[0]],
            'delay_frames': lag,
            'velocity_persistence': inertia,
            'velocity_impulse_per_action': impulses,
        }


@dataclass
class _RelationEvidence:
    trials: int = 0
    visual_returns: int = 0
    terminal_losses: int = 0
    score: float = 0.
    utility: float = 0.

    def add(self, visual_return, score, terminal_loss):
        self.trials += 1
        self.visual_returns += int(visual_return)
        self.terminal_losses += int(terminal_loss)
        self.score += score
        self.utility += float(visual_return) - float(terminal_loss) + math.tanh(score)

    def state(self):
        return {
            'trials': self.trials,
            'visual_returns': self.visual_returns,
            'terminal_losses': self.terminal_losses,
            'score': self.score,
            'mean_utility': self.utility / self.trials if self.trials else 0.,
        }


class ColdVisualSession:
    """Learn from requests, visible positions, public elapsed time and reward."""

    def __init__(self, policy_seed):
        self.policy_seed = int(policy_seed)
        self.rng = random.Random(self.policy_seed)
        self.roles = RoleAdapter()
        self.motor = _VisualMotor()
        self.relations = {relation: _RelationEvidence() for relation in RELATIONS}
        self.frames = 0
        self.lives = 0
        self.actor = None

    def begin_life(self):
        if self.actor is not None:
            raise RuntimeError('end the current life before beginning another')
        self.actor = _ColdVisualActor(self)
        return self.actor

    def end_life(self):
        if self.actor is None:
            raise RuntimeError('no life to end')
        self.actor.finish_trial()
        self.actor = None
        self.lives += 1

    def select_relation(self):
        unseen = [relation for relation, e in self.relations.items() if not e.trials]
        if unseen:
            return self.rng.choice(unseen)
        exposures = sum(e.trials for e in self.relations.values())
        priorities = {
            relation: e.utility / e.trials + math.sqrt(math.log(exposures) / e.trials)
            for relation, e in self.relations.items()
        }
        best = max(priorities.values())
        return self.rng.choice([relation for relation, value in priorities.items() if value == best])

    def state(self):
        return {
            'policy_seed': self.policy_seed,
            'camera_transitions': self.frames,
            'lives': self.lives,
            'role_binding': self.roles.state(),
            'motor': self.motor.state(),
            'relations': {relation: e.state() for relation, e in self.relations.items()},
        }


class _ColdVisualActor:
    def __init__(self, session):
        self.session = session
        self.roles = session.roles
        self.paddle = None
        self.paddle_id = None
        self.paddle_speed = 0.
        self.elapsed = 0.
        self.last_action = None
        self.requests = deque(maxlen=MOTOR_HISTORY)
        self.ball = deque(maxlen=3)
        self.ball_id = None
        self.closing = None
        self.relation = None
        self.visual_return = False
        self.trial_score = 0.
        self.started = False
        self.finished = False

    def reset(self, frame):
        if self.started:
            raise RuntimeError('a fresh actor is required for each life')
        self.started = True
        self.roles.reset()
        self._read(self.roles.observe(frame), 0.)

    def choose(self, baseline_action=1):
        if not self.started or self.finished or self.last_action is not None:
            raise RuntimeError('choose requires a new public camera frame during a live episode')
        target = None
        if not self.roles.ready or self.paddle is None:
            action = self.roles.probe_action()
            mode = 'role_probe'
        else:
            # A missing ball freezes the target at the current paddle position.
            # New options are selected only once their visual target exists.
            target = self.paddle[0]
            if self.ball and self.roles.seconds - self.ball[-1][0] <= 4 * self.elapsed:
                if self.relation is None:
                    self.relation = self.session.select_relation()
                forecast = self._projected_ball()
                target = forecast if self.relation == 'toward' else -forecast
            action, probing = self.session.motor.choose(
                self.session.rng, self.paddle[0], target, self.paddle_speed, self.elapsed, self.requests)
            mode = 'motor_probe' if probing else 'relation'

        self.last_action = action
        self.requests.append(action)
        return action, {'action': action, 'mode': mode, 'relation': self.relation,
                        'visual_target': target}

    def observe(self, transition):
        if self.last_action is None or self.finished:
            raise RuntimeError('a fresh action is required before each transition')

        dt = float(transition.elapsed_seconds)
        self.elapsed = dt
        self.session.frames += 1
        if self.relation is not None:
            self.trial_score += float(transition.reward)
        self.roles.observe(transition.frame, dt, self.last_action)
        self._read(self.roles.canonical_entities, dt)

        self.last_action = None
        self.finished = transition.terminated or transition.truncated
        if self.finished:
            self.finish_trial(terminal_loss=bool(transition.terminated and transition.reward <= 0))

    def finish_trial(self, terminal_loss=False):
        if self.relation is not None and (self.visual_return or self.trial_score or terminal_loss):
            self.session.relations[self.relation].add(self.visual_return, self.trial_score, terminal_loss)
        self.relation = None
        self.visual_return = False
        self.trial_score = 0.

    def _read(self, rows, dt):
        pair = self.roles.pair_ids
        observed = {row['id']: row['normalized_position'] for row in rows if row['observed']}
        previous_paddle = self.paddle
        previous_id = self.paddle_id
        self.paddle_id = None if pair is None else pair[0]
        self.paddle = observed.get(self.paddle_id)
        if (previous_paddle is not None and self.paddle is not None
                and self.paddle_id == previous_id and dt > 0):
            displacement = (self.paddle[0] - previous_paddle[0]) / dt
            self.session.motor.add(self.requests, self.paddle_speed, displacement)
            self.paddle_speed = displacement
        else:
            self.paddle_speed = 0.

        if pair is None or pair[1] not in observed or self.paddle is None:
            # The contact can temporarily merge both silhouettes. Keep the last
            # progress direction until a fresh ball separates again.
            return
        position = observed[pair[1]]
        gap = abs(position[1] - self.paddle[1])
        if self.ball and (pair[1] != self.ball_id or self.roles.seconds - self.ball[-1][0] > 4 * dt):
            self.ball.clear()
            self.closing = None
        self.ball_id = pair[1]
        if self.ball:
            preceding_gap = self.ball[-1][3]
            change = gap - preceding_gap
            if change:
                closing = change < 0
                if closing and self.closing is False:
                    # The next approach closes the previous action trial, leaving
                    # room for public score earned during the outbound flight.
                    if self.visual_return or self.trial_score:
                        self.finish_trial()
                elif not closing and self.closing and min(gap, preceding_gap) <= 4 * PIXEL + change:
                    self.visual_return = True
                self.closing = closing
        self.ball.append((self.roles.seconds, *position, gap))

    def _projected_ball(self):
        _, along, _, gap = self.ball[-1]
        if len(self.ball) < 2 or not self.closing:
            return along

        time, _, _, _ = self.ball[-1]
        start, past_along, _, past_gap = self.ball[0]
        if time == start or past_gap <= gap:
            return along
        # Forward ray from the observed motion to the rail. Mirror the ray at
        # the normalized camera boundaries to account for a possible side wall.
        forecast = along + (along - past_along) * gap / (past_gap - gap)
        return .5 - abs((forecast + .5) % 2. - 1.)
