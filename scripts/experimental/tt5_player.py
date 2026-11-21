"""A camera-only table tennis player with fitted body response and timed strokes."""

import math

import numpy as np

from .tt4_transfer import TransferPlayer
from .tt5_player_body import BodyServo, fit_arm


# Apparatus and ordinary flight priors in table-relative coordinates, with time
# in camera frames. They are shared by every source condition.
BALL_RADIUS = .020
TABLE_END = 1.37
TABLE_HALF_WIDTH = .7625
NET_CLEARANCE = .178
FRAME_DT = 1. / 30.
GRAVITY = 9.81 * FRAME_DT**2
DRAG = .17
TRANSITION = np.eye(6)
TRANSITION[:3, 3:] = np.eye(3)
GRAVITY_VECTOR = np.array([0., 0., GRAVITY])


def _prediction(position, velocity, duration, restitution):
    acceleration = -DRAG * np.linalg.norm(velocity) * velocity - GRAVITY_VECTOR
    end = position + velocity * duration + acceleration * (.5 * duration**2)
    end_speed = velocity + acceleration * duration
    bounced = False
    if end[2] < BALL_RADIUS <= position[2] and end_speed[2] < 0.:
        contact_time = 2. * (position[2] - BALL_RADIUS) / max(.001, -velocity[2] + math.sqrt(max(
            0., velocity[2]**2 - 2. * acceleration[2] * (position[2] - BALL_RADIUS))))
        contact_time = np.clip(contact_time, 0., duration)
        hit = position + velocity * contact_time + acceleration * (.5 * contact_time**2)
        if abs(hit[0]) <= TABLE_HALF_WIDTH and abs(hit[1]) <= TABLE_END:
            end_speed = velocity + acceleration * contact_time
            end_speed = end_speed * [.95, .94, -restitution]
            remaining = duration - contact_time
            hit[2] = BALL_RADIUS
            end = hit + end_speed * remaining - GRAVITY_VECTOR * (.5 * remaining**2)
            end_speed -= GRAVITY_VECTOR * remaining
            bounced = True
    return end, end_speed, bounced


class BallFlight:
    """Fuse stereo positions and coast through occlusions with a simple flight model."""

    def __init__(self):
        self.state = None
        self.covariance = None
        self.last = None
        self.sightings = 0
        self.bounce_samples = []
        self.last_bounce = None

    def update(self, tick, observation, sigma, restitution):
        self.last_bounce = None
        if observation is None and self.state is None:
            return None
        point = None if observation is None else np.asarray(observation.position)
        if point is not None:
            default_depth = .012 + .019 * (point[1] + 1.42)**2
            sigma = np.array([.006, default_depth, .009]) if sigma is None else np.asarray(sigma)
            sigma = np.clip(sigma, (.0025, .0055, .0025), (.07, .26, .08))
        if self.state is None or tick - self.last[0] > 7:
            self.state = None
            if point is None:
                return None
            speed = np.clip(observation.velocity, -.17, .17)
            self.state = np.concatenate((point, speed))
            self.covariance = np.diag(np.concatenate((sigma**2, [.033**2, .065**2, .043**2])))
            self.last = tick, point, sigma
            self.sightings = 1
            return None
        before = self.state.copy()
        position, velocity, bounced = _prediction(before[:3], before[3:], 1., restitution)
        extra_uncertainty = np.array([.0035, .006, .0038, .0025, .005, .0032])
        transform = TRANSITION
        if bounced:
            transform = TRANSITION.copy()
            transform[5, 5] = -restitution
            extra_uncertainty[[4, 5]] *= 5.
            self.last_bounce = tick, position.copy()
        self.covariance = transform @ self.covariance @ transform.T + np.diag(extra_uncertainty**2)
        self.state = np.concatenate((position, velocity))
        if point is not None:
            distance = tick - self.last[0]
            displacement = (point - self.last[1]) / distance
            residual = point - position
            strong = np.abs(before[4]) > .035 and distance <= 3
            at_racket = abs(point[1]) > 1.23 and abs(self.last[1][1]) > 1.18
            near_bounce = min(point[2], self.last[1][2], position[2]) < .090
            direction = np.sign(before[4])
            if strong and at_racket and direction * displacement[1] < -.020 and abs(residual[1]) > .035:
                # A racket can reverse the depth velocity between camera frames.
                launch = np.clip(displacement, -.24, .24)
                self.state = np.concatenate((point, launch))
                self.covariance = np.diag(np.concatenate((sigma**2, [.026**2, .040**2, .032**2])))
            elif not bounced and near_bounce and before[5] < -.030 and residual[2] > .016:
                upward = max(displacement[2], restitution * -before[5] - GRAVITY)
                self.state[[2, 5]] = point[2], max(.012, upward)
                self.covariance[[2, 5], [2, 5]] += [.012**2, .025**2]
                self.last_bounce = tick, point.copy()
            else:
                # Inconsistent stereo pairs have limited leverage until supported
                # by another sighting. Uncertainty is largest in optical depth.
                bound = 4. * np.sqrt(np.diag(self.covariance)[:3] + sigma**2)
                adjustment = np.clip(residual, -bound, bound)
                if np.any(np.abs(residual) > 2. * bound):
                    return self._current(tick)
                innovation = self.covariance[:3, :3] + np.diag(sigma**2)
                gain = np.linalg.solve(innovation, self.covariance[:3, :]).T
                self.state += gain @ adjustment
                self.covariance -= gain @ self.covariance[:3, :]
            if self.last_bounce is not None and displacement[2] > .027 and before[5] < -.045:
                ratio = (displacement[2] + GRAVITY * .5) / -before[5]
                if .56 < ratio < 1.18:
                    self.bounce_samples.append(float(ratio))
            self.last = tick, point, sigma
            self.sightings += 1
        return self._current(tick)

    def _current(self, tick):
        if self.sightings >= 3 and self.last is not None and tick - self.last[0] <= 6:
            return self.state[:3].copy(), self.state[3:].copy()
        return None

    @staticmethod
    def crossing(position, velocity, plane, restitution):
        point, speed = position.copy(), velocity.copy()
        if point[1] <= plane:
            past = (plane - point[1]) / min(-.010, speed[1])
            return past, point + speed * past, speed
        duration = 0.
        while point[1] > plane and duration < 42. and speed[1] < -.012:
            step = min(3., max(.1, (plane - point[1]) / speed[1]))
            next_point, next_speed, _ = _prediction(point, speed, step, restitution)
            if next_point[1] <= plane:
                fraction = (plane - point[1]) / (next_point[1] - point[1])
                return duration + step * fraction, point + fraction * (next_point - point), next_speed
            point, speed = next_point, next_speed
            duration += step
        return duration, point, speed


def _launch_pitch(position, incoming, forward_speed, restitution):
    """Choose a return trajectory aimed inside the opponent's half of the table."""
    angles = np.linspace(-.02, .72, 51)
    sine, cosine = np.sin(angles), np.cos(angles)
    y_speed = incoming[1] - forward_speed
    z_speed = incoming[2]
    normal = y_speed * cosine + z_speed * sine
    tangent = -y_speed * sine + z_speed * cosine
    impulse = -(1. + restitution) * normal
    friction = -.23 * tangent
    out = np.maximum(.02, incoming[1] + impulse * cosine - friction * sine)
    rise = incoming[2] + impulse * sine + friction * cosine
    travel = np.array([-position[1], .76 - position[1]])[:, None]
    # Treat the depth drag rate as constant over the planned arc.
    loss = DRAG * out
    intervals = np.expm1(DRAG * travel) / out / DRAG
    decay = -np.expm1(-loss * intervals) / loss
    elevation = position[2] + (rise + GRAVITY / loss) * decay - GRAVITY / loss * intervals
    error = np.abs(elevation[1] - BALL_RADIUS) + 2.5 * np.maximum(0., NET_CLEARANCE - elevation[0])
    return float(angles[np.argmin(error)])


class RallyPlayer(TransferPlayer):
    """Receive only camera pairs, previous commands, and point rewards."""

    def __init__(self, adapter, prior, tracker_factory, *, adjuster=None, retain_trace=False,
                 stroke_speed=.049, transfer_strength=1., impact_restitution=.80):
        self.adjuster = adjuster
        self.retain_trace = retain_trace
        self.stroke_speed = stroke_speed
        self.transfer_strength = transfer_strength
        self.impact_restitution = impact_restitution
        super().__init__(adapter, prior, tracker_factory)

    def reset(self):
        super().reset()
        self.servo = BodyServo(self.adapter)
        self.flight = BallFlight()
        self.plan = None
        self.release = -100
        self.trace = []
        self.far_leg = True
        self.stats['coasted_frames'] = 0
        self.stats['visual_bounces'] = 0
        self.stats['apparent_hits'] = 0
        if self.adjuster is not None:
            self.adjuster.reset()

    def _actor(self, tracks):
        if hasattr(self.scene.tracker, 'actor'):
            actor = self.scene.tracker.actor
        else:
            supported = self.scene.tracker.support
            eligible = [track for track in tracks if track.observed and supported.get(track.id, (0.,))[0] > 36
                        and np.linalg.norm(np.asarray(track.position) - self.servo.pose) < .20]
            actor = min(eligible, key=lambda t: np.linalg.norm(np.asarray(t.position) - self.servo.pose), default=None)
        if actor is not None and actor.observed:
            self.stats['actor_frames'] += 1
            self.actor_id = actor.id
        return actor

    def _ball(self, tracks, actor):
        if hasattr(self.scene.tracker, 'ball'):
            ball = self.scene.tracker.ball
            if ball is not None and ball.observed:
                self.stats['relevant_frames'] += 1
            return ball
        return self._relevant(tracks, actor)

    def _surface_ratio(self):
        ratios = self.prior.bounce_values
        # Source ratios come from camera samples around a bounce. Treat the
        # historical median as one weak observation on a different surface.
        donor = (np.median(ratios[-12:]) - .65) * .27 * self.transfer_strength if ratios else 0.
        return float(np.clip(.86 + donor, .72, .99))

    def _stroke_goals(self, estimate):
        ready = self.adapter.ready
        tick = self.scene.tick
        if estimate is None:
            return ready, ready, self.adapter.home[3], None
        position, velocity = estimate
        if position[1] > .38:
            self.far_leg = True
        if velocity[1] > .027 and position[1] < -.97 and self.plan is not None and tick - self.plan[0] < 10.:
            self.outgoing = True
            self.stats['apparent_hits'] += 1
            self.release = tick
            self.far_leg = False
            self.plan = None
        if velocity[1] >= -.025 or position[1] < -1.86 or tick - self.release < 10 or not self.far_leg:
            return ready, ready, self.adapter.home[3], None
        self.stats['approach_frames'] += 1
        bias, gain, lead = self.prior.numeric[0][self.option]
        delay = self.prior.delay[0][self.delay] * self.transfer_strength
        plane = -TABLE_END - .022 + np.clip(delay * velocity[1] * .25, -.048, .048)
        restitution = self._surface_ratio()
        horizon, ball, speed = self.flight.crossing(position, velocity, plane, restitution)
        if not -2.8 < horizon < 26.:
            return ready, ready, self.adapter.home[3], None
        geometry = self.adapter.geometry
        radial = math.sqrt(max(.14, (1.82 * geometry[3])**2 - (ball[0] - geometry[0])**2
                               - (ball[2] - geometry[2])**2))
        corrected_plane = min(plane, geometry[1] + radial - .021)
        if abs(corrected_plane - plane) > .019:
            plane = corrected_plane
            horizon, ball, speed = self.flight.crossing(position, velocity, plane, restitution)
        if not -3. < horizon < 26. or ball[2] < -.04:
            return ready, ready, self.adapter.home[3], None
        self.opportunity = True
        # Keep the ballistic strike as the anchor. Numeric experience only
        # adjusts the measured lateral displacement and remaining extrapolation.
        projected = ball.copy()
        projected[0] += self.transfer_strength * (.12 * (gain - 1.) * (position[0] - ready[0])
                                                  + .12 * (lead - 1.) * (ball[0] - position[0]) + .035 * bias)
        projected[2] = np.clip(projected[2], .080, .61)
        pitch = _launch_pitch(projected, speed, self.stroke_speed, self.impact_restitution)
        if self.adapter.gains[3] == 0.:
            pitch = self.adapter.home[3]
        next_plan = np.array([tick + horizon, projected[0], plane, projected[2], pitch])
        if self.plan is None or tick - self.plan[0] > 5.:
            self.plan = next_plan
        else:
            mix = .60 if horizon < 7. else .38
            self.plan = (1. - mix) * self.plan + mix * next_plan
        contact_tick, x, depth, height, pitch = self.plan
        horizon = contact_tick - tick
        impact = .027
        point = np.array([x, depth - impact * math.cos(pitch), height - impact * math.sin(pitch)])

        def stroke(remaining):
            offset = np.clip(self.stroke_speed * remaining, -.107, .145)
            goal = point - (0., offset, 0.)
            if self.adjuster is not None:
                goal = self.adjuster.correct(goal, remaining)
            goal[2] = max(.054, goal[2])
            return goal

        return stroke(horizon), stroke(horizon - 1.), pitch, horizon

    def update(self, frames, preceding=None):
        self.servo.advance(preceding)
        tracks = self.scene.update(frames, preceding)
        self.recent.append((self.scene.tick, tracks, self.scene.tracker.support))
        actor = self._actor(tracks)
        pitch = getattr(self.scene.tracker, 'actor_pitch', None) if actor is not None and actor.observed else None
        self.servo.observe(actor, pitch)
        ball = self._ball(tracks, actor)
        if ball is not None and not ball.observed:
            ball = None
        if self.adjuster is not None:
            self.adjuster.observe(self.scene.tick, None if ball is None else ball.position,
                                  None if actor is None or not actor.observed else actor.position)
        estimate = self.flight.update(self.scene.tick, ball, getattr(self.scene.tracker, 'ball_sigma', None),
                                      self._surface_ratio())
        self.stats['coasted_frames'] += int(ball is None and estimate is not None)
        self.stats['visual_bounces'] += int(self.flight.last_bounce is not None)
        goal, lead, desired_pitch, horizon = self._stroke_goals(estimate)
        command = self.servo.command(goal, lead, desired_pitch)
        if self.retain_trace:
            self.trace.append(dict(tick=self.scene.tick, ball=None if ball is None else list(ball.position),
                                   estimate=None if estimate is None else [row.tolist() for row in estimate],
                                   actor=self.servo.pose.tolist(), goal=goal.tolist(), horizon=horizon,
                                   pitch=float(self.servo.pitch), desired_pitch=float(desired_pitch)))
        return command

    def finish(self, reward):
        self.prior.bounce_values.extend(self.flight.bounce_samples)
        self.prior.bounce_values[:] = self.prior.bounce_values[-40:]
        self.prior.observe(self.option, self.delay, reward, opportunity=self.opportunity, outgoing=self.outgoing)
        if reward:
            self._attribute_reward()
        self.stats['predicted_returns'] = int(self.outgoing)
        details = dict(**self.stats, option=self.option, delay_frames=float(self.prior.delay[0][self.delay]),
                       opportunity=self.opportunity, gravity=-GRAVITY, restitution=self._surface_ratio())
        if self.adjuster is not None:
            details['contact_adjustment'] = self.adjuster.finish(reward)
        return details
