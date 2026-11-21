"""Recall the blade motion of an earlier optically observed service reply."""

from dataclasses import asdict, dataclass, field

import numpy as np

from .tt5_player import DRAG, NET_CLEARANCE, TABLE_END, TABLE_HALF_WIDTH, BallFlight
from .tt7_flight import nominal_exit, outgoing_speed
from .tt8_service_drive import ServiceDriveServo
from .tt9_precontact_registration import RegistrationPlayer


@dataclass
class VisualReply:
    farthest: float = 0.
    previous: tuple | None = None
    retreats: int = 0
    tick: int | None = None

    def observe(self, tick, ball, serve_tick):
        if self.tick is not None or ball is None or tick <= serve_tick + 3:
            return
        position = np.asarray(ball)
        depth = position[1]
        if .48 < depth < 1.70:
            self.farthest = max(self.farthest, depth)
        if self.previous is not None:
            then, last = self.previous
            interval = tick - then
            movement = (position - last) / interval
            # Two consecutive observations moving toward us prevent an
            # isolated reflected far-side depth from triggering the stroke.
            inbound = (interval <= 3 and -.22 < movement[1] < -.035
                       and abs(movement[0]) < .06 and abs(movement[2]) < .14)
            self.retreats = self.retreats + 1 if inbound else 0
        if self.farthest > .48 and 0. < depth < self.farthest - .14 and self.retreats >= 2:
            self.tick = tick
        self.previous = tick, position


def reply_tick(trace, serve_tick):
    observer = VisualReply()
    for frame in trace:
        observer.observe(frame['tick'], frame['ball'], serve_tick)
        if observer.tick is not None:
            break
    return observer.tick


def _visible_bounce(trace, contact):
    points = [(row['tick'], row['ball']) for row in trace
              if row['ball'] is not None and contact + 5 <= row['tick'] <= contact + 35]
    crossed_net = False
    for i, (tick, ball) in enumerate(points):
        x, y, z = ball
        if -.40 < y < .24 and z > NET_CLEARANCE:
            crossed_net = True
        if not crossed_net or not .07 < y < TABLE_END - .04 or abs(x) >= TABLE_HALF_WIDTH or not -.015 < z < .085:
            continue
        starts = [(t, p) for t, p in points[max(0, i - 4):i]
                  if 0 < tick - t <= 5 and .015 < y - p[1] < .66
                  and abs(x - p[0]) < .055 and p[2] > z + .035]
        ends = [(t, p) for t, p in points[i + 1:i + 5]
                if 0 < t - tick <= 5 and .005 < p[1] - y < .66
                and abs(x - p[0]) < .055 and p[2] > z + .035]
        if starts and ends:
            return dict(tick=tick, ball=ball, before=list(starts[-1]), after=list(ends[0]))
    return None


def _blade_motion(trace, contact):
    views = [row for row in trace if contact - 3. <= row['tick'] <= contact + 1.25
             and row['blade'] is not None and row['blade_pitch'] is not None]
    if len(views) < 2 or views[-1]['tick'] < contact - 1.7:
        return None
    t = np.array([row['tick'] - contact for row in views])
    poses = [[*row['blade'], row['blade_pitch']] for row in views]
    displacement = np.linalg.lstsq(np.column_stack((np.ones(len(t)), t)), poses, rcond=None)[0]
    pitch = float(displacement[0, 3])
    speed = displacement[1, 1:3]
    if not -.05 < pitch < .9 or not -.015 < speed[0] < .18 or abs(speed[1]) > .09:
        return None
    return pitch, speed, [row['tick'] for row in views]


def optical_stroke(trace, serve_tick):
    """Keep a short approach and departure only when a far-half bounce is seen."""
    reply = reply_tick(trace, serve_tick)
    if reply is None:
        return None
    for row in trace:
        tick, horizon, estimate = (row[key] for key in ('tick', 'horizon', 'estimate'))
        if not reply + 5 <= tick <= reply + 32 or estimate is None or horizon is None or not 0. <= horizon < 2.8:
            continue
        position, incoming = (np.array(values) for values in estimate)
        if not -1.6 < position[1] < -.82 or incoming[1] >= -.035:
            continue
        contact = tick + horizon
        exit_fit = outgoing_speed(trace, contact)
        bounce = _visible_bounce(trace, contact)
        blade = _blade_motion(trace, contact)
        if exit_fit is None or bounce is None or blade is None:
            continue
        departure, hit, sightings = exit_fit
        # The first fitted departing arc must lead to the bounce in the same
        # exchange, before a second stroke could supply another departure.
        elapsed = bounce['tick'] - contact
        predicted_depth = hit[1] + np.log1p(DRAG * departure[0] * elapsed) / DRAG
        if (not 8. < elapsed < 31. or bounce['tick'] <= sightings[-1]
                or abs(predicted_depth - bounce['ball'][1]) > .35):
            continue
        _, _, incoming = BallFlight.crossing(position, incoming, hit[1], .86)
        pitch, speed, eye_frames = blade
        return dict(reply=reply, contact=float(contact), incoming=incoming.tolist(), position=hit.tolist(),
                    blade_pitch=pitch, desired_pitch=row['desired_pitch'], blade_speed=speed.tolist(),
                    blade_frames=eye_frames, departure=departure.tolist(), departure_frames=list(sightings),
                    bounce=bounce)
    return None


@dataclass
class StrokeMemory:
    samples: list = field(default_factory=list)

    def state(self):
        return asdict(self)

    def nearest(self, ball, incoming):
        if not self.samples:
            return None
        # The lateral position enters weakly because the primitive shifts with
        # the new projected impact. The vertical and depth velocities set the
        # main relative blade-to-ball motion.
        context = np.array([ball[0], ball[2], *incoming[1:]])
        scale = np.array([.35, .14, .045, .036])

        def distance(sample):
            reference = np.array([sample['position'][0], sample['position'][2], *sample['incoming'][1:]])
            return float(np.linalg.norm((context - reference) / scale))

        closest = min(self.samples, key=distance)
        gap = distance(closest)
        return (closest, gap) if gap <= 1.75 else None


class StrokeServo(ServiceDriveServo):
    recalled_speed = None

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        super()._strike_objective(normal, rhs, states, effect, jac, targets)
        if self.recalled_speed is None or self.intercept is None or not 0. < self.intercept < self.horizon:
            return
        motion = np.diff(self.fit.pose(states)[:, 1:3], axis=0)
        response = np.einsum('an,tmn->tamn', jac[1:3], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weight = np.exp(-.5 * ((times - self.intercept) / 1.2)**2)[:, None] * [2.1, 1.3]
        design = (response * weight[..., None, None]).reshape(-1, rhs.size)
        normal += design.T @ design
        rhs += design.T @ ((self.recalled_speed - motion) * weight).ravel()


class StrokePlayer(RegistrationPlayer):
    """Shift a remembered stroke to the current visually predicted meeting point."""

    def __init__(self, *args, strokes, retrieve=True, learn_strokes=False, **kwargs):
        self.strokes = strokes
        self.retrieve = retrieve
        self.learn_strokes = learn_strokes
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return StrokeServo(self.adapter, self.timing_memory, learn=self.adjust)

    def reset(self):
        super().reset()
        self.reply_observer = VisualReply()
        self.recalled = None
        self.choice = None
        self.stroke_closed = False

    @property
    def reply(self):
        return self.reply_observer.tick

    def _observe_reply(self):
        if self.serve_plan is None:
            return
        ball = self.scene.tracker.ball
        if ball is not None and ball.observed:
            self.reply_observer.observe(self.scene.tick, ball.position, self.serve_plan[0])

    def _retarget(self, sample, hit, incoming, pitch):
        previous = np.array(sample['incoming'][1:])
        speed = np.array(sample['blade_speed'])
        pace = float(np.clip(speed[0] + .45 * (incoming[1] - previous[0]), .028, .105))
        lift = float(np.clip(speed[1], -.020, .030))
        # Adjust the recalled upward velocity for the current impact height,
        # then use the local change in the collision model to tilt the blade.
        # The optical departure is the anchor for adjusting the arc.
        out, up = sample['departure']
        flight_frames = (np.expm1(DRAG * (.76 - hit[1])) / (DRAG * out))
        desired_rise = up - (hit[2] - sample['position'][2]) / flight_frames
        achieved = sample['blade_pitch']
        model_before = nominal_exit(previous - [0., speed[1]], achieved, speed[0], self.impact_restitution)[1]
        model_after = nominal_exit(incoming[1:] - [0., lift], achieved, pace, self.impact_restitution)[1]
        target = model_before + desired_rise - up
        tilt = float(np.clip((target - model_after) / max(.12, 1.7 * (pace - incoming[1])), -.12, .12))
        request = sample['desired_pitch'] + tilt
        return np.array([pace, lift]), float(np.clip(request - pitch, -.13, .13))

    def _stroke_goals(self, estimate):
        self._observe_reply()
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        tick = self.scene.tick
        self.servo.recalled_speed = None
        if self.reply is not None and (self.release == tick or (
                self.choice is not None and tick > self.choice['deadline'] + 3)):
            self.stroke_closed = True
        if (not self.retrieve or self.reply is None or self.stroke_closed or horizon is None
                or estimate is None or self.servo.drive is not None or estimate[1][1] >= -.025):
            return goal, lead, pitch, horizon
        position, velocity = estimate
        if self.choice is None and 3.5 <= horizon <= 7.0 and -.99 < position[1] < -.48:
            _, hit, incoming = self.flight.crossing(position, velocity, self.plan[2], self._surface_ratio())
            match = self.strokes.nearest(hit, incoming)
            if match is not None:
                sample, distance = match
                self.recalled, offset = self._retarget(sample, hit, incoming, pitch)
                self.choice = dict(tick=tick, deadline=float(tick + horizon), example=sample['source'],
                                   distance=distance, speed=self.recalled.tolist(), pitch_offset=offset)
        if self.choice is not None:
            self.servo.recalled_speed = self.recalled
            angle = float(np.clip(pitch + self.choice['pitch_offset'], -.02, .72))
            face_shift = .027 * np.array([0., np.cos(pitch) - np.cos(angle), np.sin(pitch) - np.sin(angle)])
            return goal + face_shift, lead + face_shift, angle, horizon
        return goal, lead, pitch, horizon

    def finish(self, reward):
        details = super().finish(reward)
        observed = None if self.serve_plan is None else optical_stroke(self.trace, self.serve_plan[0])
        if observed is not None and self.learn_strokes:
            self.strokes.samples.append(dict(observed, source=f'game-sample-{len(self.strokes.samples)}'))
        return dict(details, stroke_recall=dict(reply=self.reply, choice=self.choice, observed=observed))
