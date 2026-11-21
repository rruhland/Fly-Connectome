"""Search and replay approach-and-swing commands using camera tracks and point score."""

from dataclasses import asdict, dataclass

import numpy as np


PREP_TICKS = 7
SWING_TICKS = 5
ARM_COMMANDS = 5


def _local_vector(frames, tick, field, offsets, dt):
    indices = [tick + offset for offset in offsets if 0 <= tick + offset < len(frames)]
    values = [(i, frames[i][field]) for i in indices if frames[i][field] is not None]
    if len(values) < 2:
        return None
    start, end = values[0], values[-1]
    return (np.asarray(end[1]) - np.asarray(start[1])) / ((end[0] - start[0]) * dt)


def optical_strokes(frames, dt, *, server, reward):
    """Locate outgoing reversals near the observed blade without evaluator contacts."""
    observations = []
    for tick, sample in enumerate(frames):
        ball, blade, pitch = sample[:3]
        if ball is None or blade is None or ball[1] > -.28:
            continue
        distance = float(np.linalg.norm(np.subtract(ball, blade)))
        if distance > .121:
            continue
        incoming = _local_vector(frames, tick, 0, (-4, -3, -2, -1), dt)
        outgoing = _local_vector(frames, tick, 0, (1, 2, 3, 4), dt)
        if (incoming is None or outgoing is None or not 0.36 < outgoing[1] < 10.
                or np.linalg.norm(outgoing) > 11.):
            continue
        is_toss = abs(incoming[1]) < .45 and incoming[2] < -.75
        if (server == 1 and not is_toss and incoming[1] > -.80) or (server == -1 and incoming[1] > -.80):
            continue
        if outgoing[1] - incoming[1] < 1.1:
            continue
        velocity = _local_vector(frames, tick, 1, (-4, -3, -2, -1, 0), dt)
        observations.append(dict(tick=tick, server=server, reward=reward, near_m=distance,
                                 ball=ball, blade=blade, pitch=pitch, incoming=incoming.tolist(),
                                 outgoing=outgoing.tolist(), blade_velocity=None if velocity is None else velocity.tolist()))
    separate = []
    for row in observations:
        if not separate or row['tick'] - separate[-1]['tick'] > 4:
            separate.append(row)
        elif row['near_m'] < separate[-1]['near_m']:
            separate[-1] = row
    return separate


def gantry_style(rows, dt):
    """Learn scoring-stroke pitch and launch slope from archived optical point tapes."""
    examples = []
    for row in rows:
        if row['public_reward'] > 0:
            examples.extend(optical_strokes(row['optical'], dt, server=row['server'], reward=row['public_reward']))
    styles = {}
    for server in (-1, 1):
        samples = [row for row in examples if row['server'] == server and row['pitch'] is not None]
        if samples:
            pitches = [row['pitch'] for row in samples]
            slopes = [row['outgoing'][2] / row['outgoing'][1] for row in samples]
            approach = [row['blade_velocity'] for row in samples if row['blade_velocity'] is not None]
            styles[server] = dict(n=len(samples), pitch=float(np.median(pitches)),
                                  slope=float(np.median(slopes)),
                                  velocity=None if not approach else np.median(approach, axis=0).tolist())
    return styles, examples


@dataclass(frozen=True)
class Stroke:
    ready: tuple[float, ...]
    drive: tuple[float, ...]
    lead: float

    @classmethod
    def empty(cls):
        return cls((0.,) * ARM_COMMANDS, (0.,) * ARM_COMMANDS, .12)

    def vector(self):
        return np.array((*self.ready, *self.drive, self.lead))

    @classmethod
    def from_vector(cls, vector):
        ready = tuple(np.clip(vector[:ARM_COMMANDS], -.61, .61).tolist())
        drive = tuple(np.clip(vector[ARM_COMMANDS:2 * ARM_COMMANDS], -.79, .79).tolist())
        return cls(ready, drive, float(np.clip(vector[-1], -.030, .23)))


@dataclass(frozen=True)
class Trial:
    stroke: Stroke
    quality: float
    reward: float
    contact: bool


class StrokeSearch:
    """Rank metered point attempts and mutate the strongest observed strokes."""

    def __init__(self, seed, *, source_style=None):
        self.rng = np.random.default_rng(seed)
        self.style = source_style or {}
        self.archive = {1: [], -1: []}
        self.probes = []
        self.preferred_serve = None

    def prepare_stroke(self, server, rank):
        style = self.style.get(server)

        def desirability(probe):
            velocity = np.asarray(probe['velocity'])
            score = velocity[1] - .4 * abs(velocity[0]) - .25 * abs(velocity[2])
            if style is not None and style['velocity'] is not None:
                score -= .35 * np.linalg.norm(velocity - style['velocity'])
            return score

        catalogue = sorted((probe for probe in self.probes if probe['velocity'] is not None),
                           key=desirability, reverse=True)
        if rank >= len(catalogue):
            return None
        drive = np.asarray(catalogue[rank]['action'][:ARM_COMMANDS])
        ready = -.32 * drive if server == 1 else -.12 * drive
        return Stroke.from_vector(np.r_[ready, drive, .082 + .022 * (rank % 3)])

    def _rank(self, history):
        trials = {}
        for trial in history:
            trials.setdefault(trial.stroke, []).append(trial)

        def score(group):
            return float(np.mean([row.quality for row in group]) - .32 / np.sqrt(len(group)))

        return [max(group, key=lambda row: row.quality) for group in sorted(trials.values(), key=score, reverse=True)]

    def choose(self, server, *, train):
        if server == 1 and not train and self.preferred_serve is not None:
            return self.preferred_serve
        history = self.archive[server]
        if not history:
            return Stroke.empty()
        ranked = self._rank(history)
        if not train:
            return ranked[0].stroke
        index = len(history)
        directed = self.prepare_stroke(server, index - 1) if 1 <= index <= 8 else None
        if directed is not None:
            return directed
        if index % 4 == 0:
            return ranked[0].stroke
        base = ranked[self.rng.integers(min(len(ranked), 3))].stroke.vector()
        scale = 1. if not ranked[0].contact else .57
        deviation = np.array((.08,) * ARM_COMMANDS + (.24,) * ARM_COMMANDS + (.030,))
        sample = base + scale * self.rng.normal(size=len(base)) * deviation
        return Stroke.from_vector(sample)

    def finish(self, stroke, server, reward, observations, dt, swing_tick, *, train):
        contact = optical_strokes(observations, dt, server=server, reward=reward)
        start = PREP_TICKS - 2 if swing_tick is None else max(PREP_TICKS - 2, swing_tick - 3)
        end = len(observations) if swing_tick is None else min(len(observations), swing_tick + SWING_TICKS + 6)
        closest = None
        for tick in range(start, end):
            ball, blade, pitch = observations[tick][:3]
            if ball is None or blade is None:
                continue
            if server == 1 and tick < 9:
                continue
            distance = float(np.linalg.norm(np.subtract(ball, blade)))
            if closest is None or distance < closest[0]:
                closest = distance, pitch, tick
        quality = -1. if closest is None else -4. * closest[0]
        style = self.style.get(server)
        if style is not None and closest is not None and closest[1] is not None:
            quality -= .22 * abs(closest[1] - style['pitch'])
        if closest is not None:
            blade_motion = _local_vector(observations, closest[2], 1, (-4, -3, -2, -1, 0), dt)
            if blade_motion is not None:
                quality += .22 * float(np.clip(blade_motion[1], -.5, 1.))
        if contact:
            outgoing = contact[0]['outgoing']
            quality += .79 + .09 * min(3., outgoing[1])
            if style is not None:
                quality -= .11 * min(2., abs(outgoing[2] / outgoing[1] - style['slope']))
        crossings = self._ball_crossings(observations)
        if crossings and server == 1:
            quality += min(2., .6 * len(crossings))
        if reward > 0. and (contact or server == 1):
            quality += 1.6
        if train:
            self.archive[server].append(Trial(stroke, quality, float(reward), bool(contact)))
        return dict(quality=float(quality), nearest_m=None if closest is None else closest[0],
                    contact=contact, optical_bounces=crossings)

    @staticmethod
    def _ball_crossings(observations):
        visible = [(i, sample[0]) for i, sample in enumerate(observations) if sample[0] is not None]
        bounces = []
        for (before_tick, before), (tick, point), (after_tick, after) in zip(visible, visible[1:], visible[2:]):
            if not (after_tick - before_tick <= 6 and after[1] - before[1] > .025 and -.045 < point[2] < .105):
                continue
            if min(before[2], after[2]) - point[2] < .018:
                continue
            if not bounces or tick - bounces[-1]['tick'] > 4:
                bounces.append(dict(tick=tick, position=point))
        return bounces

    def prefer_placement(self, practice):
        """Select a serving motion from practice images of the two court halves."""
        scores = {}
        for row in practice:
            if row['server'] != 1:
                continue
            stroke = _restore_stroke(row['stroke'])
            bounces = row['detail']['optical_bounces']
            own = any(bounce['position'][1] < 0. for bounce in bounces)
            away = any(bounce['position'][1] > 0. for bounce in bounces)
            grade = 2. * away + 2. * (own and away) + .25 * own + .5 * row['public_reward']
            scores.setdefault(stroke, []).append(grade)
        if scores:
            self.preferred_serve = max(scores, key=lambda stroke: np.mean(scores[stroke]) - .35 / np.sqrt(len(scores[stroke])))

    def state(self):
        return dict(archive={str(server): [asdict(row) for row in history]
                             for server, history in self.archive.items()},
                    style={str(server): row for server, row in self.style.items()},
                    probes=self.probes,
                    preferred_serve=None if self.preferred_serve is None else asdict(self.preferred_serve),
                    rng=self.rng.bit_generator.state)

    @classmethod
    def restore(cls, state):
        self = cls(0, source_style={int(server): row for server, row in state['style'].items()})
        self.archive = {int(server): [Trial(_restore_stroke(row['stroke']), row['quality'], row['reward'], row['contact'])
                                     for row in history] for server, history in state['archive'].items()}
        self.probes = state.get('probes', [])
        if state.get('preferred_serve') is not None:
            self.preferred_serve = _restore_stroke(state['preferred_serve'])
        self.rng.bit_generator.state = state['rng']
        return self


def _restore_stroke(data):
    return Stroke(tuple(data['ready']), tuple(data['drive']), data['lead'])


def track_coordinates(track):
    return None if track is None or not track.observed else list(track.position)


def lateral_gain(rows):
    """Measure centimeters of lateral blade motion per accumulated pan command."""
    samples = []
    for row in rows:
        if len(row['optical']) <= 6:
            continue
        first = row['optical'][0][1]
        final = row['optical'][6][1]
        pan = sum(command[0] for command in row['actions'][:6])
        if first is not None and final is not None and abs(pan) >= .12:
            samples.append((final[0] - first[0]) / pan)
    if len(samples) < 8 or not .008 <= abs(np.median(samples)) <= .08:
        raise ValueError('insufficient optical evidence of a lateral response')
    return float(np.median(samples)), len(samples)


class StrokePoint:
    """Choose strike time from the visible ball, then execute the stored rate burst."""

    def __init__(self, stroke, tracker, server, dt, *, forecast=None):
        self.stroke = stroke
        self.tracker = tracker
        self.server = server
        self.dt = dt
        self.forecast = forecast
        self.observations = []
        self.swing_tick = None

    def observe(self, frames):
        if frames is None:
            self.observations.append([None, None, None])
            return
        self.tracker.update(frames)
        self.observations.append([track_coordinates(self.tracker.ball), track_coordinates(self.tracker.actor),
                                  self.tracker.actor_pitch])

    def _time_to_blade(self, tick):
        indices = [i for i in range(tick, max(-1, tick - 3), -1) if self.observations[i][0] is not None]
        blade = next((sample[1] for sample in reversed(self.observations[-4:]) if sample[1] is not None), None)
        if not indices or blade is None:
            return None
        current = indices[0]
        ball = self.observations[current][0]
        elapsed = (tick - current) * self.dt
        velocity = _local_vector(self.observations, current, 0, (-4, -3, -2, -1, 0), self.dt)
        if velocity is None:
            return None
        if self.forecast is not None:
            history = [(i, sample[0]) for i, sample in enumerate(self.observations[:current + 1])]
            horizons = np.arange(1, 11)
            guesses = self.forecast.predict(history, horizons, self.server)
            if guesses is not None:
                axis = 2 if self.server == 1 else 1
                compatible = np.flatnonzero(guesses[:, axis] <= blade[axis])
                if len(compatible):
                    return float(horizons[compatible[0]] * self.dt - elapsed)
        if self.server == -1:
            return (blade[1] - ball[1]) / velocity[1] - elapsed if velocity[1] < -.55 else None
        vertical = velocity[2] - .5 * 9.81 * min(4, tick) * self.dt
        discriminant = vertical**2 + 2. * 9.81 * (ball[2] - blade[2])
        if discriminant < 0.:
            return None
        return float((vertical + np.sqrt(discriminant)) / 9.81 - elapsed)

    def command(self):
        tick = len(self.observations) - 1
        if tick < PREP_TICKS:
            return (*self.stroke.ready, 0.)
        if self.swing_tick is None:
            arrival = self._time_to_blade(tick)
            if arrival is not None and -.045 <= arrival <= self.stroke.lead:
                self.swing_tick = tick
        if self.swing_tick is not None and tick < self.swing_tick + SWING_TICKS:
            return (*self.stroke.drive, 0.)
        return (0.,) * (ARM_COMMANDS + 1)


class RallyStrokePoint(StrokePoint):
    """Re-arm the receiver after each swing using the command integral and incoming RGB."""

    def __init__(self, stroke, receive, tracker, server, dt, *, forecast=None, rearm=True, pan_gain=None):
        super().__init__(stroke, tracker, server, dt, forecast=forecast)
        self.receive = receive
        self.command_position = np.zeros(ARM_COMMANDS)
        self.swing_ticks = []
        self.returning = False
        self.departed = False
        self.guard_until = 0
        self.rearm = rearm
        self.pan_gain = pan_gain
        self.pan_offset = 0.

    def _separated(self, tick):
        ball, blade, _ = self.observations[-1]
        if ball is None or blade is None:
            return False
        velocity = _local_vector(self.observations, tick, 0, (-4, -3, -2, -1, 0), self.dt)
        return velocity is not None and ball[1] - blade[1] > .26 and velocity[1] > .30

    def _time_to_blade(self, tick):
        if self.returning and self.pan_gain is not None:
            recent = [index for index in range(tick, max(-1, tick - 2), -1)
                      if self.observations[index][0] is not None]
            if not recent:
                return None
            current = recent[0]
            x, _, z = self.observations[current][0]
            speed = _local_vector(self.observations, current, 0, (-4, -3, -2, -1, 0), self.dt)
            if speed is None or not -.04 < z < .64 or abs(x) > .33 or not -7. < speed[1] < -.55:
                return None
        return super()._time_to_blade(tick)

    def _aim(self, tick, command):
        if self.pan_gain is None or self.server != -1 or self.swing_tick is not None or tick < PREP_TICKS:
            return command
        recent = [(index, self.observations[index]) for index in range(tick, max(-1, tick - 2), -1)
                  if all(value is not None for value in self.observations[index][:2])]
        if not recent:
            return command
        current, (ball, blade, _) = recent[0]
        speed = _local_vector(self.observations, current, 0, (-4, -3, -2, -1, 0), self.dt)
        if speed is None or not -7. < speed[1] < -.6 or abs(speed[0]) > 3.5:
            return command
        horizon = (blade[1] - ball[1]) / speed[1]
        if not .09 < horizon < .76 or not -.02 < ball[2] < .7:
            return command
        x = ball[0] + np.clip(horizon * speed[0], -.11, .11)
        if abs(x) > .30:
            return command
        change = float(np.clip(.34 * (x - blade[0]) / self.pan_gain, -.58, .58))
        total = float(np.clip(self.pan_offset + change, -4., 4.))
        correction = total - self.pan_offset
        self.pan_offset = total
        return (command[0] + correction, *command[1:])

    def command(self):
        tick = len(self.observations) - 1
        if self.swing_tick is not None:
            self.departed |= self._separated(tick)
            if self.rearm and tick >= self.swing_tick + SWING_TICKS + 2:
                self.server = -1
                self.stroke = self.receive
                self.swing_tick = None
                self.returning = True
                self.guard_until = tick + 3
                self.pan_offset = 0.
        if self.returning:
            self.departed |= self._separated(tick)
            target = PREP_TICKS * np.asarray(self.receive.ready)
            target[0] += self.pan_offset
            correction = target - self.command_position
            if tick < self.guard_until or np.max(np.abs(correction)) > .16 or not self.departed:
                command = (*np.clip(correction, -.76, .76), 0.)
            else:
                command = self._aim(tick, super().command())
        else:
            command = self._aim(tick, super().command())
        self.command_position += command[:ARM_COMMANDS]
        if self.swing_tick == tick:
            self.swing_ticks.append(tick)
            self.returning = False
            self.departed = False
        return command
