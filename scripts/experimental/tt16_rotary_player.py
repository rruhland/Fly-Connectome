"""Choose a first return depth using observed reply reach and public points."""

from dataclasses import dataclass

import numpy as np

from .tt6_continuous import PredictiveServo
from .tt13_place import depth_pitch, visible_landing
from .tt14_rotary_player import FlightCorrectedPlayer


DEPTHS = {'base': .76, 'shallow': .27, 'deep': 1.16}
CONTEXT_SCALE = np.array([.24, .13, .045, .045])


def second_approach(trace, contact):
    """Look for a far excursion followed by a coherent incoming intercept."""
    farthest = .32
    sightings = []
    for frame in trace:
        tick = frame['tick']
        if not contact + 8 <= tick <= contact + 71 or frame['ball'] is None:
            continue
        x, y, _ = frame['ball']
        farthest = max(farthest, y)
        estimate, horizon = frame['estimate'], frame['horizon']
        if (tick < contact + 19 or farthest < .36 or y > min(.52, farthest - .22)
                or estimate is None or estimate[1][1] > -.035 or horizon is None or not 0. < horizon < 26.):
            continue
        if sightings:
            last = sightings[-1]
            interval = tick - last['tick']
            dx = x - last['ball'][0]
            dy = y - last['ball'][1]
            if 0 < interval <= 5 and -.21 < dy / interval < -.025 and abs(dx) / interval < .050:
                first = last
                break
        sightings.append(frame)
    else:
        return None

    approach = [frame for frame in trace if first['tick'] <= frame['tick'] <= first['tick'] + 25
                and frame['horizon'] is not None and 0. < frame['horizon'] < 26.]
    closest = min(approach, key=lambda frame: abs(frame['horizon']))
    deadline = closest['tick'] + closest['horizon']
    x, _, z = closest['goal']
    lateral = np.clip(1. - abs(x) / .48, 0., 1.)
    height = np.clip(1. - abs(z - .29) / .27, 0., 1.)
    lead = np.clip((deadline - first['tick']) / 17., 0., 1.)
    return dict(tick=first['tick'], deadline=deadline, goal=closest['goal'],
                reach=float(.45 * lateral + .30 * height + .25 * lead))


def observed_outcome(trace, contact, reward):
    """Score the first returned ball from RGB and the public terminal reward."""
    reply = second_approach(trace, contact)
    first = visible_landing(trace, contact)
    second = None if reply is None else visible_landing(trace, reply['deadline'])
    score = (.22 * (first is not None or reply is not None)
             + .45 * (reply['reach'] if reply else 0.)
             + .20 * (second is not None) + .13 * (reward > 0.))
    return dict(first_landing=first, reply=reply, second_landing=second, reward=float(reward), value=score)


@dataclass
class FirstShotMemory:
    observations: list

    def values(self, context, local):
        scores = {}
        for mode in DEPTHS:
            points = [item for item in self.observations if item['mode'] == mode]
            value = np.asarray([item['outcome']['value'] for item in points])
            global_score = (1. + value.sum()) / (4. + len(value))
            if local and points:
                cues = np.asarray([item['context'] for item in points])
                distance = np.linalg.norm((cues - context) / CONTEXT_SCALE, axis=1)
                weight = np.exp(-.50 * distance**2)
                global_score = (weight @ value + 3. * global_score) / (3. + weight.sum())
            scores[mode] = float(global_score)
        return scores

    def choose(self, context, local):
        values = self.values(context, local)
        return max(values, key=values.get), values


class CenteringServo(PredictiveServo):
    """Add bounded lateral blade speed through the first collision window."""

    sweep = None

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if self.sweep is None or self.intercept is None or not 0. < self.intercept < self.horizon + 1.:
            return
        motion = np.diff(self.fit.pose(states)[:, 0])
        response = np.einsum('n,tmn->tmn', jac[0], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weight = 2.4 * np.exp(-.5 * ((times - self.intercept) / 1.25)**2)
        design = (response * weight[:, None, None]).reshape(len(times), -1)
        normal += design.T @ design
        rhs += design.T @ ((self.sweep - motion) * weight)


class FirstShotPlayer(FlightCorrectedPlayer):
    """Change only the first rival-served return, retaining own serve and recovery."""

    def __init__(self, *args, shot_mode, shot_memory, **kwargs):
        self.shot_mode = shot_mode
        self.shot_memory = shot_memory
        super().__init__(*args, mode='recenter', **kwargs)

    def reset(self):
        super().reset()
        if self.shot_mode == 'center':
            self.receiving_servo = CenteringServo(self.receiving_model, self.timing_memory, learn=self.adjust)
        self.first_shot = None
        self.first_approach = None

    def _stroke_goals(self, estimate):
        self.first_approach = estimate
        goals = super()._stroke_goals(estimate)
        if self.shot_mode == 'center':
            shot = self.first_shot
            armed = (shot is not None and goals[-1] is not None and self.scene.tick <= shot['deadline'] + 5.)
            self.receiving_servo.sweep = shot['sweep'] if armed else None
        return goals

    def _strike_pitch(self, position, incoming):
        anchor = super()._strike_pitch(position, incoming)
        tick = self.scene.tick
        if self.visual_role != 'rival' or self.first_approach is None or tick > 46:
            return anchor
        observed, speed = self.first_approach
        remaining, _, _ = self.flight.crossing(observed, speed, position[1], self._surface_ratio())
        if self.first_shot is None and -.95 < observed[1] < -.32 and 3. < remaining < 22. and speed[1] < -.035:
            context = np.array([position[0], position[2], *incoming[1:]])
            mode, values = self.shot_mode, None
            if mode in ('context', 'global'):
                mode, values = self.shot_memory.choose(context, local=mode == 'context')
            self.first_shot = dict(tick=tick, mode=mode, deadline=float(tick + remaining), context=context.tolist(),
                                   values=values)
            if mode == 'center':
                offset = self.receiving_model.ready[0] - position[0]
                incoming_vx = incoming[0]
                # A first approximation to lateral ball motion carries the
                # blade's tangential speed weakly into the ball departure.
                desired_vx = offset / 24.
                sweep = incoming_vx + (desired_vx - incoming_vx) / .35
                self.first_shot['sweep'] = float(np.clip(sweep, -.055, .055)) if abs(offset) > .10 else None
        shot = self.first_shot
        if shot is None or tick > shot['deadline'] + 5.:
            return anchor
        if 0. <= remaining < 3.:
            shot['deadline'] = float(tick + remaining)
        if shot['mode'] in ('base', 'center'):
            return anchor
        pitch = depth_pitch(position, incoming, self.stroke_speed, self.impact_restitution,
                            self.memory.correction(), DEPTHS[shot['mode']], anchor)
        self.prediction = incoming[1:].copy(), pitch, self.stroke_speed
        return pitch

    def finish(self, reward):
        details = super().finish(reward)
        shot = self.first_shot
        if shot is not None:
            shot['outcome'] = observed_outcome(self.trace, shot['deadline'], reward)
        return dict(details, first_shot=shot)
