"""Change the sideways placement of an otherwise registered serve."""

from dataclasses import asdict, dataclass, field

import numpy as np

from .tt8_service_drive import ServiceDriveServo
from .tt9_precontact_registration import RegistrationPlayer


SWEEPS = (0., -.052, .052)


class SweepServo(ServiceDriveServo):
    sweep = 0.

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        super()._strike_objective(normal, rhs, states, effect, jac, targets)
        if self.sweep == 0. or self.intercept is None:
            return
        times = np.arange(1, self.horizon + 1)
        response = np.einsum('n,tmn->tmn', jac[0], effect)
        # Bias the sideways trajectory around its inherited impact position.
        # The center at the planned impact, forward motion and height are retained as targets.
        offsets = np.clip(self.sweep * (times - self.intercept), -.095, .095)
        rows = (response * self.scale[:, None, None]).reshape(len(times), -1)
        rhs += rows.T @ (offsets * self.scale)
        if 0. < self.intercept < self.horizon + 1.:
            weight = 2.8 * np.exp(-.5 * ((times[1:] - .5 - self.intercept) / 1.25)**2)
            velocity_rows = (np.diff(response, axis=0) * weight[:, None, None]).reshape(len(weight), -1)
            motion = np.diff(self.fit.pose(states)[:, 0])
            normal += velocity_rows.T @ velocity_rows
            rhs += velocity_rows.T @ ((self.sweep - motion) * weight)


def optical_exchange(trace, deadline):
    """Extract coherent outbound and inbound motion from stereo ball positions."""
    points = [(row['tick'], np.array(row['ball'])) for row in trace
              if row['ball'] is not None and row['tick'] >= deadline - 2.]
    departure = []
    far_side = None
    reply = None
    returned = None
    for (earlier, start), (tick, end) in zip(points, points[1:]):
        interval = tick - earlier
        if not 1 <= interval <= 5:
            continue
        motion = (end - start) / interval
        if (-1.58 < start[1] < -.40 and end[1] < -.30 and motion[1] > .045
                and tick <= deadline + 19.):
            departure.append(float(motion[0] / motion[1]))
        if far_side is None and start[1] >= .45 and end[1] >= .55 and motion[1] > .025:
            far_side = tick
        if (far_side is not None and reply is None and tick > far_side and start[1] > .65
                and motion[1] < -.065 and end[1] >= -.80):
            reply = tick
        if (reply is not None and returned is None and tick > reply and start[1] < -.42
                and motion[1] > .038):
            returned = tick
    # The outward follow-up should travel through the center of the table.
    through = returned is not None and any(tick > returned and ball[1] > .15 for tick, ball in points)
    return dict(outbound_slope=None if not departure else float(np.median(departure)),
                far_side=far_side, reply=reply, follow_up=bool(through))


@dataclass
class StrategyMemory:
    samples: list = field(default_factory=list)

    def observe(self, context, choice, reward, exchange):
        if reward == 0.:
            return
        self.samples.append(dict(context=list(context), choice=choice, reward=reward,
                                 effect=reward + .50 * exchange['follow_up']))

    def values(self, context, *, contextual=True):
        prior = (0., -.15, -.15)
        estimates = np.array(prior)
        for choice, initial in enumerate(prior):
            samples = [row for row in self.samples if row['choice'] == choice]
            if not samples:
                continue
            target = np.array([row['effect'] for row in samples])
            if contextual:
                design = np.asarray([row['context'] for row in samples])
                coefficients = np.linalg.solve(design.T @ design + np.diag([2., 6., 6.]),
                                               design.T @ target + [2. * initial, 0., 0.])
                estimates[choice] = context @ coefficients
            else:
                estimates[choice] = (target.sum() + 2. * initial) / (len(target) + 2.)
        return estimates

    def state(self):
        return asdict(self)


class StrategyPlayer(RegistrationPlayer):
    def __init__(self, *args, strategy, placement=None, learn_strategy=False, **kwargs):
        self.strategy = strategy
        self.placement = placement
        self.learn_strategy = learn_strategy
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return SweepServo(self.adapter, self.timing_memory, learn=self.adjust)

    def reset(self):
        super().reset()
        self.strategy_choice = None

    def _stroke_goals(self, estimate):
        goals = super()._stroke_goals(estimate)
        self.servo.sweep = 0.
        if self.servo.drive is not None and self.features is not None:
            if self.strategy_choice is None:
                self.strategy_choice = (int(np.argmax(self.strategy.values(self.features)))
                                        if self.placement is None else self.placement)
            self.servo.sweep = SWEEPS[self.strategy_choice]
        return goals

    def finish(self, reward):
        result = super().finish(reward)
        result['serve_strategy'] = None
        if self.strategy_choice is not None:
            exchange = optical_exchange(self.trace, self.serve_plan[0])
            if self.learn_strategy:
                self.strategy.observe(self.features, self.strategy_choice, reward, exchange)
            result['serve_strategy'] = dict(context=self.features.tolist(), choice=self.strategy_choice,
                                            sweep=SWEEPS[self.strategy_choice], **exchange)
        return result
