"""Learn repeatable lateral and vertical strike errors from stereo observations."""

import numpy as np

from .tt6_continuous import ContinualPlayer


class StrikeMemory:
    """Retain correction estimates for each supported public action width."""

    limits = np.array([.028, .026])

    def __init__(self, contexts=None):
        self.contexts = {} if contexts is None else {
            int(width): (np.asarray(record['correction'], dtype=float), record['points'])
            for width, record in contexts.items()
        }

    def correction(self, width):
        return self.contexts.get(width, (np.zeros(2), 0))[0].copy()

    def observe(self, width, error, reward):
        previous, points = self.contexts.get(width, (np.zeros(2), 0))
        gain = (.10 if reward > 0 else .22) / np.sqrt(1. + points / 12.)
        update = gain * np.clip(error, -.04, .04)
        change = np.clip(previous + update, -self.limits, self.limits)
        self.contexts[width] = (change, points + 1)
        return change.copy()

    def state(self):
        return dict(schema='tt7-strike-memory-v1', contexts={
            str(width): dict(correction=correction.tolist(), points=points)
            for width, (correction, points) in self.contexts.items()
        })

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'tt7-strike-memory-v1':
            raise ValueError('unsupported strike memory')
        return cls(state['contexts'])


class VisualStrikePlayer(ContinualPlayer):
    """Correct consistent blade shortfall without observing true impact or landing."""

    def __init__(self, adapter, prior, factory, response, timing, *, strike_memory):
        self.strike_memory = strike_memory
        super().__init__(adapter, prior, factory, response, timing, predictive=True)

    def reset(self):
        super().reset()
        self.correction = self.strike_memory.correction(self.adapter.action_size)
        self.strike_errors = []
        self.current_target = None
        self.arrived = False

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        self.current_target = None
        if horizon is None:
            return goal, lead, pitch, horizon
        if -.9 <= horizon <= 2.5 and not self.arrived:
            self.current_target = goal[[0, 2]], horizon
        if horizon < -.9 or self.outgoing:
            self.arrived = True
        lateral, vertical = self.correction
        offset = np.array([lateral, 0., vertical]) * np.clip((19. - horizon) / 7., 0., 1.)
        goal, lead = goal + offset, lead + offset
        goal[2], lead[2] = max(goal[2], .054), max(lead[2], .054)
        return goal, lead, pitch, horizon

    def update(self, frames, preceding=None):
        action = super().update(frames, preceding)
        actor = getattr(self.scene.tracker, 'actor', None)
        if self.current_target is not None and actor is not None and actor.observed:
            target, horizon = self.current_target
            error = target - np.asarray(actor.position)[[0, 2]]
            if np.max(np.abs(error)) < .11:
                self.strike_errors.append((float(horizon), error))
        return action

    def finish(self, reward):
        details = super().finish(reward)
        after = self.correction
        accepted = None
        if len(self.strike_errors) >= 2:
            errors = np.asarray([error for _, error in self.strike_errors])
            consistent = np.ptp(errors, axis=0) < .045
            if consistent.any():
                accepted = np.where(consistent, np.median(errors, axis=0), 0.)
                after = self.strike_memory.observe(self.adapter.action_size, accepted, reward)
        details['strike_compensation'] = dict(before=self.correction.tolist(), after=after.tolist(),
                                              accepted=None if accepted is None else accepted.tolist(),
                                              errors=[dict(horizon=horizon, error=error.tolist())
                                                      for horizon, error in self.strike_errors])
        return details
