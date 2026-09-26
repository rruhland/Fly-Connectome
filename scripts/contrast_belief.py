"""Opt-in probabilistic contrast transitions, separate from event amplitude."""

import torch


class ContrastBelief:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.reset_state()

    def reset_state(self):
        self.probabilities = torch.zeros(3, self.height, self.width)
        self.probabilities[1] = 1.

    @property
    def mean(self):
        return self.probabilities[2]-self.probabilities[0]

    @torch.no_grad()
    def step(self, credible, *, absolute=None):
        off, on = credible
        q_on, q_off = on*(1-off), off*(1-on)
        negative, zero, positive = self.probabilities
        up = torch.stack((torch.zeros_like(zero), negative, zero+positive))
        down = torch.stack((negative+zero, positive, torch.zeros_like(zero)))
        self.probabilities = ((1-q_on-q_off)*self.probabilities +
                              q_on*up + q_off*down)
        if absolute is not None:
            observation = torch.stack(((-absolute).clamp(min=0.),
                                       1-absolute.abs(),
                                       absolute.clamp(min=0.)))
            self.probabilities.lerp_(observation, .35)


class BeliefObserver:
    """Preserve the existing observer; replace only its exported contrast."""

    def __init__(self, observer):
        self.observer = observer
        self.belief = ContrastBelief(height=observer.height, width=observer.width)

    def reset_state(self):
        self.observer.reset_state()
        self.belief.reset_state()

    @property
    def fast(self):
        return self.observer.fast

    @property
    def slow(self):
        return self.observer.slow

    @property
    def contrast(self):
        return self.belief.mean

    @torch.no_grad()
    def step(self, raw, *, absolute=None):
        filtered, features = self.observer.step(raw, absolute=absolute)
        self.belief.step(filtered, absolute=absolute)
        return filtered, features
