"""Local predictive-error credit selects among generic future hypotheses."""

import torch

from local_metric_association import LocalMetricAssociation
from local_motion_dynamics import LocalMotionDynamics


class ForecastCompetence:
    def __init__(self, base, horizon):
        self.base = base
        self.horizon = horizon
        self.risk = LocalMetricAssociation(dimensions=8, outputs=4)

    @property
    def keys(self):
        return self.base.keys

    def candidates(self, history):
        replay = sum((history[-2+(step % 2)] for step in range(self.horizon)), torch.zeros(2))
        return torch.stack((self.base.predict(history), self.horizon*history.mean(0),
                            replay, self.horizon*history[-1]))

    @torch.no_grad()
    def predict_with_credit(self, history):
        key, _, scale = LocalMotionDynamics.encode(history)
        candidates = self.candidates(history)
        selected = int(self.risk.predict(key).argmin()) if len(self.risk.keys) else 0
        return candidates[selected], dict(key=key.clone(), scale=scale.clone(),
                                          candidates=candidates.clone())

    def predict(self, history):
        return self.predict_with_credit(history)[0]

    @torch.no_grad()
    def observe(self, history, displacement, *, credit):
        error = ((credit['candidates']-displacement)/credit['scale']).square().sum(1)
        self.risk.observe(credit['key'], error)
        self.base.observe(history, displacement)
