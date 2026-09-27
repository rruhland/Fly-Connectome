"""Reuse local context-dependent selectivity for observed temporal associations."""

import torch

from local_metric_association import LocalMetricAssociation
from local_motion_dynamics import LocalMotionDynamics


class ContextualMotionDynamics(LocalMotionDynamics):
    @torch.no_grad()
    def predict(self, history):
        if not len(self.keys):
            return history[-1].clone()
        key, basis, scale = self.encode(history)
        memory = LocalMetricAssociation(dimensions=8)
        memory.keys, memory.values = self.keys, self.values
        return basis.T @ memory.predict(key)*scale
