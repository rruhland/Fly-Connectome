"""Local temporal correlations with shared coefficients across image axes."""

import torch


class LocalLinearDynamics:
    def __init__(self):
        self.correlation = torch.eye(4)*.001
        self.cross = torch.zeros(4)
        self.coefficients = torch.zeros(4)
        self.samples = 0

    @torch.no_grad()
    def observe(self, history, displacement):
        self.correlation.add_(history @ history.T)
        self.cross.add_(history @ displacement)
        self.coefficients = torch.linalg.solve(self.correlation, self.cross)
        self.samples += 1

    @torch.no_grad()
    def predict(self, history):
        return self.coefficients @ history
