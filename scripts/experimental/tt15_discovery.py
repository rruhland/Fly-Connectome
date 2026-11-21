"""Learn new local motor maps during hidden-source search and recall them from RGB."""
from collections import deque

import numpy as np

from .tt15_context import ContextPlayer, MotorEvidence


class DiscoveryMemory(MotorEvidence):
    def __init__(self, priors):
        super().__init__(priors)
        self.samples = deque(maxlen=56)

    def suggest(self, motor, position, active, tick):
        effective = motor.requested + (motor.curve - 1.) @ motor.history
        self.samples.append((effective.copy(), position))
        if tick % 8 or len(self.samples) < 40:
            return active
        u, x = map(np.asarray,zip(*self.samples))
        times = np.arange(12,len(x),2)
        motion = x[times] - x[times-6]
        split = len(times)//2
        # Fit the geometry on earlier motion. Only subsequent motion can
        # validate a new model. Choose delay on the fit half in both cases.
        best = []
        for prior in self.models:
            best.append(self._fit_delay(prior,u,times,motion,split,False))
        novel = self._fit_delay(self.models[active],u,times,motion,split,True)
        costs = [b[1] for b in best] + [novel[1]+.002]
        winner = int(np.argmin(costs))
        if winner == active or costs[winner]+.006 >= .79*costs[active]:
            return active
        candidate = best[winner] if winner < len(best) else novel
        if candidate[0]+.003 >= best[active][0]:
            return active
        if winner == len(best):
            self.models = np.concatenate([self.models,candidate[2][None]])
        self.samples.clear()
        return winner

    @staticmethod
    def _fit_delay(prior, u, times, motion, split, adapt):
        candidates = []
        for lag in (0,2,4,6):
            delta = u[times-lag] - u[times-lag-6]
            jac = prior
            if adapt:
                a = delta[:split]
                jac = np.linalg.solve(a.T@a + .03*np.eye(a.shape[1]),
                                      a.T@motion[:split] + .03*prior.T).T
            errors = np.linalg.norm(delta@jac.T-motion,axis=1)
            candidates.append((float(errors[:split].mean()),float(errors[split:].mean()),jac))
        return min(candidates,key=lambda b:b[0])


class DiscoveryPlayer(ContextPlayer):
    def __init__(self, prior, discover=True):
        super().__init__([prior], 'selective' if discover else 'reward_only')
        self.motor.probe = True
        self.evidence = DiscoveryMemory([prior]) if discover else MotorEvidence([prior])
