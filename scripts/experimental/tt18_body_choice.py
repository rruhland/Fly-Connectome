"""Retain a working action model while alternative fits learn from scored movement."""
from dataclasses import replace

import numpy as np

from .tt18_online import BodyLearning, OnlineBodyPlayer


class BodyChoice:
    """Choose between carried, updated and restarted geometry between points."""

    def __init__(self, prior, *, seed):
        fresh = np.zeros_like(prior.coefficients)
        fresh[0] = prior.home
        cold = replace(prior, coefficients=fresh, pitch_controllable=False)
        candidates = (('fixed',prior,False),('warm', prior, True),('relearn',cold, True))
        self.experts = {key:BodyLearning(model,enabled=train,seed=seed,directed=False)
                        for key,model,train in candidates}
        self.active = 'fixed'
        self.cost = dict.fromkeys(self.experts, .08)
        self.choices = []
        self.last_state = None
        self.last_pose = None
        self.window = []
        self.sightings = 0

    @property
    def model(self):
        return self.experts[self.active].model

    @property
    def updates(self):
        return self.experts['warm'].updates

    @property
    def uncertainty(self):
        return self.experts[self.active].uncertainty

    def select(self):
        # Only adopt challengers after predicting unseen commanded motion more accurately.
        if self.window:
            point_errors = np.median(self.window,axis=0)
            for key,error in zip(self.experts,point_errors,strict=True):
                self.cost[key] = .55*self.cost[key] + .45*float(error)
            best = min(self.cost,key=self.cost.get)
            if self.cost[best] + .005 < self.cost[self.active]:
                self.active = best
        self.choices.append(dict(selected=self.active,cost=self.cost.copy(),observations=len(self.window)))

    def begin_point(self):
        self.window = []
        self.last_state = self.last_pose = None
        for expert in self.experts.values():
            expert.begin_point()

    def observe(self, state, pose):
        output = self.model.pose(state)[:3]
        error = float(np.linalg.norm(output - pose[:3]))
        if self.last_state is None or np.linalg.norm(state-self.last_state) > .28:
            if self.last_state is not None:
                observed = pose[:3] - self.last_pose
                residuals = []
                for expert in self.experts.values():
                    predicted = expert.model.pose(state)[:3] - expert.model.pose(self.last_state)[:3]
                    residuals.append(np.linalg.norm(observed-predicted))
                self.window.append(residuals)
            self.last_state = state.copy()
            self.last_pose = pose[:3].copy()
        for expert in self.experts.values():
            expert.observe(state,pose)
        self.sightings += 1
        return error

    def explore(self, command, position, deadline):
        # Scored strokes supply the identifying commands, with no additional dither.
        return command

    def state(self):
        return dict(experts={key:e.state() for key,e in self.experts.items()},
                    active=self.active,cost=self.cost,choices=self.choices,sightings=self.sightings,
                    window=np.asarray(self.window).tolist())

    @classmethod
    def restore(cls, state):
        experts = {key:BodyLearning.restore(e) for key,e in state['experts'].items()}
        result = cls(experts['fixed'].prior,seed=0)
        result.experts = experts
        result.active = state['active']
        result.cost = state['cost'].copy()
        result.window = state['window'].copy()
        result.choices = state['choices'].copy()
        result.sightings = state['sightings']
        return result


class ChoicePlayer(OnlineBodyPlayer):
    def _make_servo(self):
        self.body_learning.select()
        self.adapter = self.body_learning.model
        return super()._make_servo()
