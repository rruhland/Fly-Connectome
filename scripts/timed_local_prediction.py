"""Experimental causal local timing comparator for one visual target."""
import math

import torch

from frame_prediction import FramePrediction


class TimedLocalPrediction(FramePrediction):
    """Gate an issued forecast and its local credit; keep recurrent neural drive."""

    def __init__(self, network, config, *, target, **kwargs):
        super().__init__(network, config, **kwargs)
        if network.batch != 1 or network.config.tau_sensory <= 0:
            raise ValueError('local timing experiment requires one sensory-decay environment')
        if not self.sensory_mask[target]:
            raise ValueError('target must receive direct local sensory input')
        self.target = int(target)
        self.reference = 0.
        self.previous_state = 0.
        self.gate = False
        self.quiet_count = 1
        self.event_count = 1
        self.margin = math.exp(-4*network.config.dt/network.config.tau_sensory)
        self.last_issue = None
        self.last_confirmation = None

    def _capture_forecast(self):
        super()._capture_forecast()
        n = self.network
        keys, eligibility, prediction = self.forecast
        edges = keys.remainder(n.e)
        keep = n.post[edges] == self.target
        gate = float(self.gate)
        self.forecast = (keys[keep], eligibility[keep]*gate, prediction[keep]*gate)

    @torch.no_grad()
    def observe(self, activity, reward):
        boundary = self.tick % 8 == 0
        self.last_confirmation = None
        if boundary:
            n = self.network
            observed = self.config.observation(activity, n.config.threshold,
                                               self.sensory_mask, self.sensory_gain)
            event = float(observed[0, self.target])
            due = self.forecast is not None
            if due:
                gain = min(8., max(1., self.quiet_count/self.event_count)) if event else 1.
                keys, eligibility, prediction = self.forecast
                self.forecast = keys, eligibility*gain, prediction
                self.last_confirmation = dict(gate=bool(self.last_issue['gate']),
                    prediction=self.last_issue['prediction'], target=event, gain=gain)
            current_state = float(n.sensory_state[0, self.target])
            if event:
                self.reference = abs(self.previous_state)
            value = abs(current_state)
            self.gate = bool(self.reference > 0 and
                self.reference*self.margin <= value <= self.reference/self.margin)
            self.previous_state = current_state
        super().observe(activity, reward)
        if boundary:
            n = self.network
            if self.last_confirmation is not None:
                edges, _, delta = self.last_visual_update
                self.last_confirmation['edges'] = edges.clone()
                self.last_confirmation['delta'] = delta.clone()
            raw = float(self.config.encode(activity.predicted[0, self.target], n.config.threshold))
            keys, eligibility, _ = self.forecast
            self.last_issue = dict(tick=self.tick-1, gate=self.gate,
                reference=self.reference, sensory_state=self.previous_state,
                raw_prediction=raw, prediction=raw*float(self.gate),
                edges=keys.remainder(n.e).clone(), eligibility=eligibility.clone())
            if event:
                self.event_count += 1
            else:
                self.quiet_count += 1
