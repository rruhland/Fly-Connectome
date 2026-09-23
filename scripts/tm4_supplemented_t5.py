"""Opt-in current release on measured Tm4-to-T5 edges during source silence."""

from dataclasses import replace
import math

import torch

from graded_visual import graded_release
from t5_local_order_current import T5LocalOrderNetwork


TM4_STATE = ('tm4_release_history', 'tm4_baseline', 'tm4_blank_residual',
             'tm4_floor', 'last_tm4_impulse')


def silent_source_release(current, baseline, floor, spikes, *, scale, cap):
    return graded_release(current, baseline, floor, scale=scale, cap=cap).masked_fill(
        spikes, 0)


class Tm4SupplementedT5Network(T5LocalOrderNetwork):
    """Experimental graded supplement; ordinary Tm4 spikes still propagate."""

    def __init__(self, *args, cell_types, tm4_release_cap,
                 tm4_current_scale, tm4_release_tau=.250, **kwargs):
        super().__init__(*args, cell_types=cell_types, **kwargs)
        if (tm4_current_scale <= 0 or tm4_release_tau <= 0
                or not 0 < tm4_release_cap <= 1):
            raise ValueError('positive Tm4 scale/timescale and bounded cap required')
        tm4 = torch.tensor([name == 'Tm4' for name in cell_types],
                           device=self.device)
        self.tm4_nodes = tm4.nonzero().flatten()
        self.tm4_supplement_edges = self.tm4_arm_edges.nonzero().flatten()
        source_lookup = torch.full((self.n,), -1, dtype=torch.long,
                                   device=self.device)
        source_lookup[self.tm4_nodes] = torch.arange(len(self.tm4_nodes),
                                                    device=self.device)
        self.tm4_sources = source_lookup[self.pre[self.tm4_supplement_edges]]
        self.tm4_posts = self.post[self.tm4_supplement_edges]
        self.tm4_delays = self.delays[self.tm4_supplement_edges]
        self.tm4_release_history = torch.zeros(self.history_length,
            len(self.tm4_nodes), device=self.device)
        self.tm4_baseline = torch.zeros(len(self.tm4_nodes), device=self.device)
        self.tm4_blank_residual = torch.zeros(128, len(self.tm4_nodes),
                                               device=self.device)
        self.tm4_floor = torch.zeros(len(self.tm4_nodes), device=self.device)
        self.last_tm4_impulse = torch.zeros_like(self.voltage)
        self.tm4_blank_count = 0
        self.tm4_enabled = False
        self.tm4_release_cap = float(tm4_release_cap)
        self.tm4_current_scale = float(tm4_current_scale)
        self.tm4_baseline_decay = math.exp(-self.config.dt/tm4_release_tau)

    def enable_tm4(self):
        if not self.graded_enabled or self.tm4_blank_count < 128:
            raise ValueError('calibrate Tm4 release on 128 Tm9-on blank ticks')
        self.tm4_floor.copy_(torch.quantile(self.tm4_blank_residual, .99, dim=0))
        self.tm4_release_history.zero_()
        self.last_tm4_impulse.zero_()
        self.tm4_enabled = True

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        tick = self.step_index
        self.last_tm4_impulse.zero_()
        if self.tm4_enabled:
            edges, posts = self.tm4_supplement_edges, self.tm4_posts
            due = self.tm4_release_history[(tick-self.tm4_delays).remainder(
                self.history_length), self.tm4_sources]
            impulse = self.magnitudes[edges]*self.signs[edges]*due
            self.last_tm4_impulse[0].index_add_(0, posts, impulse)
            decay = self.current_decay[posts]
            self.feedforward_current[0].index_add_(0, posts, impulse/decay)
            self.arm_traces[0].index_add_(0, self.t5_lookup[posts],
                                          impulse/decay)
        activity = super().step(sensory_current,
                                capture_increments=capture_increments)
        if capture_increments:
            activity = replace(activity, feedforward_arrivals=(
                activity.feedforward_arrivals+self.last_tm4_impulse))
        source_current = (self.feedforward_current[0, self.tm4_nodes]
            +self.predictive_current[0, self.tm4_nodes]
            +self.behavioral_current[0, self.tm4_nodes])
        residual = (source_current-self.tm4_baseline).clamp(min=0)
        if not self.tm4_enabled:
            self.tm4_blank_residual[tick % 128].copy_(residual)
            self.tm4_blank_count += 1
        else:
            self.tm4_release_history[tick % self.history_length].copy_(
                silent_source_release(source_current, self.tm4_baseline,
                    self.tm4_floor, activity.spikes[0, self.tm4_nodes],
                    scale=self.tm4_current_scale, cap=self.tm4_release_cap))
        self.tm4_baseline.mul_(self.tm4_baseline_decay).add_(
            source_current, alpha=1-self.tm4_baseline_decay)
        return activity
