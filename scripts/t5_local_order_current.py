"""Opt-in local T5 timing current on measured Tm4 and Tm9 afferents."""

import math

import torch

from graded_visual import GradedVisualNetwork


LAG = 8
ORDER_STATE = ('arm_traces', 'arm_baseline', 'residual_history',
               'blank_order', 'order_floor', 'pending_gate', 'last_gate_current')


def local_order_current(delayed_tm4, delayed_tm9, now_tm4, now_tm9,
                        floor, *, gain, cap):
    if gain <= 0 or cap <= 0:
        raise ValueError('local gain and current cap must be positive')
    order = delayed_tm4*now_tm9 - delayed_tm9*now_tm4
    return ((order-floor).clamp(min=0)*gain).clamp(max=cap)


class T5LocalOrderNetwork(GradedVisualNetwork):
    """Causal local arm interaction injected into each T5's own membrane."""

    def __init__(self, *args, cell_types, gate_gain, gate_cap=1., **kwargs):
        super().__init__(*args, cell_types=cell_types, **kwargs)
        types = torch.tensor([name.startswith('T5') for name in cell_types],
                             device=self.device)
        self.t5_nodes = types.nonzero().flatten()
        lookup = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        lookup[self.t5_nodes] = torch.arange(len(self.t5_nodes), device=self.device)
        self.t5_lookup = lookup
        tm4 = torch.tensor([name == 'Tm4' for name in cell_types],
                           device=self.device)
        self.tm4_arm_edges = tm4[self.pre] & types[self.post] & (self.pathways == 0)
        tm9_t5 = self.graded_edge_mask & types[self.post]
        if not self.tm4_arm_edges.any() or not tm9_t5.any():
            raise ValueError('measured Tm4 and graded Tm9 T5 arms required')
        if not (self.pathways[tm9_t5] == 0).all():
            raise ValueError('graded Tm9 T5 arm must be feedforward')
        if gate_gain <= 0 or gate_cap <= 0:
            raise ValueError('positive gate gain and cap required')
        self.gate_gain = float(gate_gain)
        self.gate_cap = float(gate_cap)
        shape = (2, len(self.t5_nodes))
        self.arm_traces = torch.zeros(shape, device=self.device)
        self.arm_baseline = torch.zeros(shape, device=self.device)
        self.residual_history = torch.zeros(LAG+1, *shape, device=self.device)
        self.blank_order = torch.zeros(128, len(self.t5_nodes), device=self.device)
        self.order_floor = torch.zeros(len(self.t5_nodes), device=self.device)
        self.pending_gate = torch.zeros_like(self.order_floor)
        self.last_gate_current = torch.zeros_like(self.order_floor)
        self.order_blank_count = 0
        self.order_enabled = False
        self.arm_baseline_decay = math.exp(-self.config.dt/.250)

    def enable_order(self):
        if not self.graded_enabled or self.order_blank_count < len(self.blank_order):
            raise ValueError('calibrate on blank after enabling graded source')
        self.order_floor.copy_(torch.quantile(self.blank_order, .99, dim=0))
        self.pending_gate.zero_()
        self.order_enabled = True

    @torch.no_grad()
    def step(self, sensory_current, *, capture_increments=False):
        applied = self.pending_gate.clone()
        self.last_gate_current.copy_(applied)
        self.rest_current[self.t5_nodes] += applied
        activity = super().step(sensory_current,
                                capture_increments=capture_increments)
        self.rest_current[self.t5_nodes] -= applied

        decay = self.current_decay[self.t5_nodes]
        self.arm_traces.mul_(decay)
        edges = activity.arrival_edges
        tm4_edges = edges[self.tm4_arm_edges[edges]]
        self.arm_traces[0].index_add_(0, self.t5_lookup[self.post[tm4_edges]],
            self.magnitudes[tm4_edges]*self.signs[tm4_edges])
        self.arm_traces[1].add_(self.last_graded_impulse[0, self.t5_nodes])
        residual = self.arm_traces-self.arm_baseline
        tick = self.step_index-1
        delayed = self.residual_history[(tick-LAG) % (LAG+1)]
        order = (delayed[0]*residual[1]-delayed[1]*residual[0]).clone()
        self.residual_history[tick % (LAG+1)].copy_(residual)
        self.arm_baseline.mul_(self.arm_baseline_decay).add_(
            self.arm_traces, alpha=1-self.arm_baseline_decay)
        if self.order_enabled:
            self.pending_gate.copy_(local_order_current(
                delayed[0], delayed[1], residual[0], residual[1],
                self.order_floor, gain=self.gate_gain, cap=self.gate_cap))
        elif self.graded_enabled:
            self.blank_order[tick % len(self.blank_order)].copy_(order.clamp(min=0))
            self.order_blank_count += 1
        return activity
