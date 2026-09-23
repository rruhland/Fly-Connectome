"""Opt-in local CT1 compartments on measured partner edges; no graph mutation."""

import math

import torch


SHADOW_STATE = ('current', 'baseline', 'release_history', 'blank_residual',
                'floor', 'last_impulse', 'last_release')


class CT1ColumnShadow:
    def __init__(self, network, n_columns, *, tm1_nodes, tm1_columns, tm1_weights,
                 tm9_indices, tm9_columns, tm9_weights,
                 output_posts, output_columns, output_weights,
                 scale=.02, cap=.10):
        self.tm1_nodes = torch.as_tensor(tm1_nodes, dtype=torch.long)
        self.tm1_columns = torch.as_tensor(tm1_columns, dtype=torch.long)
        self.tm1_weights = torch.as_tensor(tm1_weights, dtype=torch.float32)
        self.tm9_indices = torch.as_tensor(tm9_indices, dtype=torch.long)
        self.tm9_columns = torch.as_tensor(tm9_columns, dtype=torch.long)
        self.tm9_weights = torch.as_tensor(tm9_weights, dtype=torch.float32)
        self.output_posts = torch.as_tensor(output_posts, dtype=torch.long)
        self.output_columns = torch.as_tensor(output_columns, dtype=torch.long)
        self.output_weights = torch.as_tensor(output_weights, dtype=torch.float32)
        if (n_columns < 1 or scale <= 0 or not 0 < cap <= 1
                or any(len(edge) != len(column) or len(edge) != len(weight)
                       for edge, column, weight in (
                           (self.tm1_nodes, self.tm1_columns, self.tm1_weights),
                           (self.tm9_indices, self.tm9_columns, self.tm9_weights),
                           (self.output_posts, self.output_columns, self.output_weights)))
                or any((column < 0).any() or (column >= n_columns).any()
                       for column in (self.tm1_columns, self.tm9_columns,
                                      self.output_columns))
                or any((weight <= 0).any() for weight in (
                    self.tm1_weights, self.tm9_weights, self.output_weights))):
            raise ValueError('positive measured local edges and bounded release required')
        self.current = torch.zeros(n_columns)
        self.baseline = torch.zeros(n_columns)
        self.release_history = torch.zeros(network.history_length, n_columns)
        self.blank_residual = torch.zeros(128, n_columns)
        self.floor = torch.zeros(n_columns)
        self.last_impulse = torch.zeros(network.n)
        self.last_release = torch.zeros(n_columns)
        self.blank_count = 0
        self.enabled = False
        self.current_decay = math.exp(-network.config.dt/network.config.tau_current)
        self.baseline_decay = math.exp(-network.config.dt/.250)
        self.scale = float(scale)
        self.cap = float(cap)

    def enable(self):
        if self.blank_count < len(self.blank_residual):
            raise ValueError('calibrate CT1 on blank before enabling release')
        self.floor.copy_(torch.quantile(self.blank_residual, .99, dim=0))
        self.release_history.zero_()
        self.enabled = True

    @torch.no_grad()
    def begin_tick(self, network, *, output_enabled):
        tick = network.step_index
        previous = (tick-1) % network.history_length
        self.last_impulse.zero_()
        if output_enabled:
            if not self.enabled:
                raise ValueError('CT1 release must be blank-calibrated')
            due = self.release_history[previous, self.output_columns]
            impulse = -self.output_weights*due
            self.last_impulse.index_add_(0, self.output_posts, impulse)
            network.feedforward_current[0].index_add_(
                0, self.output_posts,
                impulse/network.current_decay[self.output_posts])
        self.current.mul_(self.current_decay)
        if len(self.tm1_nodes):
            source = network.history[previous, 0, self.tm1_nodes].float()
            self.current.index_add_(0, self.tm1_columns,
                                    self.tm1_weights*source)
        if len(self.tm9_indices):
            source = network.release_history[previous, self.tm9_indices]
            self.current.index_add_(0, self.tm9_columns,
                                    self.tm9_weights*source)
        residual = (self.current-self.baseline).clamp(min=0)
        if self.enabled:
            self.last_release.copy_(((self.current-self.baseline-self.floor)
                                     /self.scale).clamp(0, self.cap))
            self.release_history[tick % network.history_length].copy_(
                self.last_release)
        else:
            self.blank_residual[tick % len(self.blank_residual)].copy_(residual)
            self.blank_count += 1
        self.baseline.mul_(self.baseline_decay).add_(
            self.current, alpha=1-self.baseline_decay)
