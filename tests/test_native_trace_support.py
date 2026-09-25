"""A local event trace retains sparse motion evidence across camera frames."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correlation_input_latent import correlation_sequence
from run_native_trace_support import trace_correlation_sequence


def test_trace_connects_delayed_neighbor_events_without_new_edges():
    events = [torch.zeros((2, 32, 64)) for _ in range(4)]
    events[0][0, 16, 16] = 1
    events[3][0, 16, 17] = 1
    immediate = correlation_sequence(events)
    traced = trace_correlation_sequence(events)
    assert immediate[3].sum() == 0
    assert traced[3].sum() > 0
    assert traced[3].shape == immediate[3].shape
