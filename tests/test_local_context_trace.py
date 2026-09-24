"""A local history trace must be available only at raw first-sighting sites."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_context_trace import context_code


def test_trace_reaches_raw_site_without_creating_other_sources():
    events = torch.zeros((2, 32, 64))
    coincidence = torch.zeros((16, 32, 64))
    trace = torch.zeros((24, 32, 64))
    events[0, 16, 20] = 1
    trace[3, 16, 12] = .5
    code = context_code(events, coincidence, trace, use_trace=True)
    assert code.shape == (26, 32, 64)
    assert code[3+2, 16, 20] == .5
    assert (code.sum(0) > 0).sum() == 1
    assert context_code(events, coincidence, trace,
                        use_trace=False)[2:].sum() == 0


def test_current_coincidence_suppresses_first_sighting_pathway():
    events = torch.zeros((2, 32, 64))
    coincidence = torch.zeros((16, 32, 64))
    trace = torch.ones((24, 32, 64))
    events[0, 16, 20] = 1
    coincidence[0, 16, 20] = 1
    assert context_code(events, coincidence, trace, use_trace=True).sum() == 0
