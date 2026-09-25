"""The sensory lag stack contains only present and past camera events."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from event_lag_stack import lagged_event_sequence


def test_lag_stack_is_causal_and_retains_quiet_frame_history():
    on = torch.zeros((2, 32, 64))
    off = torch.zeros_like(on)
    on[1, 8, 9] = 1
    off[0, 8, 10] = 1
    stack = lagged_event_sequence([on, torch.zeros_like(on), off], window=3)
    assert len(stack) == 3
    assert stack[0].shape == (6, 32, 64)
    assert stack[0][1, 8, 9] == 1
    assert stack[0][2:].sum() == 0
    assert stack[1][3, 8, 9] == 1
    assert stack[2][0, 8, 10] == 1
    assert stack[2][5, 8, 9] == 1
