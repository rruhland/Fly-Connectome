import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from m1a_event_correlation import directed_correlation


def test_same_polarity_local_correlation_has_direction_and_no_static_alarm():
    trace = torch.zeros(2, 5, 5)
    current = torch.zeros_like(trace)
    trace[0, 2, 1] = 1
    current[0, 2, 2] = 1
    x, y = directed_correlation(trace, current)
    assert x > 0 and y == 0
    x, y = directed_correlation(trace, torch.zeros_like(current))
    assert x == y == 0
    current.zero_()
    current[1, 2, 2] = 1
    x, y = directed_correlation(trace, current)
    assert x == y == 0
