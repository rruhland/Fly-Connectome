"""Fixed local signed-event history for learning temporal conjunction units."""

from collections import deque

import torch


def lagged_event_sequence(events, *, window=8):
    if window < 1:
        raise ValueError('event history window must be positive')
    history = deque(maxlen=window)
    zero = torch.zeros_like(events[0])
    result = []
    for event in events:
        history.appendleft(event)
        result.append(torch.cat((*history, *((zero,)*(window-len(history))))))
    return result
