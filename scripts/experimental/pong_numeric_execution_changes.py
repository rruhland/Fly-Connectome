"""Execution-only numeric motor changes, hidden from public visual learners."""
from collections import deque
import math


def install_intervention(environment, settings):
    if set(settings) - {'delay_commands', 'drive_bias', 'drive_gain'}:
        raise ValueError('unknown physical execution setting')
    delay = settings.get('delay_commands', 0)
    bias = float(settings.get('drive_bias', 0.))
    gain = float(settings.get('drive_gain', 1.))
    if (type(delay) is not int or delay < 0 or not math.isfinite(bias)
            or not math.isfinite(gain) or gain <= 0):
        raise ValueError('finite motor change and nonnegative integer delay required')
    queue = deque([1] * delay)
    original = environment.step
    physical_actions = []
    def execute(action):
        queue.append(action)
        physical = queue.popleft()
        physical_actions.append(physical)
        return original(physical)
    environment.step = execute
    if gain != 1. or bias:
        physical_step = environment.game.step
        def changed(drive, **kwargs):
            return physical_step(gain * drive + bias, **kwargs)
        environment.game.step = changed
    return physical_actions
