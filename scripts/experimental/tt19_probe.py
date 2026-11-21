"""Choose a short camera motor-identification tape before seeing a new body."""
import numpy as np
from scipy.signal import lfilter


def command_tape(width, frames, amplitude=.75, kernel=None):
    """Exercise all public ports and condition the regression at plausible lags."""
    if width < 1 or frames < 2 * width or not 0 < amplitude <= 1:
        raise ValueError('need at least two frames per port and a bounded amplitude')
    if kernel is None:
        from .tt19_online_body import kernel_for
        kernel = kernel_for()
    rng = np.random.default_rng([190101, width, frames])
    best_score = -np.inf
    winner = None
    # Keep the physical command excursion bounded even if individual port
    # limits are unknown. Do not evaluate candidate tapes on the target arm.
    for _ in range(6):
        pulses = rng.choice((-amplitude, amplitude), (128, (frames + 1) // 2, width))
        commands = np.repeat(pulses, 2, axis=1)[:, :frames]
        demand = np.zeros((len(commands), width))
        for tick in range(frames):
            next_demand = np.clip(demand + commands[:, tick], -3.2, 3.2)
            commands[:, tick] = next_demand - demand
            demand = next_demand
        responses = np.cumsum(commands, axis=1) + lfilter(np.asarray(kernel) - 1., [1.], commands, axis=1)
        differences = responses[:,3:] - responses[:,:-3]
        centered = responses - responses.mean(axis=1, keepdims=True)
        slow = np.linalg.svd(centered, compute_uv=False)[:, -1]
        fast = np.linalg.svd(differences, compute_uv=False)[:, -1]
        scores = np.minimum(slow, fast)
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best_score = scores[index]
            winner = commands[index].copy()
    return winner
