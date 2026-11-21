"""Keep a visual motor estimate through a camera remount during a pursuit game."""
from copy import deepcopy

import numpy as np

from .tt10_target import read_markers
from .tt11_tracking import PursuitPlayer, orientation


class RemountPlayer(PursuitPlayer):
    def __init__(self, prior, method):
        super().__init__(prior)
        self.method = method
        self.last_pair = None
        self.pair_velocity = np.zeros((2, 3))
        self.frame_count = 0
        self.events = []

    def observe(self, rgb):
        pair = np.stack(read_markers(rgb))
        if self.last_pair is not None:
            prediction = self.last_pair + self.pair_velocity
            residual = np.linalg.norm(pair - prediction, axis=1)
            if min(residual) > .035 and np.linalg.norm(np.cross(*prediction)) > .001:
                shift = orientation(pair.T @ prediction)
                remaining = np.linalg.norm(pair - prediction @ shift.T, axis=1)
                if max(remaining) < .025 and np.mean(remaining) < .35 * np.mean(residual):
                    self.events.append(self.frame_count)
                    if self.method == 'transport':
                        self.cross = shift @ self.cross
                        self.view_evidence = shift @ self.view_evidence
                        if self.previous is not None:
                            self.previous = shift @ self.previous
                            self.previous_goal = shift @ self.previous_goal
                    elif self.method == 'restart_fit':
                        self.cross.fill(0.)
                        self.gram.fill(0.)
                        self.view_evidence = np.eye(3) * .00035
                        self.previous = None
                    self.last_pair = self.last_pair @ shift.T
                    self.pair_velocity = self.pair_velocity @ shift.T
            self.pair_velocity = .5 * self.pair_velocity + .5 * (pair - self.last_pair)
        self.last_pair = pair
        self.frame_count += 1


def state(player):
    return {k: v.tolist() if isinstance(v, np.ndarray) else deepcopy(v) for k, v in vars(player).items()}


def restore(data):
    player = RemountPlayer(np.array(data['prior']), data['method'])
    for key, value in data.items():
        if isinstance(getattr(player, key), np.ndarray) or key in ('previous', 'previous_goal', 'last_pair'):
            setattr(player, key, None if value is None else np.array(value))
        else:
            setattr(player, key, value)
    return player
