"""Score-only Pong boundary with terminal camera capture before explicit reset."""
from dataclasses import dataclass
import math
import torch
from fly_connectome.pong import Pong,Physics


@dataclass(frozen=True)
class CameraTransition:
    frame: torch.Tensor
    reward: float
    elapsed_seconds: float
    terminated: bool
    truncated: bool
    physics_ticks: int
    hits: int


class PongCamera:
    actions = (-1., 0., 1.)

    def __init__(self, seed=0, *, max_episode_ticks=None,opponent_speed=.45):
        if max_episode_ticks is not None and (type(max_episode_ticks) is not int or max_episode_ticks < 1):
            raise ValueError('positive episode tick limit required')
        if not isinstance(opponent_speed,(int,float)) or not math.isfinite(opponent_speed) or opponent_speed<=0:
            raise ValueError('finite positive opponent speed required')
        self.game = Pong([seed],config=Physics(opponent_speed=float(opponent_speed)))
        self.max_episode_ticks = max_episode_ticks
        self.decisions = self.episode_ticks = self.total_ticks = 0
        self.needs_reset = False

    def reset(self):
        self.game.reset(torch.ones(1, dtype=torch.bool))
        self.episode_ticks = 0
        self.needs_reset = False
        return self.game.render(64, 64)[0].clone()

    def step(self, action):
        if self.needs_reset:
            raise ValueError('reset required after termination or truncation')
        if type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('legal actuator index required')
        budget = (self.decisions+1)*12//5 - self.decisions*12//5
        reward = 0.
        hits = 0
        terminated = truncated = False
        for ticks in range(1, budget+1):
            outcome = self.game.step(torch.tensor([self.actions[action]]), auto_reset=False)
            self.episode_ticks += 1
            self.total_ticks += 1
            reward += float(outcome.scores[0])
            hits += int(outcome.hits[0])
            terminated = bool(outcome.scores[0])
            truncated = bool(not terminated and self.max_episode_ticks is not None
                             and self.episode_ticks >= self.max_episode_ticks)
            if terminated or truncated:
                break
        self.decisions += 1
        self.needs_reset = terminated or truncated
        return CameraTransition(self.game.render(64, 64)[0].clone(), reward,
                                ticks*self.game.config.dt, terminated, truncated, ticks, hits)
