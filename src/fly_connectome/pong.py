"""Fixed batched Pong physics. Privileged state stays on the environment side."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Physics:
    dt: float = 1 / 120
    paddle_half: float = .10
    paddle_x: float = .05
    ball_radius: float = .0125
    ball_speed: float = .55
    tau_activation: float = .08
    motor_gain: float = 1.
    motor_force: float = 3.
    drag: float = 4.
    max_velocity: float = .7
    opponent_acceleration: float = 2.
    opponent_speed: float = .45


class Body:
    def __init__(self, batch, config, device='cpu'):
        self.config = config
        self.position = torch.full((batch,), .5, device=device)
        self.activation = torch.zeros(batch, device=device)
        self.velocity = torch.zeros(batch, device=device)

    @torch.no_grad()
    def step(self, up_minus_down):
        p = self.config
        self.activation.lerp_(p.motor_gain * up_minus_down, 1 - math.exp(-p.dt / p.tau_activation))
        # Screen coordinates increase downward; positive opponent drive points up.
        self.velocity.add_(p.dt * (-p.motor_force * self.activation - p.drag * self.velocity))
        self.velocity.clamp_(-p.max_velocity, p.max_velocity)
        self.position.add_(p.dt * self.velocity)
        blocked = (self.position <= p.paddle_half) | (self.position >= 1 - p.paddle_half)
        self.position.clamp_(p.paddle_half, 1 - p.paddle_half)
        self.velocity.masked_fill_(blocked, 0.)

    def reset(self, mask):
        self.position[mask] = .5
        self.activation[mask] = 0.
        self.velocity[mask] = 0.


@dataclass(frozen=True)
class Outcome:
    scores: torch.Tensor
    hits: torch.Tensor
    rally_steps: torch.Tensor


class Pong:
    def __init__(self, seeds, config=Physics(), device='cpu'):
        if not seeds:
            raise ValueError("at least one environment seed required")
        self.config = config
        self.device = device
        self.batch = len(seeds)
        # Counter-style per-environment integer PRNG; no shared generator on resets.
        self.rng = torch.tensor(seeds, dtype=torch.int64, device=device).remainder(2147483647)
        self.body = Body(self.batch, config, device)
        self.ball = torch.zeros(self.batch, 2, device=device)
        self.ball_velocity = torch.zeros_like(self.ball)
        self.opponent = torch.full((self.batch,), .5, device=device)
        self.opponent_velocity = torch.zeros(self.batch, device=device)
        self.rally_steps = torch.zeros(self.batch, dtype=torch.long, device=device)
        self.reset(torch.ones(self.batch, dtype=torch.bool, device=device))

    @torch.no_grad()
    def reset(self, mask):
        self.rng[mask] = (self.rng[mask] * 48271 + 1).remainder(2147483647)
        random = self.rng[mask].double() / 2147483647
        self.ball[mask, 0] = .5
        self.ball[mask, 1] = (.2 + .6 * random).float()
        angle = (random - .5) * 1.2
        direction = torch.where(self.rng[mask].remainder(2) == 0, 1., -1.)
        self.ball_velocity[mask, 0] = (direction * self.config.ball_speed * angle.cos()).float()
        self.ball_velocity[mask, 1] = (self.config.ball_speed * angle.sin()).float()
        self.body.reset(mask)
        self.opponent[mask] = .5
        self.opponent_velocity[mask] = 0.
        self.rally_steps[mask] = 0

    @torch.no_grad()
    def step(self, drive):
        if drive.shape != (self.batch,) or not torch.isfinite(drive).all():
            raise ValueError("finite continuous motor drive required per environment")
        p = self.config
        self.body.step(drive.detach())
        desired = ((self.ball[:, 1] - self.opponent) * 4).clamp(-p.opponent_speed, p.opponent_speed)
        self.opponent_velocity.add_((desired - self.opponent_velocity).clamp(
            -p.opponent_acceleration * p.dt, p.opponent_acceleration * p.dt))
        self.opponent.add_(p.dt * self.opponent_velocity).clamp_(p.paddle_half, 1 - p.paddle_half)
        old_x = self.ball[:, 0].clone()
        self.ball.add_(self.ball_velocity * p.dt)
        low = self.ball[:, 1] < p.ball_radius
        high = self.ball[:, 1] > 1 - p.ball_radius
        self.ball[low, 1] = 2 * p.ball_radius - self.ball[low, 1]
        self.ball[high, 1] = 2 * (1 - p.ball_radius) - self.ball[high, 1]
        self.ball_velocity[:, 1] = torch.where(low | high, -self.ball_velocity[:, 1], self.ball_velocity[:, 1])
        left_plane, right_plane = p.paddle_x + p.ball_radius, 1 - p.paddle_x - p.ball_radius
        left = ((old_x >= left_plane) & (self.ball[:, 0] <= left_plane)
                & (self.ball_velocity[:, 0] < 0)
                & ((self.ball[:, 1] - self.body.position).abs() <= p.paddle_half + p.ball_radius))
        right = ((old_x <= right_plane) & (self.ball[:, 0] >= right_plane)
                 & (self.ball_velocity[:, 0] > 0)
                 & ((self.ball[:, 1] - self.opponent).abs() <= p.paddle_half + p.ball_radius))
        collision = left | right
        center = torch.where(left, self.body.position, self.opponent)
        angle = ((self.ball[:, 1] - center) / p.paddle_half).clamp(-1, 1) * .9
        self.ball_velocity[collision, 0] = (torch.where(left, 1., -1.) * p.ball_speed * angle.cos())[collision]
        self.ball_velocity[collision, 1] = (p.ball_speed * angle.sin())[collision]
        self.ball[left, 0] = 2 * left_plane - self.ball[left, 0]
        self.ball[right, 0] = 2 * right_plane - self.ball[right, 0]
        scores = (self.ball[:, 0] > 1).float() - (self.ball[:, 0] < 0).float()
        self.rally_steps.add_(1)
        outcome = Outcome(scores, left.float(), self.rally_steps.clone())
        self.reset(scores != 0)
        return outcome

    @torch.no_grad()
    def render(self, height=32, width=64):
        p = self.config
        y, x = torch.meshgrid((torch.arange(height, device=self.device) + .5) / height,
                              (torch.arange(width, device=self.device) + .5) / width, indexing='ij')
        ball = ((x - self.ball[:, 0, None, None]).abs() <= max(p.ball_radius, .5 / width)) & (
            (y - self.ball[:, 1, None, None]).abs() <= max(p.ball_radius, .5 / height))
        left = ((x - p.paddle_x).abs() <= .0125) & ((y - self.body.position[:, None, None]).abs() <= p.paddle_half)
        right = ((x - (1 - p.paddle_x)).abs() <= .0125) & ((y - self.opponent[:, None, None]).abs() <= p.paddle_half)
        return ball | left | right


@dataclass(frozen=True)
class Curriculum:
    bootstrap: int = 10000
    fade: int = 20000
    hit_scale: float = .25
    score_scale: float = 1.

    def __post_init__(self):
        if self.bootstrap < 0 or self.fade <= 0 or not 0 <= self.hit_scale < self.score_scale <= 1:
            raise ValueError("bounded score reward must exceed hit shaping; fade must be positive")

    def alpha(self, step):
        return self.hit_scale * max(0., min(1., 1 - (step - self.bootstrap) / self.fade))

    def reward(self, step, scores, hits):
        return (self.score_scale * scores + self.alpha(step) * hits).clamp(-1, 1)
