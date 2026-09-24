"""Event-only Pong ball tracker and fixed forecast ceilings for M1A."""

import json
import time
from pathlib import Path

import torch

from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera


OUT = Path('docs/experiments/2026-09-23-m1a-pong-state-ceiling-results.json')


class EventBallTracker:
    """Reconstruct lit pixels from events; ignore fixed paddle columns."""

    def __init__(self, batch):
        self.batch = batch
        self.occupancy = torch.zeros((batch, 32*64), dtype=torch.bool)
        self.x = (torch.arange(6, 58, dtype=torch.float32)+.5)/64
        self.y = (torch.arange(32, dtype=torch.float32)+.5)/32

    def observe(self, events):
        self.occupancy[events.environments, events.pixels] = events.on
        interior = self.occupancy.view(self.batch, 32, 64)[:, :, 6:58].float()
        mass = interior.sum((1, 2))
        x = (interior*self.x[None, None, :]).sum((1, 2))/mass.clamp(min=1)
        y = (interior*self.y[None, :, None]).sum((1, 2))/mass.clamp(min=1)
        center = torch.stack((x, y), 1)
        center[mass == 0] = float('nan')
        return center


def error_summary(prediction, actual, valid, category):
    groups = {}
    for name, members in category.items():
        selected = valid & members
        error = (prediction-actual).norm(dim=-1)[selected]
        groups[name] = dict(samples=int(selected.sum()),
                            mean_error=float(error.mean()) if error.numel() else None)
    return groups


@torch.no_grad()
def main():
    started = time.perf_counter()
    seeds = tuple(range(16))
    frames = 800
    pong = Pong(seeds)
    camera = EventCamera(len(seeds), 32, 64)
    tracker = EventBallTracker(len(seeds))
    observations = torch.empty((frames, len(seeds), 2))
    truth = torch.empty_like(observations)
    velocity = torch.empty_like(observations)
    rally = torch.empty((frames, len(seeds)), dtype=torch.int32)
    rally_id = torch.zeros(len(seeds), dtype=torch.int32)
    for t in range(frames):
        observations[t] = tracker.observe(camera.observe(pong.render()))
        truth[t] = pong.ball
        velocity[t] = pong.ball_velocity
        rally[t] = rally_id
        outcome = pong.step(torch.zeros(len(seeds)))
        rally_id += (outcome.scores != 0).int()
    visible = torch.isfinite(observations).all(-1)
    localization = (observations-truth).norm(dim=-1)[visible]
    report = dict(seeds=list(seeds), frames_per_seed=frames, height=32, width=64,
        observation=dict(visible=int(visible.sum()), total=visible.numel(),
            coverage=float(visible.float().mean()),
            mean_error=float(localization.mean()),
            p90_error=float(torch.quantile(localization, .9))),
        horizons={})
    for horizon in (8, 16, 32):
        t = torch.arange(8, frames-horizon)
        future = truth[t+horizon]
        same_rally = rally[t] == rally[t+horizon]
        paddle = (velocity[t, :, 0]*velocity[t+horizon, :, 0] < 0)
        wall = (velocity[t, :, 1]*velocity[t+horizon, :, 1] < 0) & ~paddle
        categories = dict(all=torch.ones_like(same_rally),
            free_flight=~paddle & ~wall,
            wall_reversal=wall, paddle_reversal=paddle)
        current = observations[t]
        past = observations[t-8]
        current_visible = torch.isfinite(current).all(-1)
        past_visible = torch.isfinite(past).all(-1)
        common_valid = same_rally & current_visible & past_visible
        forecasts = dict(persistence=current,
            observed_velocity=current+(current-past)*(horizon/8),
            privileged_velocity=truth[t]+velocity[t]*(pong.config.dt*horizon))
        scores = {}
        for name, prediction in forecasts.items():
            scores[name] = error_summary(prediction, future, common_valid, categories)
        report['horizons'][str(horizon)] = scores
    report['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
