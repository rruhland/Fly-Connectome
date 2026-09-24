"""Opt-in event-observation-only local learning of wall reflection."""

import json
import time
from pathlib import Path

import torch

from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from m1a_pong_state_ceiling import EventBallTracker


OUT = Path('docs/experiments/2026-09-23-m1a-local-wall-model-results.json')
LOW = .5/32
HIGH = 1-LOW


def reflect_feature(y, displacement, *, horizon):
    linear = y+displacement*(horizon/8)
    feature = torch.where(linear < LOW, 2*(LOW-linear),
                          torch.where(linear > HIGH, 2*(HIGH-linear),
                                      torch.zeros_like(linear)))
    return linear, feature


def update_gain(gain, feature, error, *, eta):
    delta = eta*(feature*error).sum()/(feature.square().sum()+1e-4)
    return float(torch.clamp(torch.as_tensor(gain)+delta, 0, 1.5))


@torch.no_grad()
def collect():
    seeds = tuple(range(16))
    frames = 800
    pong = Pong(seeds)
    camera = EventCamera(len(seeds), 32, 64)
    tracker = EventBallTracker(len(seeds))
    observed = torch.empty((frames, len(seeds), 2))
    truth = torch.empty_like(observed)
    velocity = torch.empty_like(observed)
    rally = torch.empty((frames, len(seeds)), dtype=torch.int32)
    rally_id = torch.zeros(len(seeds), dtype=torch.int32)
    for t in range(frames):
        observed[t] = tracker.observe(camera.observe(pong.render()))
        truth[t] = pong.ball
        velocity[t] = pong.ball_velocity
        rally[t] = rally_id
        outcome = pong.step(torch.zeros(len(seeds)))
        rally_id += (outcome.scores != 0).int()
    return observed, truth, velocity, rally


def forecast(current, past, horizon, gain):
    linear = current+(current-past)*(horizon/8)
    vertical, feature = reflect_feature(
        current[..., 1], current[..., 1]-past[..., 1], horizon=horizon)
    prediction = torch.stack((linear[..., 0], vertical+gain*feature), -1)
    return prediction, feature


def score(prediction, target, valid, categories):
    distance = (prediction-target).norm(dim=-1)
    vertical = (prediction[..., 1]-target[..., 1]).abs()
    return {name: dict(samples=int((valid & mask).sum()),
        mean_error=float(distance[valid & mask].mean()) if (valid & mask).any() else None,
        mean_y_error=float(vertical[valid & mask].mean()) if (valid & mask).any() else None)
        for name, mask in categories.items()}


@torch.no_grad()
def main():
    started = time.perf_counter()
    observed, truth, velocity, rally = collect()
    report = dict(training_seeds=list(range(8)), test_seeds=list(range(8, 16)),
        frames_per_seed=800, learning_rate=.05, horizons={})
    for horizon in (16, 32):
        gain = 0.
        updates = examples = 0
        pending_y = torch.full((800, 8), float('nan'))
        pending_feature = torch.zeros((800, 8))
        pending_x = torch.full((800, 8), float('nan'))
        for tick in range(800):
            if tick >= horizon+8:
                issue = tick-horizon
                target = observed[tick, :8]
                feature = pending_feature[issue]
                valid = (torch.isfinite(pending_y[issue])
                    & torch.isfinite(pending_x[issue])
                    & torch.isfinite(target).all(-1)
                    & ((target[:, 0]-pending_x[issue]).abs() <= .3)
                    & (feature != 0))
                if valid.any():
                    gain = update_gain(gain, feature[valid],
                        target[valid, 1]-pending_y[issue, valid], eta=.05)
                    updates += 1
                    examples += int(valid.sum())
            if 8 <= tick < 800-horizon:
                current = observed[tick, :8]
                past = observed[tick-8, :8]
                prediction, feature = forecast(current, past, horizon, gain)
                pending_y[tick] = prediction[:, 1]
                pending_feature[tick] = feature
                pending_x[tick] = current[:, 0]
        t = torch.arange(8, 800-horizon)
        current = observed[t, 8:]
        past = observed[t-8, 8:]
        target = truth[t+horizon, 8:]
        valid = (torch.isfinite(current).all(-1)
            & torch.isfinite(past).all(-1)
            & (rally[t, 8:] == rally[t+horizon, 8:]))
        paddle = velocity[t, 8:, 0]*velocity[t+horizon, 8:, 0] < 0
        wall = (velocity[t, 8:, 1]*velocity[t+horizon, 8:, 1] < 0) & ~paddle
        categories = dict(all=torch.ones_like(valid),
            free_flight=~wall & ~paddle,
            wall_reversal=wall, paddle_reversal=paddle)
        methods = {name: score(forecast(current, past, horizon, value)[0],
                               target, valid, categories)
                   for name, value in (('constant_velocity', 0.),
                                       ('learned_reflection', gain),
                                       ('unit_reflection', 1.))}
        report['horizons'][str(horizon)] = dict(
            learned_gain=gain, update_batches=updates,
            update_examples=examples, methods=methods)
    report['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
