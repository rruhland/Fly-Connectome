"""Fixed-drive 120 Hz Pong physics with 50 Hz promoted visual perception."""

import argparse
import json
import time
from pathlib import Path

import torch

from fly_connectome.pong import Pong
from fly_connectome.vision import ProbabilisticVisualState, VisualStateEncoder, load_default


def dense_events(previous, frame):
    """OFF then ON, equivalent to EventCamera's binary frame differences."""
    return torch.stack((previous & ~frame, ~previous & frame)).float()


def physics_ticks_for_sample(sample):
    return (sample+1)*12//5 - sample*12//5


@torch.no_grad()
def run(*, seed=1101, learn=True, samples=240, warmup=40,
        checkpoint=None, save=None):
    if not 0 <= warmup < samples:
        raise ValueError('warmup must be smaller than sample count')
    torch.set_num_threads(1)
    game = Pong([seed])
    vision = load_default() if checkpoint is None else ProbabilisticVisualState.load(checkpoint)
    encoder = VisualStateEncoder(64, 64, .02)
    previous = torch.zeros(64, 64, dtype=torch.bool)
    drive = torch.zeros(1)
    phase = {name: [] for name in ('render', 'events', 'vision', 'encoder', 'physics', 'total')}
    scores = hits = physics_ticks = 0
    active_features = entities = 0
    for sample in range(samples):
        started = time.perf_counter()
        frame = game.render(64, 64)[0]
        rendered = time.perf_counter()
        events = dense_events(previous, frame)
        previous.copy_(frame)
        image = frame.float()
        sensed = time.perf_counter()
        state = vision.step(events, image, learn=learn)
        predicted = time.perf_counter()
        encoded = encoder.encode(state, image)
        transported = time.perf_counter()
        ticks = physics_ticks_for_sample(sample)
        for _ in range(ticks):
            outcome = game.step(drive)
            scores += int(outcome.scores[0])
            hits += int(outcome.hits[0])
        advanced = time.perf_counter()
        physics_ticks += ticks
        if sample >= warmup:
            for name, elapsed in zip(phase, (rendered-started, sensed-rendered,
                                            predicted-sensed, transported-predicted,
                                            advanced-transported, advanced-started)):
                phase[name].append(elapsed*1000)
            active_features += len(encoded['indices'])
            entities += len(state['entities'])
    if save is not None:
        vision.save(save)
    def quantiles(values):
        p50, p95 = torch.tensor(values, dtype=torch.float64).quantile(
            torch.tensor([.5, .95], dtype=torch.float64)).tolist()
        return dict(p50_ms=p50, p95_ms=p95)
    banks = {}
    for horizon, belief in vision.dynamics.items():
        if hasattr(belief, 'short'):
            banks[f'{horizon}/short'] = len(belief.short.keys)
            banks[f'{horizon}/long'] = len(belief.long.keys)
        else:
            banks[str(horizon)] = len(belief.keys)
    timed = samples-warmup
    return dict(seed=seed, learning=learn, samples=samples, timed_samples=timed,
                physics_ticks=physics_ticks, physics_hz=120, vision_hz=50,
                phases_ms={name: quantiles(values) for name, values in phase.items()},
                mean_active_features=active_features/timed,
                mean_entities=entities/timed, scores=scores, hits=hits,
                final_bank_examples=banks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--samples', type=int, default=240)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(seed=args.seed, learn=not args.evaluation, samples=args.samples,
                 warmup=args.warmup, checkpoint=args.checkpoint, save=args.save)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
