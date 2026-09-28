"""Fixed-drive 120 Hz Pong physics with 50 Hz promoted visual perception."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def load_runtime(source_root=None):
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    package = (root / 'src' / 'fly_connectome').resolve()
    if not package.is_dir():
        raise ValueError(f'no fly_connectome source at {package}')
    sys.path.insert(0, str(package.parent))
    import fly_connectome
    actual = Path(fly_connectome.__file__).resolve().parent
    if actual != package:
        raise RuntimeError(f'loaded {actual}, expected {package}; use a fresh Python process')
    from fly_connectome.pong import Pong
    from fly_connectome.vision import ProbabilisticVisualState, VisualStateEncoder, load_default
    return Pong, ProbabilisticVisualState, VisualStateEncoder, load_default, str(actual)


def dense_events(previous, frame):
    """OFF then ON, equivalent to EventCamera's binary frame differences."""
    return torch.stack((previous & ~frame, ~previous & frame)).float()


def physics_ticks_for_sample(sample):
    return (sample+1)*12//5 - sample*12//5


def score_counts(scores):
    return int((scores > 0).sum()), int((scores < 0).sum())


def bank_examples(vision):
    banks = {}
    for horizon, belief in vision.dynamics.items():
        if hasattr(belief, 'short'):
            banks[f'{horizon}/short'] = len(belief.short.keys)
            banks[f'{horizon}/long'] = len(belief.long.keys)
        else:
            banks[str(horizon)] = len(belief.keys)
    return banks


@torch.no_grad()
def run(*, seed=1101, learn=True, samples=240, warmup=40,
        checkpoint=None, save=None, source_root=None):
    if not 0 <= warmup < samples:
        raise ValueError('warmup must be smaller than sample count')
    torch.set_num_threads(1)
    Pong, ProbabilisticVisualState, VisualStateEncoder, load_default, source_package = load_runtime(source_root)
    game = Pong([seed])
    vision = load_default() if checkpoint is None else ProbabilisticVisualState.load(checkpoint)
    initial_banks = bank_examples(vision)
    encoder = VisualStateEncoder(64, 64, .02)
    previous = torch.zeros(64, 64, dtype=torch.bool)
    drive = torch.zeros(1)
    phase = {name: [] for name in ('render', 'events', 'vision', 'encoder', 'physics', 'total')}
    points = misses = hits = physics_ticks = 0
    active_features = entities = long_context_forecasts = 0
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
            point, miss = score_counts(outcome.scores)
            points += point
            misses += miss
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
            for forecast in state['forecasts']:
                if forecast['evidence_age'] or forecast['id'] not in vision.histories:
                    continue
                model = vision.dynamics[forecast['horizon_samples']]
                if hasattr(model, 'uses_context') and model.uses_context(
                        vision.observed_history(forecast['id'])[1:]):
                    long_context_forecasts += 1
    if save is not None:
        vision.save(save)
    def quantiles(values):
        p50, p95 = torch.tensor(values, dtype=torch.float64).quantile(
            torch.tensor([.5, .95], dtype=torch.float64)).tolist()
        return dict(p50_ms=p50, p95_ms=p95)
    timed = samples-warmup
    mean_total_ms = sum(phase['total'])/timed
    return dict(source_package=source_package, seed=seed, learning=learn,
                samples=samples, timed_samples=timed,
                physics_ticks=physics_ticks, physics_hz=120, vision_hz=50,
                phases_ms={name: quantiles(values) for name, values in phase.items()},
                mean_total_ms=mean_total_ms,
                compute_samples_per_second=1000/mean_total_ms,
                deadline_misses_20ms=sum(value > 20 for value in phase['total']),
                mean_active_features=active_features/timed,
                mean_entities=entities/timed, points=points, misses=misses, hits=hits,
                long_context_forecasts=long_context_forecasts,
                initial_bank_examples=initial_banks,
                final_bank_examples=bank_examples(vision))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--samples', type=int, default=240)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    result = run(seed=args.seed, learn=not args.evaluation, samples=args.samples,
                 warmup=args.warmup, checkpoint=args.checkpoint, save=args.save,
                 source_root=args.source_root)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
