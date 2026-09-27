"""Horizon-specific local credit, with causally delayed observed endpoints."""

import copy
import json
import math
from pathlib import Path

import torch

from event_only_observer_teacher import fit_event_only_observer
from frame_observation_state import FrameObservationState
from generic_native_cadence import scene_events
from local_motion_dynamics import LocalMotionDynamics
from run_local_motion_dynamics import KINDS, visual_episode, observed_positions, fixed_displacement
from run_zero_shot_visual_transfer import corrupt_events


HORIZONS = (1, 4, 8)


def history_at(positions, t):
    if t < 4 or not all(p is not None for p in positions[t-4:t+1]):
        return None
    prior = torch.stack(positions[t-4:t+1])
    return prior[1:]-prior[:-1]


@torch.no_grad()
def forecast(models, positions, *, online=False):
    models = copy.deepcopy(models)
    targets = {h: [] for h in HORIZONS}
    generator = torch.Generator().manual_seed(992)
    result = {}
    for t in range(len(positions)):
        for h in HORIZONS:
            past = history_at(positions, t-h)
            if online and past is not None and positions[t] is not None:
                displacement = positions[t]-positions[t-h]
                models['learned'][h].observe(past, displacement)
                if targets[h]:
                    index = int(torch.randint(len(targets[h]), (), generator=generator))
                    models['shuffled'][h].observe(past, targets[h][index])
                targets[h].append(displacement)
        if t < 6:
            continue
        history = history_at(positions, t)
        for h in HORIZONS:
            row = {}
            for name in ('learned', 'shuffled', 'persistence', 'velocity', 'acceleration', 'lag_two'):
                if history is None:
                    row[name] = None
                    continue
                displacement = (models[name][h].predict(history) if name in models
                                else fixed_displacement(history, h, name))
                row[name] = positions[t]+displacement
            result[t, h] = row
    return result


@torch.no_grad()
def main(*, model_type=LocalMotionDynamics,
         out=Path('docs/experiments/2026-09-26-direct-dynamics-results.json')):
    torch.set_num_threads(1)
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    examples = {h: [] for h in HORIZONS}
    for kind in KINDS:
        for phase in (0, 2):
            for shape in ('dot', 'square'):
                case, _ = visual_episode(kind, phase, shape, background=bool(phase))
                positions = observed_positions(base, case, state_type=FrameObservationState)
                for t in range(4, len(positions)):
                    history = history_at(positions, t)
                    for h in HORIZONS:
                        if history is not None and t+h < len(positions) and positions[t+h] is not None:
                            examples[h].append((history, positions[t+h]-positions[t]))
    models = {name: {h: model_type() for h in HORIZONS}
              for name in ('learned', 'shuffled')}
    for h, pairs in examples.items():
        permutation = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(991))
        for i, (history, target) in enumerate(pairs):
            models['learned'][h].observe(history, target)
            models['shuffled'][h].observe(history, pairs[int(permutation[i])][1])
    results = dict(sensor='current visible frames plus events', teacher='matched observed endpoints',
                   model=model_type.__name__,
                   pairs={h: len(v) for h, v in examples.items()}, arms={})
    for online in (False, True):
        errors = {kind: {h: {} for h in HORIZONS} for kind in KINDS}
        for kind in KINDS:
            for phase in (3, 5):
                for angle in (0., math.pi/4, math.pi/2):
                    case, truth = visual_episode(kind, phase, 'plus', angle=angle,
                                                 scale=1.25, background=bool(phase % 2))
                    positions = observed_positions(base, case, state_type=FrameObservationState)
                    forecasts = forecast(models, positions, online=online)
                    for (t, h), row in forecasts.items():
                        if t+h >= len(truth):
                            continue
                        for name, value in row.items():
                            error = 64. if value is None else float((value-truth[t+h]).norm())
                            errors[kind][h].setdefault(name, []).append(error)
        mean = lambda values: sum(values)/len(values)
        families = {k: {h: {n: mean(v) for n, v in row.items()}
                        for h, row in arm.items()} for k, arm in errors.items()}
        totals = {h: {n: mean([v for k in KINDS for v in errors[k][h][n]])
                      for n in errors[KINDS[0]][h]} for h in HORIZONS}
        results['arms']['online' if online else 'frozen'] = dict(families=families, means=totals)
    out.write_text(
        json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
