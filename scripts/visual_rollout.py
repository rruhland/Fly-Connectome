"""Closed-loop visual forecast of the frozen opt-in learned circuit."""

import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, train_models
from diverse_visual_experience import unseen_shape_sequences
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)


OUT = Path('docs/experiments/2026-09-24-visual-rollout-results.json')
HORIZONS = (1, 2, 4)
MAX_HORIZON = max(HORIZONS)


@torch.no_grad()
def rollout_from_state(model, first, *, steps):
    saved = (model.previous, model.previous_sources,
             model.pending_prediction, model.latent)
    forecasts = [first]
    for _ in range(1, steps):
        forecasts.append(model.step((forecasts[-1] >= .5).float()))
    (model.previous, model.previous_sources,
     model.pending_prediction, model.latent) = saved
    return forecasts


@torch.no_grad()
def rollout_fixed(model, first, *, steps):
    saved = (model.previous, model.pending_features,
             model.pending_prediction)
    forecasts = [first]
    for _ in range(1, steps):
        forecasts.append(model.step((forecasts[-1] >= .5).float(),
                                    learn=False))
    (model.previous, model.pending_features,
     model.pending_prediction) = saved
    return forecasts


@torch.no_grad()
def evaluate(models, cases):
    families = {family for family, _ in cases}
    names = (*models, 'fixed')
    scores = {family: {h: {name: empty_score() for name in names}
                       for h in HORIZONS}
              for family in families}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, events in cases:
        codes = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        fixed.reset_state()
        for t in range(11):
            first = {name: model.step(codes[t])
                     for name, model in models.items()}
            fixed_first = fixed.step(events[t], learn=False)
            if t < 3:
                continue
            forecasts = {name: rollout_from_state(
                model, first[name], steps=MAX_HORIZON)
                for name, model in models.items()}
            fixed_forecasts = rollout_fixed(fixed, fixed_first,
                                            steps=MAX_HORIZON)
            for h in HORIZONS:
                target = events[t+h]
                for name, sequence in forecasts.items():
                    accumulate(scores[family][h][name],
                               primitive_to_events(sequence[h-1]), target)
                accumulate(scores[family][h]['fixed'],
                           fixed_forecasts[h-1], target)
    return {family: {h: {name: finish(value)
                         for name, value in methods.items()}
                     for h, methods in horizons.items()}
            for family, horizons in scores.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    models = train_models()
    original = [('single', events)
                for *_, events in unseen_shape_sequences()]
    robust = [(family, events)
              for family, _, events in make_robust_cases()
              if family in ('independent', 'crossing', 'speed_change')]
    scores = evaluate(models, original+robust)
    result = dict(anchor_frames=[3, 10], horizons=list(HORIZONS),
                  scores=scores, elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({family: {
        h: {name: round(value['f1'], 3) for name, value in methods.items()}
        for h, methods in horizons.items()}
        for family, horizons in scores.items()}), flush=True)


if __name__ == '__main__':
    main()
