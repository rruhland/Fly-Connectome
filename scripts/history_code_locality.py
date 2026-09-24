"""Post-hoc test of site versus neighborhood direction information."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence
from diverse_visual_experience import TRAIN_SHAPES, TEST_SHAPES
from history_gated_code import (DIRECTIONS, direction_probe,
                                interrupted_cases, train_models_for_history)
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-24-history-code-locality-results.json')
SCOPES = (1, 5, 9)


@torch.no_grad()
def site_features(model, events):
    model.reset_state()
    codes = correlation_sequence(events)
    for t in range(11):
        model.step(events[t], codes[t])
    latent = model.history.latent
    sources = model.history.previous_sources
    pooled = {scope: F.avg_pool2d(latent[None], scope, stride=1,
                                   padding=scope//2)[0]
              for scope in SCOPES}
    return {scope: torch.stack([pooled[scope][:, y, x]
                                for y, x, _ in sources])
            if sources else torch.empty((0, model.history.units))
            for scope in SCOPES}


@torch.no_grad()
def locality_probe(model, training, heldout):
    prototypes = {scope: {direction: torch.zeros(model.history.units)
                          for direction in DIRECTIONS}
                  for scope in SCOPES}
    counts = {scope: {direction: 0 for direction in DIRECTIONS}
              for scope in SCOPES}
    for _, direction, _, _, events in training:
        features = site_features(model, events)
        for scope in SCOPES:
            prototypes[scope][direction] += features[scope].sum(0)
            counts[scope][direction] += len(features[scope])
    for scope in SCOPES:
        for direction in DIRECTIONS:
            prototype = prototypes[scope][direction]
            prototype /= max(counts[scope][direction], 1)
            prototype /= prototype.norm().clamp(min=1e-6)
    heldout_features = [(direction, site_features(model, events))
                        for _, direction, _, _, events in heldout]
    results = {}
    for scope in SCOPES:
        correct = total = 0
        by_direction = {direction: dict(correct=0, total=0)
                        for direction in DIRECTIONS}
        for direction, episode_features in heldout_features:
            features = episode_features[scope]
            if not len(features):
                continue
            scores = torch.stack([features @ prototypes[scope][candidate]
                                  for candidate in DIRECTIONS], dim=1)
            predictions = scores.argmax(1)
            index = tuple(DIRECTIONS).index(direction)
            hits = int((predictions == index).sum())
            correct += hits
            total += len(features)
            by_direction[direction]['correct'] += hits
            by_direction[direction]['total'] += len(features)
        results[scope] = dict(correct=correct, total=total,
                              accuracy=correct/total if total else 0,
                              by_direction=by_direction)
    return results


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    models = train_models_for_history(make_training_cases())
    training = interrupted_cases(TRAIN_SHAPES)
    heldout = interrupted_cases(TEST_SHAPES)
    locality = {name: locality_probe(model, training, heldout)
                for name, model in models.items()}
    episode = {name: direction_probe(model, training, heldout)
               for name, model in models.items()}
    result = dict(training_cases=len(training), heldout_cases=len(heldout),
                  locality=locality, episode=episode,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        locality={name: {scope: round(row['accuracy'], 3)
                         for scope, row in methods.items()}
                  for name, methods in locality.items()},
        episode={name: round(row['accuracy'], 3)
                 for name, row in episode.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
