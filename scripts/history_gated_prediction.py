"""Local delayed-event readout from learned history-gated visual units."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, scene_sequence
from diverse_visual_experience import TEST_SHAPES, unseen_shape_sequences
from generic_local_transition import accumulate, empty_score, finish
from history_gated_code import DIRECTIONS, train_models_for_history
from run_separated_visual_state import make_training_cases
from separated_visual_state import local_update, scatter_local


OUT = Path('docs/experiments/2026-09-24-history-gated-prediction-results.json')
NAMES = ('archived', 'learned', 'random', 'trace_reset')


def expanded_interruption_cases():
    cases = []
    for shape in TEST_SHAPES:
        for direction, (dy, dx) in DIRECTIONS.items():
            for speed in (1, 2):
                for background in (True, False):
                    for center in ((13, 27), (16, 32), (19, 37)):
                        obj = dict(shape=shape, center=center,
                                   before=(dy*speed, dx*speed),
                                   after=(dy*speed, dx*speed),
                                   hidden=(7, 8, 9))
                        cases.append((f'occlusion:{direction}', shape,
                                      scene_sequence([obj],
                                                     background=background)))
    return cases


class LocalEventReadout:
    def __init__(self, code, *, eta=.5, unit_normalized=False):
        self.code = code
        self.eta = eta
        self.unit_normalized = unit_normalized
        self.weights = torch.zeros((2, code.history.units, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.code.reset_state()
        self.previous_sources = []
        self.pending_prediction = None

    @torch.no_grad()
    def step(self, events, coincidence, *, learn=False):
        if learn and self.pending_prediction is not None:
            error = events-self.pending_prediction
            if self.unit_normalized:
                unit_local_update(self.weights, self.previous_sources,
                                  error, self.eta)
            else:
                local_update(self.weights, self.previous_sources,
                             error, self.eta)
        sources = [(y, x, unit, 1.)
                   for y, x, unit in self.code.step(events, coincidence)]
        prediction = scatter_local(self.weights, sources)
        self.previous_sources = sources
        self.pending_prediction = prediction
        return prediction


def unit_local_update(weights, sources, error, eta):
    """Average eligible error independently for each active presynaptic unit."""
    if not sources:
        return
    padded = F.pad(error, (2, 2, 2, 2))
    grouped = {}
    for y, x, unit, amplitude in sources:
        row = grouped.setdefault(unit, [])
        row.append((y, x, amplitude))
    for unit, sites in grouped.items():
        denominator = max(sum(amplitude for _, _, amplitude in sites), 1.)
        update = sum((amplitude*padded[:, y:y+5, x:x+5]
                      for y, x, amplitude in sites))
        weights[:, unit] += eta/denominator*update
    weights.clamp_(0, 1)


@torch.no_grad()
def train_readouts(cases):
    codes = train_models_for_history(cases)
    archived = copy.deepcopy(codes['learned'].motion)
    models = {name: LocalEventReadout(code) for name, code in codes.items()}
    encoded = [(events, correlation_sequence(events))
               for _, _, events in cases]
    random.Random(3).shuffle(encoded)
    for events, correlations in encoded:
        for model in models.values():
            model.reset_state()
        for event, correlation in zip(events, correlations):
            for model in models.values():
                model.step(event, correlation, learn=True)
    models['trace_reset'] = copy.deepcopy(models['learned'])
    return archived, models


@torch.no_grad()
def evaluate(archived, models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    overall = {name: empty_score() for name in NAMES}
    exact = {name: empty_score() for name in NAMES}
    exact_groups = {family: {name: empty_score() for name in NAMES}
                    for family, _, _ in cases
                    if family == 'interruption' or family.startswith('occlusion')}
    quiet = {name: dict(frames=0, false_alarm_pixels=0) for name in NAMES}
    for family, _, events in cases:
        correlations = correlation_sequence(events)
        archived.reset_state()
        for model in models.values():
            model.reset_state()
        for t, (event, correlation) in enumerate(zip(events, correlations)):
            if (family == 'interruption' or family.startswith('occlusion')) and t == 10:
                models['trace_reset'].code.trace.zero_()
            baseline = primitive_to_events(archived.step(correlation))
            predictions = {'archived': baseline}
            for name, model in models.items():
                predictions[name] = torch.maximum(
                    baseline, model.step(event, correlation))
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    accumulate(overall[name], prediction, target)
                    if (family == 'interruption' or family.startswith('occlusion')) and t == 10:
                        accumulate(exact[name], prediction, target)
                        accumulate(exact_groups[family][name], prediction, target)
                    if target.sum() == 0:
                        quiet[name]['frames'] += 1
                        quiet[name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(overall={name: finish(row) for name, row in overall.items()},
                groups={family: {name: finish(row) for name, row in rows.items()}
                        for family, rows in groups.items()},
                exact_reappearance={name: finish(row) for name, row in exact.items()},
                exact_by_family={family: {name: finish(row)
                                          for name, row in rows.items()}
                                 for family, rows in exact_groups.items()},
                quiet=quiet)


@torch.no_grad()
def matched_pair(archived, models):
    pair = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        correlations = correlation_sequence(events)
        archived.reset_state()
        for model in models.values():
            model.reset_state()
        for t, (event, correlation) in enumerate(
                zip(events[:11], correlations[:11])):
            if t == 10:
                models['trace_reset'].code.trace.zero_()
            baseline = primitive_to_events(archived.step(correlation))
            predictions = {name: torch.maximum(
                baseline, model.step(event, correlation))
                for name, model in models.items()}
        pair.append((predictions, events[11]))
    scores = {}
    for name in models:
        score = empty_score()
        for predictions, target in pair:
            accumulate(score, predictions[name], target)
        scores[name] = dict(**finish(score),
            opposite_forecast_l1=float((pair[0][0][name]-pair[1][0][name])
                                       .abs().sum()))
    return scores


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = make_training_cases()
    archived, models = train_readouts(training)
    training_scores = evaluate(archived, models, training)
    heldout = [('single', shape, events)
               for shape, _, _, _, events in unseen_shape_sequences()]
    heldout += [(family, 'unseen', events)
                for family, _, events in make_robust_cases()]
    heldout_scores = evaluate(archived, models, heldout)
    expanded_scores = evaluate(archived, models,
                               expanded_interruption_cases())
    pair = matched_pair(archived, models)
    result = dict(training_episodes=len(training),
                  training=training_scores, heldout=heldout_scores,
                  expanded=expanded_scores,
                  matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training_exact={name: round(row['f1'], 3)
                        for name, row in
                        training_scores['exact_reappearance'].items()},
        heldout={family: {name: round(row['f1'], 3)
                          for name, row in rows.items()}
                 for family, rows in heldout_scores['groups'].items()},
        heldout_exact={name: round(row['f1'], 3)
                       for name, row in
                       heldout_scores['exact_reappearance'].items()},
        expanded_exact={name: round(row['f1'], 3)
                        for name, row in
                        expanded_scores['exact_reappearance'].items()},
        expanded_by_direction={family: {name: round(row['f1'], 3)
                                         for name, row in rows.items()}
                               for family, rows in
                               expanded_scores['exact_by_family'].items()},
        quiet=heldout_scores['quiet'],
        pair={name: dict(f1=round(row['f1'], 3),
                         opposite_l1=round(row['opposite_forecast_l1'], 3))
              for name, row in pair.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
