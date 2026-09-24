"""Frozen opt-in test of local predicted-primitive feedback to inference."""

import copy
import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, train_models
from diverse_visual_experience import unseen_shape_sequences
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from run_learned_transition_units import evaluate as evaluate_local


OUT = Path('docs/experiments/2026-09-24-predictive-feedback-results.json')


class FeedbackInference:
    def __init__(self, model, *, gain):
        self.model = model
        self.gain = gain

    def reset_state(self):
        self.model.reset_state()

    def step(self, observed):
        prior = self.model.pending_prediction
        if prior is not None:
            observed = torch.maximum(observed,
                                     (prior >= .5).float()*self.gain)
        return self.model.step(observed)


@torch.no_grad()
def family_scores(scores, raw_cases):
    families = {family for family, _, _ in raw_cases}
    names = ('baseline', 'feedback', 'fixed')
    grouped = {family: {name: empty_score() for name in names}
               for family in families}
    for label, methods in scores['by_shape_polarity'].items():
        family = label.split(':', 1)[0]
        for name, row in methods.items():
            for key in grouped[family][name]:
                grouped[family][name][key] += row['event'][key]
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, _, events in raw_cases:
        fixed.reset_state()
        for t, event in enumerate(events):
            forecast = fixed.step(event, learn=False)
            if 3 <= t < 14:
                accumulate(grouped[family]['fixed'], forecast, events[t+1])
    return {family: {name: finish(row) for name, row in methods.items()}
            for family, methods in grouped.items()}


@torch.no_grad()
def quiet_diagnostics(models, raw_cases):
    diagnostics = {name: dict(quiet_targets=0, false_alarms=0,
                              blank_inputs=0, active_blank_states=0)
                   for name in models}
    for family, _, events in raw_cases:
        if family not in ('occlusion', 'noise'):
            continue
        codes = correlation_sequence(events)
        for name, model in models.items():
            model.reset_state()
            row = diagnostics[name]
            for t, code in enumerate(codes):
                forecast = primitive_to_events(model.step(code))
                if code.sum() == 0:
                    row['blank_inputs'] += 1
                    state = model.model.latent if isinstance(
                        model, FeedbackInference) else model.latent
                    row['active_blank_states'] += int(state.sum() > 0)
                if 3 <= t < 14 and events[t+1].sum() == 0:
                    row['quiet_targets'] += 1
                    row['false_alarms'] += int((forecast >= .5).sum())
    return diagnostics


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    trained = train_models()['learned']
    models = {'baseline': trained,
              'feedback': FeedbackInference(copy.deepcopy(trained), gain=.5)}
    decoder = torch.zeros((16, 2))
    decoder[:8, 0] = 1
    decoder[8:, 1] = 1
    original = [(shape, direction, speed, background, events,
                 correlation_sequence(events))
                for shape, direction, speed, background, events
                in unseen_shape_sequences()]
    original_scores = evaluate_local(models, original, decoder)
    raw_robust = make_robust_cases()
    robust = [(family, 'mixed', meta['speed'], meta['background'], events,
               correlation_sequence(events))
              for family, meta, events in raw_robust]
    robust_scores = evaluate_local(models, robust, decoder)
    grouped = family_scores(robust_scores, raw_robust)
    quiet = quiet_diagnostics(models, raw_robust)
    result = dict(original=original_scores, robust=robust_scores,
                  families=grouped, quiet=quiet,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        original={name: round(row['event']['f1'], 3)
                  for name, row in original_scores['overall'].items()},
        families={family: {name: round(row['f1'], 3)
                           for name, row in methods.items()}
                  for family, methods in grouped.items()},
        quiet=quiet, elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
