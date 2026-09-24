"""Opt-in local trace binding of learned motion state to raw first sightings."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, scene_sequence, train_models
from diverse_visual_experience import unseen_shape_sequences
from generic_local_transition import accumulate, empty_score, finish
from learned_transition_units import TransitionPopulation
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-24-local-context-trace-results.json')
NAMES = ('archived', 'trace', 'zero_trace', 'trace_reset')


def context_code(events, coincidence, trace, *, use_trace):
    onset = ((events.sum(0) > 0) & (coincidence.sum(0) == 0))
    history = (F.max_pool2d(trace[None], 17, stride=1, padding=8)[0]
               if use_trace and onset.any() and trace.any()
               else torch.zeros_like(trace))
    return torch.cat((events, history))*onset


class ContextPathway:
    def __init__(self, motion, *, use_trace):
        self.motion = motion
        self.context = TransitionPopulation(channels=26, units=8, seed=0)
        self.use_trace = use_trace
        self.reset_state()

    def reset_state(self):
        self.motion.reset_state()
        self.context.reset_state()
        self.trace = torch.zeros((self.motion.units, 32, 64))
        self.last_code = torch.zeros((26, 32, 64))

    @torch.no_grad()
    def step(self, events, coincidence, *, learn_dictionary=False,
             learn_prediction=False):
        self.motion.step(coincidence)
        code = context_code(events, coincidence, self.trace,
                            use_trace=self.use_trace)
        target = torch.cat((events, torch.zeros_like(self.trace)))
        prediction = self.context.step(
            code, learn_dictionary=learn_dictionary,
            learn_prediction=learn_prediction, credit_target=target)
        self.trace.mul_(.8).add_(self.motion.latent).clamp_(0, 1)
        self.last_code = code
        return prediction[:2]


@torch.no_grad()
def train_candidates(cases):
    archived = train_models()['learned']
    models = dict(
        trace=ContextPathway(copy.deepcopy(archived), use_trace=True),
        zero_trace=ContextPathway(copy.deepcopy(archived), use_trace=False))
    encoded = [(events, correlation_sequence(events))
               for _, _, events in cases]
    rng = random.Random(3)
    for _ in range(2):
        rng.shuffle(encoded)
        for events, codes in encoded:
            for model in models.values():
                model.reset_state()
            for event, code in zip(events, codes):
                for model in models.values():
                    model.step(event, code, learn_dictionary=True)
    rng.shuffle(encoded)
    for events, codes in encoded:
        for model in models.values():
            model.reset_state()
        for event, code in zip(events, codes):
            for model in models.values():
                model.step(event, code, learn_prediction=True)
    models['trace_reset'] = copy.deepcopy(models['trace'])
    return archived, models


@torch.no_grad()
def evaluate(archived, models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    overall = {name: empty_score() for name in NAMES}
    exact = {name: empty_score() for name in NAMES}
    quiet = {name: dict(frames=0, false_alarm_pixels=0) for name in NAMES}
    activity = dict(onset_sites=0, traced_onset_sites=0)
    for family, _, events in cases:
        codes = correlation_sequence(events)
        archived.reset_state()
        for model in models.values():
            model.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            if family == 'occlusion' and t == 10:
                models['trace_reset'].trace.zero_()
            baseline = primitive_to_events(archived.step(code))
            predictions = {'archived': baseline}
            for name, model in models.items():
                local = model.step(event, code)
                predictions[name] = torch.maximum(baseline, local)
            if family == 'occlusion' and t == 10:
                activity['onset_sites'] += int(
                    (models['trace'].last_code[:2].sum(0) > 0).sum())
                activity['traced_onset_sites'] += int(
                    (models['trace'].last_code[2:].sum(0) > 0).sum())
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    accumulate(overall[name], prediction, target)
                    if family == 'occlusion' and t == 10:
                        accumulate(exact[name], prediction, target)
                    if target.sum() == 0:
                        quiet[name]['frames'] += 1
                        quiet[name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(overall={name: finish(row) for name, row in overall.items()},
                groups={family: {name: finish(row) for name, row in rows.items()}
                        for family, rows in groups.items()},
                exact_reappearance={name: finish(row)
                                    for name, row in exact.items()},
                quiet=quiet, activity=activity)


@torch.no_grad()
def matched_pair(archived, models):
    pairs = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        codes = correlation_sequence(events)
        archived.reset_state()
        for model in models.values():
            model.reset_state()
        for t, (event, code) in enumerate(zip(events[:11], codes[:11])):
            if t == 10:
                models['trace_reset'].trace.zero_()
            baseline = primitive_to_events(archived.step(code))
            forecasts = {name: torch.maximum(baseline,
                         model.step(event, code)) for name, model in models.items()}
        pairs.append((forecasts, events[11],
                      models['trace'].last_code.clone()))
    scores = {}
    for name in models:
        score = empty_score()
        for forecasts, target, _ in pairs:
            accumulate(score, forecasts[name], target)
        scores[name] = dict(**finish(score),
            opposite_forecast_l1=float((pairs[0][0][name]-pairs[1][0][name])
                                       .abs().sum()))
    return dict(scores=scores,
                opposite_context_l1=float((pairs[0][2]-pairs[1][2])
                                          .abs().sum()))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = make_training_cases()
    archived, models = train_candidates(training)
    training_scores = evaluate(archived, models, training)
    heldout = [('single', shape, events)
               for shape, _, _, _, events in unseen_shape_sequences()]
    heldout += [(family, 'unseen', events)
                for family, _, events in make_robust_cases()]
    heldout_scores = evaluate(archived, models, heldout)
    pair = matched_pair(archived, models)
    result = dict(training_episodes=len(training), training=training_scores,
                  heldout=heldout_scores, matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        heldout={family: {name: round(score['f1'], 3)
                          for name, score in rows.items()}
                 for family, rows in heldout_scores['groups'].items()},
        reappearance={name: round(score['f1'], 3)
                      for name, score in
                      heldout_scores['exact_reappearance'].items()},
        quiet=heldout_scores['quiet'], activity=heldout_scores['activity'],
        pair=dict(context_l1=round(pair['opposite_context_l1'], 3),
                  scores={name: dict(f1=round(row['f1'], 3),
                                     opposite_l1=round(
                                         row['opposite_forecast_l1'], 3))
                          for name, row in pair['scores'].items()}),
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
