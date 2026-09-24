"""Opt-in raw-event access alongside local coincidence primitives."""

import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, train_models
from diverse_visual_experience import (diverse_training_sequences,
                                       unseen_shape_sequences)
from generic_local_transition import accumulate, empty_score, finish
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import evaluate as evaluate_local, probe


OUT = Path('docs/experiments/2026-09-24-raw-event-reanchoring-results.json')


def raw_correlation_sequence(events):
    return [torch.cat((event, coincidence))
            for event, coincidence in zip(events, correlation_sequence(events))]


def decode_correlations(prediction):
    return primitive_to_events(prediction[2:])


@torch.no_grad()
def train_raw_models():
    passes = [[(direction, events, raw_correlation_sequence(events))
               for _, direction, _, _, events in episodes]
              for episodes in diverse_training_sequences()]
    learned = TransitionPopulation(channels=18, seed=0)
    random_code = TransitionPopulation(channels=18, seed=0)
    rng = random.Random(0)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, _, codes in order:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)
    shuffled = TransitionPopulation(channels=18, seed=0)
    shuffled.dictionary.copy_(learned.dictionary)
    shuffled.usage.copy_(learned.usage)
    shuffled.total_assignments = learned.total_assignments
    rng = random.Random(1)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, _, codes in order:
            future = codes[1:].copy()
            rng.shuffle(future)
            for model in (learned, random_code, shuffled):
                model.reset_state()
            for t, code in enumerate(codes):
                learned.step(code, learn_prediction=True)
                random_code.step(code, learn_prediction=True)
                shuffled.step(code, learn_prediction=True,
                              credit_target=future[t-1] if t else None)
    return dict(learned=learned, random_dictionary=random_code,
                shuffled_future=shuffled), [case for episodes in passes
                                           for case in episodes]


def decoder_matrix(channels):
    decoder = torch.zeros((channels, 2))
    offset = channels-16
    decoder[offset:offset+8, 0] = 1
    decoder[offset+8:offset+16, 1] = 1
    return decoder


def family_event_scores(result):
    groups = {}
    for label, methods in result['by_shape_polarity'].items():
        family = label.split(':', 1)[0]
        family_scores = groups.setdefault(
            family, {name: empty_score() for name in methods})
        for name, pair in methods.items():
            for key in family_scores[name]:
                family_scores[name][key] += pair['event'][key]
    return {family: {name: finish(score) for name, score in methods.items()}
            for family, methods in groups.items()}


@torch.no_grad()
def reappearance_scores(models, baseline, raw_occlusion):
    scores = {name: empty_score() for name in (*models, 'baseline')}
    activity = dict(raw_events=0, coincidence_input=0,
                    raw_augmented_input=0, baseline_latent=0,
                    augmented_latent=0)
    for _, _, events in raw_occlusion:
        coincidence = correlation_sequence(events)
        augmented = raw_correlation_sequence(events)
        baseline.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(11):
            baseline_prediction = baseline.step(coincidence[t])
            predictions = {name: model.step(augmented[t])
                           for name, model in models.items()}
        target = events[11]
        accumulate(scores['baseline'],
                   primitive_to_events(baseline_prediction), target)
        for name, prediction in predictions.items():
            accumulate(scores[name], decode_correlations(prediction), target)
        activity['raw_events'] += int(events[10].sum())
        activity['coincidence_input'] += int(coincidence[10].sum())
        activity['raw_augmented_input'] += int(augmented[10].sum())
        activity['baseline_latent'] += int(baseline.latent.sum())
        activity['augmented_latent'] += int(models['learned'].latent.sum())
    return dict(scores={name: finish(value) for name, value in scores.items()},
                activity=activity)


@torch.no_grad()
def quiet_false_alarms(models, baseline, raw_cases):
    counts = {name: dict(frames=0, false_alarm_pixels=0)
              for name in (*models, 'baseline')}
    for family, _, events in raw_cases:
        if family not in ('occlusion', 'noise'):
            continue
        coincidence = correlation_sequence(events)
        augmented = raw_correlation_sequence(events)
        baseline.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(18):
            predictions = {'baseline': primitive_to_events(
                baseline.step(coincidence[t]))}
            predictions.update({name: decode_correlations(
                model.step(augmented[t])) for name, model in models.items()})
            if 3 <= t < 14 and events[t+1].sum() == 0:
                for name, prediction in predictions.items():
                    counts[name]['frames'] += 1
                    counts[name]['false_alarm_pixels'] += int(
                        (prediction >= .5).sum())
    return counts


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    baseline = train_models()['learned']
    models, training = train_raw_models()
    raw_single = unseen_shape_sequences()
    raw_robust = make_robust_cases()
    single = [(shape, direction, speed, background, events,
               raw_correlation_sequence(events))
              for shape, direction, speed, background, events in raw_single]
    robust = [(family, 'mixed', meta['speed'], meta['background'], events,
               raw_correlation_sequence(events))
              for family, meta, events in raw_robust]
    decoder = decoder_matrix(18)
    single_scores = evaluate_local(models, single, decoder)
    robust_scores = evaluate_local(models, robust, decoder)
    baseline_single = evaluate_local(
        {'baseline': baseline},
        [(shape, direction, speed, background, events,
          correlation_sequence(events))
         for shape, direction, speed, background, events in raw_single],
        decoder_matrix(16))
    baseline_robust = evaluate_local(
        {'baseline': baseline},
        [(family, 'mixed', meta['speed'], meta['background'], events,
          correlation_sequence(events))
         for family, meta, events in raw_robust], decoder_matrix(16))
    raw_occlusion = [case for case in raw_robust if case[0] == 'occlusion']
    reappearance = reappearance_scores(models, baseline, raw_occlusion)
    quiet = quiet_false_alarms(models, baseline, raw_robust)
    probes = {name: probe(model, training, single)
              for name, model in models.items()}
    result = dict(training_episodes=len(training),
                  single=single_scores, robust=robust_scores,
                  baseline_single=baseline_single,
                  baseline_robust=baseline_robust,
                  robust_families=family_event_scores(robust_scores),
                  baseline_families=family_event_scores(baseline_robust),
                  reappearance=reappearance, quiet=quiet,
                  direction_probe=probes,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        single={name: round(row['event']['f1'], 3)
                for name, row in single_scores['overall'].items()},
        baseline_single=round(
            baseline_single['overall']['baseline']['event']['f1'], 3),
        families={family: {name: round(row['f1'], 3)
                           for name, row in methods.items()}
                  for family, methods in result['robust_families'].items()},
        baseline_families={family: round(methods['baseline']['f1'], 3)
                           for family, methods in
                           result['baseline_families'].items()},
        reappearance={name: round(row['f1'], 3) for name, row in
                      reappearance['scores'].items()},
        activity=reappearance['activity'], quiet=quiet,
        direction={name: row['correct'] for name, row in probes.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
