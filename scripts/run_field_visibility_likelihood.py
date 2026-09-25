"""Bounded test of visibility credit pooled over local retinotopic fields."""

import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from diverse_visual_experience import unseen_shape_sequences
from field_visibility_likelihood import FieldVisibilityHead, field_events
from gap_timing_transfer import MODEL_OUT
from occlusion_identifiability import WINDOWS, matched_prefix_suite
from run_local_hidden_transition import train as train_transition
from run_local_visibility_likelihood import (add_score, clean_training_cases,
                                            empty_score, finish,
                                            training_cases)


OUT = Path('docs/experiments/2026-09-25-field-visibility-results.json')
ARMS = ('learned', 'shuffled_time', 'fixed_hazard')


@torch.no_grad()
def train_heads(transition, cases):
    heads = {name: FieldVisibilityHead(eta=.05)
             for name in ('learned', 'shuffled_time')}
    order = list(range(len(cases)))
    rng = random.Random(3)
    rng.shuffle(order)
    positive = count = 0
    for index in order:
        events = cases[index][1]
        codes = correlation_sequence(events)
        times = list(range(len(events)))
        rng.shuffle(times)
        transition.reset_state()
        for head in heads.values():
            head.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            transition.step(code)
            sites = heads['learned'].previous_sites
            if sites is not None:
                y, x = sites.T
                positive += int(field_events(event)[y, x].sum())
                count += len(sites)
            heads['learned'].step(transition, event, learn=True)
            heads['shuffled_time'].step(
                transition, event, learn=True,
                credit_target=events[times[t]])
    return heads, positive/max(count, 1)


@torch.no_grad()
def evaluate(transition, heads, prior, clean, matched):
    global_scores = {phase: {name: empty_score() for name in ARMS}
                     for phase in ('overall', 'visible', 'quiet', 'gap_exit')}
    field_scores = {name: empty_score() for name in ARMS}
    pairs = {name: empty_score() for name in ARMS}
    identical = dict(two_vs_three=0, three_vs_four=0)
    coverage = dict(positive_frames=0, uncovered_frames=0,
                    event_pixels=0, covered_event_pixels=0)

    def replay(events, hidden=None):
        transition.reset_state()
        for head in heads.values():
            head.reset_state()
        codes = correlation_sequence(events)
        snapshots = {}
        for t in range(14):
            transition.step(codes[t])
            maps = {name: head.step(transition, events[t]).clone()
                    for name, head in heads.items()}
            sites = heads['learned'].previous_sites
            fixed = torch.zeros((3, 7))
            if sites is not None:
                y, x = sites.T
                fixed[y, x] = prior
            maps['fixed_hazard'] = fixed
            if t in (8, 9):
                snapshots[t] = maps
            if t < 3:
                continue
            target = events[t+1]
            target_any = int(target.sum() > 0)
            phase = 'visible' if target_any else 'quiet'
            for name, probabilities in maps.items():
                value = float(probabilities.max())
                add_score(global_scores['overall'][name], value, target_any)
                add_score(global_scores[phase][name], value, target_any)
                if hidden is not None and t == hidden[-1]:
                    add_score(global_scores['gap_exit'][name], value,
                              target_any)
            if target_any:
                coverage['positive_frames'] += 1
            if sites is None:
                if target_any:
                    coverage['uncovered_frames'] += 1
                coverage['event_pixels'] += int(target.sum())
                continue
            y, x = sites.T
            labels = field_events(target)[y, x]
            for name, probabilities in maps.items():
                values = probabilities[y, x]
                row = field_scores[name]
                row['brier_sum'] += float(((values-labels)**2).sum())
                row['predicted_sum'] += float(values.sum())
                row['target_sum'] += float(labels.sum())
                row['count'] += len(sites)
            footprint = torch.zeros((32, 64), dtype=torch.bool)
            for fy, fx in sites.tolist():
                footprint[fy*8:fy*8+16, fx*8:fx*8+16] = True
            pixels = target.sum(0) > 0
            coverage['event_pixels'] += int(pixels.sum())
            coverage['covered_event_pixels'] += int((pixels & footprint).sum())
        return snapshots

    for events in clean:
        replay(events)
    for *_, sequences in matched:
        snapshots = [replay(events, hidden)
                     for events, hidden in zip(sequences, WINDOWS)]
        for label, left, right, t in (
                ('two_vs_three', 0, 1, 8),
                ('three_vs_four', 1, 2, 9)):
            identical[label] += int(torch.equal(
                snapshots[left][t]['learned'],
                snapshots[right][t]['learned']))
            for name in ARMS:
                probability = float(snapshots[left][t][name].max())
                add_score(pairs[name], probability, 1)
                add_score(pairs[name], probability, 0)
    return dict(global_scores={phase: {name: finish(row)
                                       for name, row in arms.items()}
                               for phase, arms in global_scores.items()},
                field_scores={name: finish(row)
                              for name, row in field_scores.items()},
                matched_pairs={name: finish(row)
                               for name, row in pairs.items()},
                identical_prefix_probabilities=identical,
                coverage={**coverage,
                          'event_pixel_coverage': coverage[
                              'covered_event_pixels']/max(
                                  coverage['event_pixels'], 1)})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    transition = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])['learned']
    cases = training_cases()
    heads, prior = train_heads(transition, cases)
    trained = time.perf_counter()-started
    print(f'field head trained on {len(cases)} episodes in '
          f'{trained:.1f}s', flush=True)
    matched = matched_prefix_suite()
    result = dict(training_episodes=len(cases),
                  clean_heldout=len(unseen_shape_sequences()),
                  matched_configurations=len(matched),
                  empirical_field_hazard=prior,
                  head_weights={name: head.weights.tolist()
                                for name, head in heads.items()},
                  evaluation=evaluate(
                      transition, heads, prior,
                      [events for *_, events in unseen_shape_sequences()],
                      matched),
                  training_seconds=trained,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        field_brier={name: round(row['brier'], 4) for name, row in
                     result['evaluation']['field_scores'].items()},
        overall_brier={name: round(row['brier'], 4) for name, row in
                       result['evaluation']['global_scores']['overall'].items()},
        quiet_brier={name: round(row['brier'], 4) for name, row in
                     result['evaluation']['global_scores']['quiet'].items()},
        matched_brier={name: round(row['brier'], 4) for name, row in
                       result['evaluation']['matched_pairs'].items()},
        matched_probability={name: round(row['mean_probability'], 3)
                             for name, row in
                             result['evaluation']['matched_pairs'].items()},
        coverage=round(result['evaluation']['coverage'][
            'event_pixel_coverage'], 3),
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
