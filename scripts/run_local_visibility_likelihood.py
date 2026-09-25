"""Opt-in local visibility likelihood on matched visual histories."""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import scene_sequence
from diverse_visual_experience import TRAIN_SHAPES, unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT
from history_gated_code import DIRECTIONS
from local_visibility_likelihood import LocalVisibilityHead, nearby_events
from occlusion_identifiability import WINDOWS, matched_prefix_suite
from run_local_hidden_transition import train as train_transition
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-25-local-visibility-results.json')
ARMS = ('learned', 'shuffled_time', 'fixed_hazard')


def clean_training_cases():
    return [(kind, events) for kind, _, events in make_training_cases()
            if kind == 'clean']


def training_cases():
    cases = clean_training_cases()
    for shape in TRAIN_SHAPES:
        for direction, (dy, dx) in DIRECTIONS.items():
            for speed in (1, 2):
                for background in (True, False):
                    for center in ((12, 24), (20, 40)):
                        for hidden in WINDOWS:
                            obj = dict(shape=shape, center=center,
                                       before=(dy*speed, dx*speed),
                                       after=(dy*speed, dx*speed),
                                       hidden=hidden)
                            cases.append(('gap', scene_sequence(
                                [obj], background=background)))
    return cases


@torch.no_grad()
def train_heads(transition, cases):
    heads = {name: LocalVisibilityHead(eta=.05)
             for name in ('learned', 'shuffled_time')}
    order = list(range(len(cases)))
    rng = random.Random(3)
    rng.shuffle(order)
    prior_positive = prior_count = 0
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
                prior_positive += int(nearby_events(event)[y, x].sum())
                prior_count += len(sites)
            heads['learned'].step(transition, event, learn=True)
            heads['shuffled_time'].step(
                transition, event, learn=True,
                credit_target=events[times[t]])
    return heads, prior_positive/max(prior_count, 1)


def empty_score():
    return dict(brier_sum=0., predicted_sum=0., target_sum=0., count=0)


def add_score(row, probability, target):
    row['brier_sum'] += (probability-target)**2
    row['predicted_sum'] += probability
    row['target_sum'] += target
    row['count'] += 1


def finish(row):
    return {**row, 'brier': row['brier_sum']/max(row['count'], 1),
            'mean_probability': row['predicted_sum']/max(row['count'], 1),
            'event_rate': row['target_sum']/max(row['count'], 1)}


@torch.no_grad()
def evaluate(transition, heads, prior, clean, matched):
    global_scores = {phase: {name: empty_score() for name in ARMS}
                     for phase in ('overall', 'visible', 'quiet', 'gap_exit')}
    local_scores = {name: empty_score() for name in ARMS}
    pairs = {name: empty_score() for name in ARMS}
    identical = dict(two_vs_three=0, three_vs_four=0)
    candidate = dict(frames=0, active_frames=0,
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
            fixed = torch.zeros((32, 64))
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
            candidate['frames'] += 1
            if sites is None:
                candidate['event_pixels'] += int(target.sum())
                continue
            candidate['active_frames'] += 1
            y, x = sites.T
            labels = nearby_events(target)[y, x]
            for name, probabilities in maps.items():
                values = probabilities[y, x]
                local_scores[name]['brier_sum'] += float(
                    ((values-labels)**2).sum())
                local_scores[name]['predicted_sum'] += float(values.sum())
                local_scores[name]['target_sum'] += float(labels.sum())
                local_scores[name]['count'] += len(sites)
            footprint = torch.zeros((32, 64))
            footprint[y, x] = 1
            covered = F.max_pool2d(footprint[None, None], 5,
                                   stride=1, padding=2)[0, 0] > 0
            pixels = target.sum(0) > 0
            candidate['event_pixels'] += int(pixels.sum())
            candidate['covered_event_pixels'] += int((covered & pixels).sum())
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
            assert sequences[left][t+1].sum() > 0
            assert sequences[right][t+1].sum() == 0
            for name in ARMS:
                probability = float(snapshots[left][t][name].max())
                add_score(pairs[name], probability, 1)
                add_score(pairs[name], probability, 0)

    return dict(global_scores={phase: {name: finish(row)
                                       for name, row in arms.items()}
                               for phase, arms in global_scores.items()},
                local_scores={name: finish(row)
                              for name, row in local_scores.items()},
                matched_pairs={name: finish(row)
                               for name, row in pairs.items()},
                identical_prefix_probabilities=identical,
                candidate={**candidate,
                           'event_pixel_coverage': candidate[
                               'covered_event_pixels']/max(
                                   candidate['event_pixels'], 1)})


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
    print(f'transition and visibility head trained on {len(cases)} '
          f'episodes in {trained:.1f}s', flush=True)
    matched = matched_prefix_suite()
    result = dict(training_episodes=len(cases),
                  clean_heldout=len(unseen_shape_sequences()),
                  matched_configurations=len(matched),
                  empirical_local_hazard=prior,
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
        local_brier={name: round(row['brier'], 4) for name, row in
                     result['evaluation']['local_scores'].items()},
        matched_brier={name: round(row['brier'], 4) for name, row in
                       result['evaluation']['matched_pairs'].items()},
        matched_probability={name: round(row['mean_probability'], 3)
                             for name, row in
                             result['evaluation']['matched_pairs'].items()},
        identical=result['evaluation']['identical_prefix_probabilities'],
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
