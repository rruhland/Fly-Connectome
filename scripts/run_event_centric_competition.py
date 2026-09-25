"""Matched local heads on an event-centric recurrent visual state."""

import copy
import json
import random
import time
from collections import deque
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from delayed_visual_emission import DelayedLocalEmission
from event_centric_forecast import (CompetitiveLocalForecast,
                                    MotionWithSeparateSurface)
from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from local_hidden_transition import LocalHiddenTransition
from run_event_forecast_ranking import empty as empty_rank, rank_episode
from run_observation_horizon_forecast import (empty_score, finish,
                                              score_event, sensory_sequence)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-event-centric-competition-results.json')
ARMS = ('competitive', 'independent', 'shuffled_credit',
        'repeat_event', 'frozen')


def make_heads(units):
    return {mode: {
        'competitive': CompetitiveLocalForecast(
            units, horizon=4 if mode == 'four_frame' else None,
            next_event=mode == 'next_event'),
        'independent': DelayedLocalEmission(
            units, horizon=4 if mode == 'four_frame' else None,
            next_event=mode == 'next_event'),
        'shuffled_credit': CompetitiveLocalForecast(
            units, horizon=4 if mode == 'four_frame' else None,
            next_event=mode == 'next_event')}
            for mode in ('four_frame', 'next_event')}


@torch.no_grad()
def train_state(motion, cases):
    model = LocalHiddenTransition(MotionWithSeparateSurface(
        copy.deepcopy(motion)))
    order = list(range(len(cases)))
    random.Random(3).shuffle(order)
    for index in order:
        model.reset_state()
        for _, code, surface in sensory_sequence(cases[index]):
            model.step((code, surface), learn=True)
    return model


@torch.no_grad()
def train_heads(model, cases):
    heads = make_heads(model.encoder.units)
    order = list(range(len(cases)))
    random.Random(11).shuffle(order)
    for n, index in enumerate(order):
        sequence = sensory_sequence(cases[index])
        events = [row[0] for row in sequence]
        rng = random.Random(100+n)
        all_indices = list(range(len(events)))
        rng.shuffle(all_indices)
        shuffled_all = [events[i] for i in all_indices]
        active = [i for i, event in enumerate(events) if event.any()]
        shuffled_active = active.copy()
        rng.shuffle(shuffled_active)
        event_credit = events.copy()
        for target, source in zip(active, shuffled_active):
            event_credit[target] = events[source]
        model.reset_state()
        for group in heads.values():
            for head in group.values():
                head.reset_state()
        for t, (event, code, surface) in enumerate(sequence):
            model.step((code, surface))
            for mode, group in heads.items():
                control = (shuffled_all[t] if mode == 'four_frame'
                           else event_credit[t])
                for name, head in group.items():
                    head.step(model.state, event, learn=True,
                              credit_target=control if name ==
                              'shuffled_credit' else None)
    return heads


@torch.no_grad()
def evaluate(model, heads, cases):
    scores = {mode: {name: empty_score() for name in ARMS}
              for mode in ('four_frame', 'next_event')}
    for events in cases:
        model.reset_state()
        for group in heads.values():
            for head in group.values():
                head.reset_state()
        frames = deque()
        previous_event = None
        zero = torch.zeros_like(events[0])
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(frames) == 4:
                origin, predictions, previous = frames.popleft()
                if origin >= 4:
                    for name, prediction in predictions.items():
                        score_event(scores['four_frame'][name], prediction,
                                    event, previous, lead=4)
            if event.any() and previous_event is not None:
                origin, predictions, previous = previous_event
                if origin >= 4:
                    for name, prediction in predictions.items():
                        score_event(scores['next_event'][name], prediction,
                                    event, previous, lead=t-origin)
            model.step((code, surface))
            frame = {name: head.step(model.state, event)
                     for name, head in heads['four_frame'].items()}
            frame.update(repeat_event=event, frozen=zero)
            frames.append((t, frame, event))
            issued = {name: head.step(model.state, event)
                      for name, head in heads['next_event'].items()}
            if event.any():
                issued.update(repeat_event=event, frozen=zero)
                previous_event = (t, issued, event)
    return {mode: {name: finish(row) for name, row in group.items()}
            for mode, group in scores.items()}


@torch.no_grad()
def rank_fixed(model, head, cases):
    row = empty_rank()
    for events in cases:
        model.reset_state()
        head.reset_state()
        pending = deque()
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, forecast = pending.popleft()
                if origin >= 4 and event.any():
                    target = (event >= .5).flatten()
                    values = forecast.flatten()
                    row['targets'] += int(target.sum())
                    row['top_8_hits'] += int(target[
                        values.topk(8).indices].sum())
                    row['top_32_hits'] += int(target[
                        values.topk(32).indices].sum())
            model.step((code, surface))
            pending.append((t, head.step(model.state, event)))
    return dict(targets=row['targets'],
                top_8_recall=row['top_8_hits']/max(row['targets'], 1),
                top_32_recall=row['top_32_hits']/max(row['targets'], 1))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    state_seconds = time.perf_counter()-started
    print(f'event-centric state trained in {state_seconds:.1f}s', flush=True)
    heads = train_heads(model, training)
    training_seconds = time.perf_counter()-started
    print(f'local heads trained in {training_seconds:.1f}s total', flush=True)
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: evaluate(model, heads, cases)
              for name, cases in groups.items()}
    ranks = {}
    fixed_ranks = {}
    for group in ('generic_heldout', 'native'):
        ranks[group] = {}
        fixed_ranks[group] = {}
        for name in ('competitive', 'independent', 'shuffled_credit'):
            row = empty_rank()
            for events in groups[group]:
                rank_episode(model, heads['next_event'][name], events, row)
            ranks[group][name] = dict(
                targets=row['targets'],
                top_8_recall=row['top_8_hits']/max(row['targets'], 1),
                top_32_recall=row['top_32_hits']/max(row['targets'], 1))
            fixed_ranks[group][name] = rank_fixed(
                model, heads['four_frame'][name], groups[group])
    result = dict(training_episodes=len(training),
                  state_seconds=state_seconds,
                  training_seconds=training_seconds,
                  scores=scores, ranks=ranks, fixed_ranks=fixed_ranks,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(scores={group: {mode: {
        name: round(row['f1'], 3) for name, row in arms.items()}
        for mode, arms in modes.items()} for group, modes in scores.items()},
        native_rank={name: round(row['top_8_recall'], 3)
                     for name, row in ranks['native'].items()},
        native_fixed_rank={name: round(row['top_8_recall'], 3)
                           for name, row in fixed_ranks['native'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
