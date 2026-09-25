"""Learn local conjunction units over causal event history, then forecast."""

import copy
import json
import random
import time
from collections import deque
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from event_budget_probe import select_by_mass
from event_centric_forecast import CompetitiveLocalForecast
from event_lag_stack import lagged_event_sequence
from generic_native_cadence import scene_events
from learned_transition_units import TransitionPopulation
from local_hidden_transition import LocalHiddenTransition
from run_observation_horizon_forecast import (empty_score, finish,
                                              score_event)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-learned-sequence-units-results.json')
ARMS = ('direct', 'recurrent', 'shuffled_recurrent')


@torch.no_grad()
def train_encoder(cases):
    encoder = TransitionPopulation(channels=16, seed=0)
    for events in cases:
        encoder.reset_state()
        for sensory in lagged_event_sequence(events):
            encoder.step(sensory, learn_dictionary=True)
    return encoder


@torch.no_grad()
def train_state(encoder, cases):
    model = LocalHiddenTransition(copy.deepcopy(encoder))
    order = list(range(len(cases)))
    random.Random(3).shuffle(order)
    for index in order:
        model.reset_state()
        for sensory in lagged_event_sequence(cases[index]):
            model.step(sensory, learn=True)
    return model


@torch.no_grad()
def train_heads(model, cases):
    heads = {name: CompetitiveLocalForecast(model.encoder.units, horizon=4)
             for name in ARMS}
    order = list(range(len(cases)))
    random.Random(11).shuffle(order)
    for n, index in enumerate(order):
        events = cases[index]
        stacks = lagged_event_sequence(events)
        indices = list(range(len(events)))
        random.Random(100+n).shuffle(indices)
        shuffled = [events[i] for i in indices]
        model.reset_state()
        for head in heads.values():
            head.reset_state()
        for t, (event, sensory) in enumerate(zip(events, stacks)):
            model.step(sensory)
            heads['direct'].step(model.observed, event, learn=True)
            heads['recurrent'].step(model.state, event, learn=True)
            heads['shuffled_recurrent'].step(
                model.state, event, learn=True,
                credit_target=shuffled[t])
    return heads


def empty():
    return dict(frames=0, targets=0, mass=0., active_frames=0,
                active_mass=0., quiet_frames=0, quiet_mass=0.,
                rank_targets=0, top_8_hits=0,
                threshold=empty_score(), budget=empty_score(),
                source_sites=0)


@torch.no_grad()
def probe(model, heads, cases, *, gains=None):
    result = {name: empty() for name in ARMS}
    baselines = {name: empty_score() for name in ('repeat_event', 'frozen')}
    for events in cases:
        model.reset_state()
        for head in heads.values():
            head.reset_state()
        pending = deque()
        zero = torch.zeros_like(events[0])
        for t, (event, sensory) in enumerate(zip(
                events, lagged_event_sequence(events))):
            if len(pending) == 4:
                origin, predictions, previous, source_counts = pending.popleft()
                if origin >= 4:
                    target = (event >= .5).flatten()
                    score_event(baselines['repeat_event'], previous, event,
                                previous, lead=4)
                    score_event(baselines['frozen'], zero, event,
                                previous, lead=4)
                    for name, prediction in predictions.items():
                        row = result[name]
                        mass = float(prediction.sum())
                        row['frames'] += 1
                        row['targets'] += int(event.sum())
                        row['mass'] += mass
                        row['source_sites'] += source_counts[name]
                        if event.any():
                            row['active_frames'] += 1
                            row['active_mass'] += mass
                            row['rank_targets'] += int(target.sum())
                            row['top_8_hits'] += int(target[
                                prediction.flatten().topk(8).indices].sum())
                        else:
                            row['quiet_frames'] += 1
                            row['quiet_mass'] += mass
                        score_event(row['threshold'], prediction, event,
                                    previous, lead=4)
                        if gains is not None:
                            selected = select_by_mass(prediction,
                                                      gain=gains[name])
                            score_event(row['budget'], selected, event,
                                        previous, lead=4)
            model.step(sensory)
            predictions = dict(
                direct=heads['direct'].step(model.observed, event),
                recurrent=heads['recurrent'].step(model.state, event),
                shuffled_recurrent=heads['shuffled_recurrent'].step(
                    model.state, event))
            observed_count = int((model.observed.max(0).values >= .5).sum())
            state_count = int((model.state.max(0).values >= .5).sum())
            pending.append((t, predictions, event,
                            dict(direct=observed_count, recurrent=state_count,
                                 shuffled_recurrent=state_count)))
    for row in result.values():
        row['mean_mass'] = row['mass']/max(row['frames'], 1)
        row['mean_active_mass'] = row['active_mass']/max(row['active_frames'], 1)
        row['mean_quiet_mass'] = row['quiet_mass']/max(row['quiet_frames'], 1)
        row['mean_source_sites'] = row['source_sites']/max(row['frames'], 1)
        row['top_8_recall'] = row['top_8_hits']/max(row['rank_targets'], 1)
        row['threshold'] = finish(row['threshold'])
        row['budget'] = finish(row['budget'])
    return dict(arms=result,
                baselines={name: finish(row) for name, row in baselines.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    encoder = train_encoder(training)
    dictionary_seconds = time.perf_counter()-started
    print(f'sequence dictionary trained in {dictionary_seconds:.1f}s',
          flush=True)
    model = train_state(encoder, training)
    heads = train_heads(model, training)
    training_seconds = time.perf_counter()-started
    print(f'local sequence state and heads trained in '
          f'{training_seconds:.1f}s total', flush=True)
    calibration = probe(model, heads, training)['arms']
    gains = {name: row['targets']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: probe(model, heads, cases, gains=gains)
              for name, cases in groups.items()}
    result = dict(training_episodes=len(training),
                  dictionary_updates=encoder.dictionary_updates,
                  dictionary_seconds=dictionary_seconds,
                  training_seconds=training_seconds,
                  gains=gains, scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({group: {name: dict(
        budget=round(row['budget']['f1'], 3),
        top_8=round(row['top_8_recall'], 3),
        quiet_alarms=round(row['budget']['quiet_alarms_per_frame'], 3),
        source_sites=round(row['mean_source_sites'], 2))
        for name, row in scores[group]['arms'].items()}
        for group in scores}), flush=True)


if __name__ == '__main__':
    main()
