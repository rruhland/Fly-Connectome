"""Matched four-frame local learning with bounded hidden-source memory."""

import json
import random
import time
from collections import deque
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from event_budget_probe import select_by_mass
from event_centric_forecast import CompetitiveLocalForecast
from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from lagged_local_memory import LagConditionalForecast, RecentLatentMemory
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import (empty_score, finish,
                                              score_event, sensory_sequence)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-lagged-local-memory-results.json')
ARMS = ('memory_4', 'memory_8', 'shuffled_8', 'current_state')


def make_models(units):
    memories = {4: RecentLatentMemory(units, window=4),
                8: RecentLatentMemory(units, window=8)}
    heads = dict(memory_4=LagConditionalForecast(units, window=4),
                 memory_8=LagConditionalForecast(units, window=8),
                 shuffled_8=LagConditionalForecast(units, window=8),
                 current_state=CompetitiveLocalForecast(units, horizon=4))
    return memories, heads


def issue(model, memories, heads, event, *, learn=False, shuffled=None):
    memories[4].step(model.state)
    memories[8].step(model.state)
    forecasts = {}
    for name, head in heads.items():
        if name == 'current_state':
            forecasts[name] = head.step(model.state, event, learn=learn)
        else:
            memory = memories[4 if name == 'memory_4' else 8]
            forecasts[name] = head.step(
                memory.activity, memory.age, event, learn=learn,
                credit_target=shuffled if name == 'shuffled_8' else None)
    return forecasts


@torch.no_grad()
def train_heads(model, cases):
    memories, heads = make_models(model.encoder.units)
    order = list(range(len(cases)))
    random.Random(11).shuffle(order)
    for n, index in enumerate(order):
        sequence = sensory_sequence(cases[index])
        events = [row[0] for row in sequence]
        indices = list(range(len(events)))
        random.Random(100+n).shuffle(indices)
        shuffled = [events[i] for i in indices]
        model.reset_state()
        for memory in memories.values():
            memory.reset_state()
        for head in heads.values():
            head.reset_state()
        for t, (event, code, surface) in enumerate(sequence):
            model.step((code, surface))
            issue(model, memories, heads, event, learn=True,
                  shuffled=shuffled[t])
    return memories, heads


def empty():
    return dict(frames=0, targets=0, mass=0.,
                active_frames=0, active_mass=0.,
                quiet_frames=0, quiet_mass=0.,
                rank_targets=0, top_8_hits=0,
                threshold=empty_score(), budget=empty_score())


@torch.no_grad()
def probe(model, memories, heads, cases, *, gains=None):
    result = {name: empty() for name in ARMS}
    baselines = {name: empty_score() for name in ('repeat_event', 'frozen')}
    for events in cases:
        model.reset_state()
        for memory in memories.values():
            memory.reset_state()
        for head in heads.values():
            head.reset_state()
        pending = deque()
        zero = torch.zeros_like(events[0])
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, forecasts, previous = pending.popleft()
                if origin >= 4:
                    target = (event >= .5).flatten()
                    score_event(baselines['repeat_event'], previous, event,
                                previous, lead=4)
                    score_event(baselines['frozen'], zero, event,
                                previous, lead=4)
                    for name, prediction in forecasts.items():
                        row = result[name]
                        mass = float(prediction.sum())
                        row['frames'] += 1
                        row['targets'] += int(event.sum())
                        row['mass'] += mass
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
            model.step((code, surface))
            forecasts = issue(model, memories, heads, event)
            pending.append((t, forecasts, event))
    for row in result.values():
        row['mean_mass'] = row['mass']/max(row['frames'], 1)
        row['mean_active_mass'] = row['active_mass']/max(row['active_frames'], 1)
        row['mean_quiet_mass'] = row['quiet_mass']/max(row['quiet_frames'], 1)
        row['top_8_recall'] = row['top_8_hits']/max(row['rank_targets'], 1)
        row['threshold'] = finish(row['threshold'])
        row['budget'] = finish(row['budget'])
    return dict(arms=result,
                baselines={name: finish(row) for name, row in baselines.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    memories, heads = train_heads(model, training)
    training_seconds = time.perf_counter()-started
    print(f'age-tagged local heads trained in {training_seconds:.1f}s',
          flush=True)
    calibration = probe(model, memories, heads, training)['arms']
    gains = {name: row['targets']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: probe(model, memories, heads, cases, gains=gains)
              for name, cases in groups.items()}
    result = dict(training_episodes=len(training),
                  training_seconds=training_seconds,
                  gains=gains, scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({group: {name: dict(
        threshold=round(row['threshold']['f1'], 3),
        budget=round(row['budget']['f1'], 3),
        quiet_alarms=round(row['budget']['quiet_alarms_per_frame'], 3),
        top_8=round(row['top_8_recall'], 3))
        for name, row in scores[group]['arms'].items()}
        for group in scores}), flush=True)


if __name__ == '__main__':
    main()
