"""Matched four-frame local forecasts from current and sparse retained state."""

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
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import (empty_score, finish,
                                              score_event, sensory_sequence)
from run_pong_camera_transfer import SEEDS, pong_events
from sparse_coherent_state import SparseCoherentState


OUT = Path('docs/experiments/2026-09-25-sparse-coherence-forecast-results.json')
ARMS = ('current', 'continuity', 'coherent', 'shuffled_coherent')


def make_probes():
    return dict(continuity=SparseCoherentState(decay=.9, radius=0),
                coherent=SparseCoherentState(decay=.9, radius=1))


def advance(model, probes, code, surface):
    model.step((code, surface))
    states = {'current': model.state}
    states.update({name: probe.step(model.state)
                   for name, probe in probes.items()})
    states['shuffled_coherent'] = states['coherent']
    return states


@torch.no_grad()
def train_heads(model, cases):
    heads = {name: CompetitiveLocalForecast(model.encoder.units,
                                            horizon=4) for name in ARMS}
    order = list(range(len(cases)))
    random.Random(11).shuffle(order)
    for n, index in enumerate(order):
        sequence = sensory_sequence(cases[index])
        events = [row[0] for row in sequence]
        indices = list(range(len(events)))
        random.Random(100+n).shuffle(indices)
        shuffled = [events[i] for i in indices]
        model.reset_state()
        probes = make_probes()
        for head in heads.values():
            head.reset_state()
        for t, (event, code, surface) in enumerate(sequence):
            states = advance(model, probes, code, surface)
            for name, head in heads.items():
                head.step(states[name], event, learn=True,
                          credit_target=shuffled[t] if name ==
                          'shuffled_coherent' else None)
    return heads


def empty():
    return dict(frames=0, targets=0, mass=0., source_sites=0,
                active_frames=0, active_mass=0., quiet_frames=0,
                quiet_mass=0., rank_targets=0, top_8_hits=0,
                threshold=empty_score(), budget=empty_score())


@torch.no_grad()
def evaluate(model, heads, cases, *, gains=None):
    rows = {name: empty() for name in ARMS}
    baselines = {name: empty_score() for name in ('repeat_event', 'frozen')}
    for events in cases:
        model.reset_state()
        probes = make_probes()
        for head in heads.values():
            head.reset_state()
        pending = deque()
        zero = torch.zeros_like(events[0])
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, predictions, previous, site_counts = pending.popleft()
                if origin >= 4:
                    score_event(baselines['repeat_event'], previous,
                                event, previous, lead=4)
                    score_event(baselines['frozen'], zero,
                                event, previous, lead=4)
                    target = (event >= .5).flatten()
                    for name, prediction in predictions.items():
                        row = rows[name]
                        mass = float(prediction.sum())
                        row['frames'] += 1
                        row['targets'] += int(target.sum())
                        row['mass'] += mass
                        row['source_sites'] += site_counts[name]
                        if target.any():
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
                            score_event(row['budget'], select_by_mass(
                                prediction, gain=gains[name]), event,
                                previous, lead=4)
            states = advance(model, probes, code, surface)
            predictions = {name: head.step(states[name], event)
                           for name, head in heads.items()}
            site_counts = {name: int((state.max(0).values >= .5).sum())
                           for name, state in states.items()}
            pending.append((t, predictions, event, site_counts))
    for row in rows.values():
        row['mean_mass'] = row['mass']/max(row['frames'], 1)
        row['mean_active_mass'] = row['active_mass']/max(
            row['active_frames'], 1)
        row['mean_quiet_mass'] = row['quiet_mass']/max(row['quiet_frames'], 1)
        row['mean_source_sites'] = row['source_sites']/max(row['frames'], 1)
        row['top_8_recall'] = row['top_8_hits']/max(row['rank_targets'], 1)
        row['threshold'] = finish(row['threshold'])
        row['budget'] = finish(row['budget'])
    return dict(arms=rows, baselines={name: finish(score)
                                     for name, score in baselines.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    heads = train_heads(model, training)
    trained_seconds = time.perf_counter()-started
    print(f'local state and heads trained in {trained_seconds:.1f}s',
          flush=True)
    calibration = evaluate(model, heads, training)['arms']
    gains = {name: row['targets']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: evaluate(model, heads, cases, gains=gains)
              for name, cases in groups.items()}
    result = dict(training_episodes=len(training),
                  training_seconds=trained_seconds, gains=gains,
                  scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({group: {name: dict(
        f1=round(row['budget']['f1'], 3),
        top_8=round(row['top_8_recall'], 3),
        quiet_alarms=round(
            row['budget']['quiet_alarms_per_frame'], 3),
        sites=round(row['mean_source_sites'], 2))
        for name, row in result['arms'].items()}
        for group, result in scores.items()}), flush=True)


if __name__ == '__main__':
    main()
