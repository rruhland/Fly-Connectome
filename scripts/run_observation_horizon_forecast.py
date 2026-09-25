"""Opt-in delayed local forecasting from persistent event evidence and motion."""

import copy
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import torch

from delayed_visual_emission import (DelayedLocalEmission,
                                     RecentObservationSurface,
                                     SurfaceMotionEncoder)
from diverse_visual_experience import unseen_shape_sequences
from correlation_latent_robustness import make_robust_cases
from gap_timing_transfer import MODEL_OUT
from local_hidden_transition import LocalHiddenTransition
from local_observation_surface import LocalObservationSurface
from run_native_trace_support import trace_correlation_sequence
from run_pong_camera_transfer import SEEDS, pong_events
from run_variable_cadence_transition import training_cases


OUT = Path('docs/experiments/2026-09-25-observation-horizon-results.json')
ARMS = ('learned', 'shuffled_credit', 'repeat_event', 'frozen')


def empty_score():
    return dict(tp=0, fp=0, fn=0, frames=0, active_frames=0,
                quiet_frames=0, quiet_alarm_pixels=0,
                novel_targets=0, novel_recalled=0, lead_sum=0)


def score_event(row, prediction, target, origin_event, *, lead=0):
    predicted = prediction >= .5
    actual = target >= .5
    novel = actual & ~(origin_event >= .5)
    row['tp'] += int((predicted & actual).sum())
    row['fp'] += int((predicted & ~actual).sum())
    row['fn'] += int((~predicted & actual).sum())
    row['novel_targets'] += int(novel.sum())
    row['novel_recalled'] += int((predicted & novel).sum())
    row['frames'] += 1
    row['lead_sum'] += lead
    if actual.any():
        row['active_frames'] += 1
    else:
        row['quiet_frames'] += 1
        row['quiet_alarm_pixels'] += int(predicted.sum())


def finish(row):
    tp, fp, fn = (row[key] for key in ('tp', 'fp', 'fn'))
    return {**row, 'f1': 2*tp/max(2*tp+fp+fn, 1),
            'precision': tp/max(tp+fp, 1),
            'recall': tp/max(tp+fn, 1),
            'novel_recall': row['novel_recalled']/max(row['novel_targets'], 1),
            'quiet_alarms_per_frame': row['quiet_alarm_pixels']/max(
                row['quiet_frames'], 1),
            'mean_lead_frames': row['lead_sum']/max(row['frames'], 1)}


def sensory_sequence(events, *, recent=False):
    surface = (RecentObservationSurface(decay=math.exp(-1/4)) if recent
               else LocalObservationSurface())
    codes = trace_correlation_sequence(events)
    return [(event, code, surface.step(event).clone())
            for event, code in zip(events, codes)]


def make_heads(units):
    return {
        mode: {name: DelayedLocalEmission(
            units, horizon=4 if mode == 'four_frame' else None,
            next_event=mode == 'next_event')
               for name in ('learned', 'shuffled_credit')}
        for mode in ('four_frame', 'next_event')}


@torch.no_grad()
def train_recurrent(motion, cases, *, recent=False):
    model = LocalHiddenTransition(SurfaceMotionEncoder(
        copy.deepcopy(motion), observation_channels=4 if recent else 2))
    order = list(range(len(cases)))
    random.Random(3).shuffle(order)
    for index in order:
        model.reset_state()
        for _, code, surface in sensory_sequence(cases[index], recent=recent):
            model.step((code, surface), learn=True)
    return model


@torch.no_grad()
def train_heads(recurrent, cases, *, recent=False):
    heads = make_heads(recurrent.encoder.units)
    order = list(range(len(cases)))
    random.Random(11).shuffle(order)
    for n, index in enumerate(order):
        sequence = sensory_sequence(cases[index], recent=recent)
        events = [row[0] for row in sequence]
        rng = random.Random(100+n)
        order_all = list(range(len(events)))
        rng.shuffle(order_all)
        shuffled_all = [events[i] for i in order_all]
        active = [i for i, event in enumerate(events) if event.any()]
        shuffled_active = active.copy()
        rng.shuffle(shuffled_active)
        event_credit = events.copy()
        for target, source in zip(active, shuffled_active):
            event_credit[target] = events[source]
        recurrent.reset_state()
        for group in heads.values():
            for head in group.values():
                head.reset_state()
        for t, (event, code, surface) in enumerate(sequence):
            recurrent.step((code, surface))
            state = recurrent.state
            for mode, group in heads.items():
                for name, head in group.items():
                    control = (shuffled_all[t] if mode == 'four_frame'
                               else event_credit[t])
                    head.step(state, event, learn=True,
                              credit_target=control if name ==
                              'shuffled_credit' else None)
    return heads


@torch.no_grad()
def evaluate_episode(recurrent, heads, events, scores, *, recent=False):
    recurrent.reset_state()
    for group in heads.values():
        for head in group.values():
            head.reset_state()
    pending_frames = deque()
    pending_event = None
    zero = torch.zeros_like(events[0])
    for t, (event, code, surface) in enumerate(sensory_sequence(
            events, recent=recent)):
        if len(pending_frames) == 4:
            origin, forecasts, old_event = pending_frames.popleft()
            if origin >= 4:
                for name, forecast in forecasts.items():
                    score_event(scores['four_frame'][name], forecast, event,
                                old_event, lead=4)
        if event.any() and pending_event is not None:
            origin, forecasts, old_event = pending_event
            if origin >= 4:
                for name, forecast in forecasts.items():
                    score_event(scores['next_event'][name], forecast, event,
                                old_event, lead=t-origin)
        recurrent.step((code, surface))
        state = recurrent.state
        frame = {name: head.step(state, event)
                 for name, head in heads['four_frame'].items()}
        frame.update(repeat_event=event, frozen=zero)
        pending_frames.append((t, frame, event))
        issued = {name: head.step(state, event)
                  for name, head in heads['next_event'].items()}
        if event.any():
            issued.update(repeat_event=event, frozen=zero)
            pending_event = (t, issued, event)


@torch.no_grad()
def evaluate(recurrent, heads, cases, *, recent=False):
    scores = {mode: {name: empty_score() for name in ARMS}
              for mode in ('four_frame', 'next_event')}
    for events in cases:
        evaluate_episode(recurrent, heads, events, scores, recent=recent)
    return {mode: {name: finish(row) for name, row in group.items()}
            for mode, group in scores.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    motion = saved['models']['learned_split'].code.motion
    training = training_cases()
    recurrent = train_recurrent(motion, training)
    recurrent_seconds = time.perf_counter()-started
    print(f'local recurrent state trained on {len(training)} episodes '
          f'in {recurrent_seconds:.1f}s', flush=True)
    heads = train_heads(recurrent, training)
    training_seconds = time.perf_counter()-started
    print(f'local delayed heads trained in {training_seconds:.1f}s total',
          flush=True)
    generic = {'unseen_single': [events for *_, events in
                                  unseen_shape_sequences()]}
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            generic.setdefault(family, []).append(events)
    native = [pong_events(seed, stride=1, frames=120) for seed in SEEDS]
    results = dict(training_episodes=len(training),
                   generic={name: evaluate(recurrent, heads, cases)
                            for name, cases in generic.items()},
                   native=evaluate(recurrent, heads, native),
                   recurrent_weight_sum=float(recurrent.weights.sum()),
                   head_weight_sums={mode: {name: float(head.weights.sum())
                                           for name, head in group.items()}
                                     for mode, group in heads.items()},
                   recurrent_seconds=recurrent_seconds,
                   training_seconds=training_seconds,
                   elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({family: {mode: {name: round(row['f1'], 3)
                                      for name, row in arms.items()}
                                for mode, arms in result.items()}
                      for family, result in (*results['generic'].items(),
                                             ('native', results['native']))}),
          flush=True)


if __name__ == '__main__':
    main()
