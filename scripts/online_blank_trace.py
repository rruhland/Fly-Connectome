"""Opt-in online local event readout from continuously active motion trace."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, scene_sequence
from diverse_visual_experience import TRAIN_SHAPES, unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT, WINDOWS, timing_cases
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from history_gated_code import DIRECTIONS
from history_gated_prediction import expanded_interruption_cases
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-25-online-blank-trace-results.json')
RADIUS = 8
NAMES = ('archived', 'fixed_raw', 'trace', 'trace_reset',
         'fixed_plus_trace', 'zero_weight')


def trace_sources(trace):
    return [(int(unit), int(y), int(x), float(trace[unit, y, x]))
            for unit, y, x in torch.nonzero(trace > 0).tolist()]


def scatter_trace(weights, sources):
    canvas = torch.zeros((2, 32+2*RADIUS, 64+2*RADIUS))
    for unit, y, x, amplitude in sources:
        canvas[:, y:y+17, x:x+17] += amplitude*weights[:, unit]
    return canvas[:, RADIUS:-RADIUS, RADIUS:-RADIUS].clamp_(0, 1)


def update_trace_weights(weights, sources, error, eta):
    if not sources:
        return
    padded = F.pad(error, (RADIUS,)*4)
    denominator = max(sum(amplitude for _, _, _, amplitude in sources), 1.)
    for unit, y, x, amplitude in sources:
        weights[:, unit] += (eta*amplitude/denominator
                             *padded[:, y:y+17, x:x+17])
    weights.clamp_(0, 1)


class ContinuousTraceReadout:
    def __init__(self, code, *, eta=.5):
        self.code = code
        self.eta = eta
        self.weights = torch.zeros((2, code.motion.units, 17, 17))
        self.reset_state()

    def reset_state(self):
        self.code.reset_state()
        self.previous_sources = []
        self.pending_prediction = None

    @torch.no_grad()
    def step(self, events, coincidence, *, learn=False):
        if learn and self.pending_prediction is not None:
            update_trace_weights(self.weights, self.previous_sources,
                                 events-self.pending_prediction, self.eta)
        self.code.step(events, coincidence)
        self.previous_sources = trace_sources(self.code.trace)
        self.pending_prediction = scatter_trace(self.weights,
                                                self.previous_sources)
        return self.pending_prediction


def training_cases():
    cases = [(family, events) for family, _, events in make_training_cases()
             if family == 'clean']
    for window, hidden in WINDOWS.items():
        for shape in TRAIN_SHAPES:
            for dy, dx in DIRECTIONS.values():
                for speed in (1, 2):
                    for background in (True, False):
                        obj = dict(shape=shape, center=(16, 32),
                                   before=(dy*speed, dx*speed),
                                   after=(dy*speed, dx*speed),
                                   hidden=hidden)
                        cases.append((window, scene_sequence(
                            [obj], background=background)))
    return cases


@torch.no_grad()
def train(code, cases):
    model = ContinuousTraceReadout(copy.deepcopy(code))
    encoded = [(events, correlation_sequence(events))
               for _, events in cases]
    random.Random(3).shuffle(encoded)
    for events, correlations in encoded:
        model.reset_state()
        for event, correlation in zip(events, correlations):
            model.step(event, correlation, learn=True)
    return model


def evaluation_cases():
    cases = [('single', events, None)
             for _, _, _, _, events in unseen_shape_sequences()]
    cases += [(family, events, (7, 8, 9) if family == 'occlusion'
               else None)
              for family, _, events in make_robust_cases()]
    cases += [(f'{window}:{direction}', events, hidden)
              for window, _, direction, _, _, hidden, events in timing_cases()]
    cases += [(family, events, (7, 8, 9))
              for family, _, events in expanded_interruption_cases()]
    return cases


@torch.no_grad()
def evaluate(archived, trained, cases):
    reset = copy.deepcopy(trained)
    scores = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    exit_scores = {family: {name: empty_score() for name in NAMES}
                   for family, _, hidden in cases if hidden is not None}
    after_scores = {family: {name: empty_score() for name in NAMES}
                    for family, _, hidden in cases if hidden is not None}
    quiet = {family: {name: dict(frames=0, false_alarm_pixels=0)
                      for name in NAMES}
             for family, _, _ in cases}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, events, hidden in cases:
        correlations = correlation_sequence(events)
        archived.reset_state()
        trained.reset_state()
        if hidden is not None:
            reset.reset_state()
        fixed.reset_state()
        for t in range(15):
            if hidden is not None and t == hidden[-1]:
                reset.code.trace.zero_()
            baseline = primitive_to_events(archived.step(correlations[t]))
            raw = fixed.step(events[t], learn=False)
            local = trained.step(events[t], correlations[t])
            predictions = dict(
                archived=baseline, fixed_raw=raw, trace=local,
                trace_reset=(reset.step(events[t], correlations[t])
                             if hidden is not None else local),
                fixed_plus_trace=torch.maximum(raw, local),
                zero_weight=torch.zeros_like(local))
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(scores[family][name], prediction, target)
                    if hidden is not None and t == hidden[-1]:
                        accumulate(exit_scores[family][name],
                                   prediction, target)
                    if hidden is not None and t == hidden[-1]+1:
                        accumulate(after_scores[family][name],
                                   prediction, target)
                    if target.sum() == 0:
                        quiet[family][name]['frames'] += 1
                        quiet[family][name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(full={family: {name: finish(row)
                              for name, row in rows.items()}
                      for family, rows in scores.items()},
                gap_exit={family: {name: finish(row)
                                   for name, row in rows.items()}
                          for family, rows in exit_scores.items()},
                after_reappearance={family: {name: finish(row)
                                            for name, row in rows.items()}
                                    for family, rows in after_scores.items()},
                quiet=quiet)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    cases = training_cases()
    model = train(saved['models']['learned_split'].code, cases)
    training_elapsed = time.perf_counter()-started
    print(f'continuous trace trained on {len(cases)} episodes in '
          f'{training_elapsed:.1f}s', flush=True)
    training_scores = evaluate(saved['archived'], model,
                               [(family, events, WINDOWS[family])
                                for family, events in cases
                                if family in WINDOWS])
    heldout_scores = evaluate(saved['archived'], model,
                              evaluation_cases())
    result = dict(training_episodes=len(cases),
                  training=training_scores, heldout=heldout_scores,
                  training_seconds=training_elapsed,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        train_exit={family: round(rows['trace']['f1'], 3)
                    for family, rows in training_scores['gap_exit'].items()},
        heldout_exit={family: round(rows['trace']['f1'], 3)
                      for family, rows in heldout_scores['gap_exit'].items()},
        full_f1={family: {name: round(row['f1'], 3)
                          for name, row in rows.items()
                          if name in ('fixed_raw', 'trace',
                                      'fixed_plus_trace')}
                 for family, rows in heldout_scores['full'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
