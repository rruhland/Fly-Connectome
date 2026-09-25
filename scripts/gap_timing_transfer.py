"""Frozen local visual forecast transfer across generic interruption windows."""

import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import scene_sequence
from diverse_visual_experience import TEST_SHAPES
from generic_local_transition import LocalTripletLearner, accumulate, empty_score, finish
from history_gated_code import DIRECTIONS
from polarity_local_prediction import NAMES, train_candidates
from balanced_interruption_exposure import make_balanced_cases


OUT = Path('docs/experiments/2026-09-25-gap-timing-transfer-results.json')
MODEL_OUT = Path('runs/2026-09-25-balanced-interruption-models.pt')
WINDOWS = {'early_2': (6, 7), 'familiar_3': (7, 8, 9),
           'late_4': (8, 9, 10, 11)}
ALL_NAMES = NAMES+('fixed_raw',)


def timing_cases():
    cases = []
    for window, hidden in WINDOWS.items():
        for shape in TEST_SHAPES:
            for direction, (dy, dx) in DIRECTIONS.items():
                for speed in (1, 2):
                    for background in (True, False):
                        obj = dict(shape=shape, center=(16, 32),
                                   before=(dy*speed, dx*speed),
                                   after=(dy*speed, dx*speed),
                                   hidden=hidden)
                        cases.append((window, shape, direction, speed,
                                      background, hidden,
                                      scene_sequence([obj],
                                                     background=background)))
    return cases


@torch.no_grad()
def evaluate_timing(archived, models, cases):
    scores = {window: {phase: {name: empty_score() for name in ALL_NAMES}
                       for phase in ('pre_gap', 'reappearance', 'followup')}
              for window in WINDOWS}
    quiet = {window: {name: dict(frames=0, false_alarm_pixels=0)
                      for name in ALL_NAMES}
             for window in WINDOWS}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for window, _, _, _, _, hidden, events in cases:
        reappearance = hidden[-1]+1
        phases = {hidden[0]-2: 'pre_gap', reappearance: 'reappearance',
                  reappearance+1: 'followup'}
        codes = correlation_sequence(events)
        archived.reset_state()
        fixed.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(reappearance+2):
            if t == reappearance:
                models['trace_reset'].code.trace.zero_()
            baseline = primitive_to_events(archived.step(codes[t]))
            predictions = {'archived': baseline,
                           'fixed_raw': fixed.step(events[t], learn=False)}
            for name, model in models.items():
                predictions[name] = torch.maximum(
                    baseline, model.step(events[t], codes[t]))
            if t in phases:
                for name, prediction in predictions.items():
                    accumulate(scores[window][phases[t]][name],
                               prediction, events[t+1])
            if hidden[0] <= t < hidden[-1]:
                assert events[t+1].sum() == 0
                for name, prediction in predictions.items():
                    quiet[window][name]['frames'] += 1
                    quiet[window][name]['false_alarm_pixels'] += int(
                        (prediction >= .5).sum())
    return dict(scores={window: {phase: {name: finish(row)
                                      for name, row in by_name.items()}
                                 for phase, by_name in by_phase.items()}
                        for window, by_phase in scores.items()},
                quiet=quiet)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original, added = make_balanced_cases()
    archived, models = train_candidates(original+added)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    torch.save(dict(archived=archived, models=models), MODEL_OUT)
    print(f'frozen balanced models saved after {time.perf_counter()-started:.1f}s',
          flush=True)
    cases = timing_cases()
    result = evaluate_timing(archived, models, cases)
    result.update(training_episodes=len(original)+len(added),
                  cases_per_window=len(cases)//len(WINDOWS),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        reappearance={window: {name: round(row['f1'], 3)
                               for name, row in phases['reappearance'].items()}
                      for window, phases in result['scores'].items()},
        quiet=result['quiet'],
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
