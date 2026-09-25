"""Frozen full-scene check of fixed sensory and learned history forecasts."""

import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT, timing_cases
from generic_local_transition import LocalTripletLearner, accumulate, empty_score, finish


OUT = Path('docs/experiments/2026-09-25-sensory-history-complement-results.json')
NAMES = ('fixed_raw', 'fixed_plus_learned', 'fixed_plus_random',
         'fixed_plus_unsplit', 'fixed_plus_reset',
         'archived', 'archived_plus_learned')


def evaluation_cases():
    cases = [('single', events, None, None)
             for _, _, _, _, events in unseen_shape_sequences()]
    cases += [(family, events, 10 if family == 'occlusion' else None,
               (7, 8, 9) if family == 'occlusion' else None)
              for family, _, events in make_robust_cases()]
    cases += [(window, events, hidden[-1]+1, hidden)
              for window, _, _, _, _, hidden, events in timing_cases()
              if window in ('early_2', 'late_4')]
    return cases


@torch.no_grad()
def evaluate(archived, models, cases):
    scores = {family: {name: empty_score() for name in NAMES}
              for family, _, _, _ in cases}
    exact = {family: {name: empty_score() for name in NAMES}
             for family, _, at, _ in cases if at is not None}
    quiet = {family: {name: dict(frames=0, false_alarm_pixels=0)
                      for name in NAMES}
             for family, _, _, _ in cases}
    gap_quiet = {family: {name: dict(frames=0, false_alarm_pixels=0)
                          for name in NAMES}
                 for family, _, _, hidden in cases if hidden is not None}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, events, at, hidden in cases:
        codes = correlation_sequence(events)
        archived.reset_state()
        fixed.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(15):
            if t == at:
                models['trace_reset'].code.trace.zero_()
            baseline = primitive_to_events(archived.step(codes[t]))
            raw = fixed.step(events[t], learn=False)
            local = {name: model.step(events[t], codes[t])
                     for name, model in models.items()}
            predictions = dict(
                fixed_raw=raw,
                fixed_plus_learned=torch.maximum(raw, local['learned_split']),
                fixed_plus_random=torch.maximum(raw, local['random_split']),
                fixed_plus_unsplit=torch.maximum(raw, local['unsplit']),
                fixed_plus_reset=torch.maximum(raw, local['trace_reset']),
                archived=baseline,
                archived_plus_learned=torch.maximum(
                    baseline, local['learned_split']))
            if 3 <= t < 14:
                target = events[t+1]
                is_quiet = target.sum() == 0
                inside_gap = (hidden is not None
                              and hidden[0] <= t < hidden[-1])
                for name, prediction in predictions.items():
                    accumulate(scores[family][name], prediction, target)
                    if t == at:
                        accumulate(exact[family][name], prediction, target)
                    if is_quiet:
                        quiet[family][name]['frames'] += 1
                        quiet[family][name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
                    if inside_gap:
                        assert is_quiet
                        gap_quiet[family][name]['frames'] += 1
                        gap_quiet[family][name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(scores={family: {name: finish(row)
                                for name, row in rows.items()}
                        for family, rows in scores.items()},
                exact={family: {name: finish(row)
                                for name, row in rows.items()}
                       for family, rows in exact.items()},
                quiet=quiet, gap_quiet=gap_quiet)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    cases = evaluation_cases()
    result = evaluate(saved['archived'], saved['models'], cases)
    result['cases'] = len(cases)
    result['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        full_f1={family: {name: round(row['f1'], 3)
                         for name, row in rows.items()}
                 for family, rows in result['scores'].items()},
        exact_f1={family: {name: round(row['f1'], 3)
                          for name, row in rows.items()}
                  for family, rows in result['exact'].items()},
        quiet=result['quiet'], gap_quiet=result['gap_quiet'],
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
