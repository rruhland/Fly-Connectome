"""Opt-in event/quiet-separated local credit for continuous trace output."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT
from generic_local_transition import LocalTripletLearner, accumulate, empty_score, finish
from online_blank_trace import (ContinuousTraceReadout, scatter_trace,
                                trace_sources, training_cases)
from online_trace_ranking import evaluate as evaluate_ranking
from online_trace_ranking import heldout_cases


OUT = Path('docs/experiments/2026-09-25-balanced-trace-credit-results.json')
SHARED_REFERENCE = Path('docs/experiments/2026-09-25-online-blank-trace-results.json')
RANKING_REFERENCE = Path('docs/experiments/2026-09-25-online-trace-ranking-results.json')
NAMES = ('fixed_raw', 'fixed_plus_balanced', 'fixed_plus_shuffled')


def update_balanced(weights, sources, target, prediction, eta):
    if not sources:
        return
    padded_target = F.pad(target, (8,)*4)
    padded_prediction = F.pad(prediction, (8,)*4)
    denominator = max(sum(amplitude for _, _, _, amplitude in sources), 1.)
    for unit, y, x, amplitude in sources:
        observed = padded_target[:, y:y+17, x:x+17]
        forecast = padded_prediction[:, y:y+17, x:x+17]
        confirming = observed*(1-forecast)
        suppressing = (1-observed)*forecast
        event_count = observed.sum((1, 2)).clamp(min=1)[:, None, None]
        predicted_count = (forecast > 0).sum((1, 2)).clamp(min=1)[
            :, None, None]
        weights[:, unit] += (eta*amplitude/denominator
                             *(confirming/event_count
                               -suppressing/predicted_count))
    weights.clamp_(0, 1)


class BalancedTraceReadout(ContinuousTraceReadout):
    @torch.no_grad()
    def step(self, events, coincidence, *, learn=False, credit_target=None):
        if learn and self.pending_prediction is not None:
            update_balanced(self.weights, self.previous_sources,
                            events if credit_target is None else credit_target,
                            self.pending_prediction, self.eta)
        self.code.step(events, coincidence)
        self.previous_sources = trace_sources(self.code.trace)
        self.pending_prediction = scatter_trace(self.weights,
                                                self.previous_sources)
        return self.pending_prediction


@torch.no_grad()
def train_pair(code, cases):
    learned = BalancedTraceReadout(copy.deepcopy(code))
    shuffled = BalancedTraceReadout(copy.deepcopy(code))
    encoded = [(events, correlation_sequence(events))
               for _, events in cases]
    random.Random(3).shuffle(encoded)
    controls = encoded[1:]+encoded[:1]
    for (events, correlations), (other_events, _) in zip(encoded, controls):
        learned.reset_state()
        shuffled.reset_state()
        for event, correlation, unrelated in zip(events, correlations,
                                                 other_events):
            learned.step(event, correlation, learn=True)
            shuffled.step(event, correlation, learn=True,
                          credit_target=unrelated)
    return learned, shuffled


def full_scene_cases():
    cases = [('single', events)
             for _, _, _, _, events in unseen_shape_sequences()]
    cases += [(family, events)
              for family, _, events in make_robust_cases()]
    return cases


@torch.no_grad()
def evaluate_full(models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _ in cases}
    quiet = {family: {name: dict(frames=0, false_alarm_pixels=0)
                      for name in NAMES}
             for family, _ in cases}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, events in cases:
        correlations = correlation_sequence(events)
        fixed.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(15):
            raw = fixed.step(events[t], learn=False)
            learned = {name: model.step(events[t], correlations[t])
                       for name, model in models.items()}
            predictions = dict(
                fixed_raw=raw,
                fixed_plus_balanced=torch.maximum(raw, learned['balanced']),
                fixed_plus_shuffled=torch.maximum(raw, learned['shuffled']))
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    if target.sum() == 0:
                        quiet[family][name]['frames'] += 1
                        quiet[family][name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(groups={family: {name: finish(row)
                                for name, row in rows.items()}
                        for family, rows in groups.items()}, quiet=quiet)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    corpus = training_cases()
    balanced, shuffled = train_pair(saved['models']['learned_split'].code,
                                    corpus)
    trained = time.perf_counter()-started
    print(f'balanced and shuffled local readouts trained in '
          f'{trained:.1f}s', flush=True)
    training_gaps = [(family, events,
                      {'early_2': (6, 7), 'familiar_3': (7, 8, 9),
                       'late_4': (8, 9, 10, 11)}[family])
                     for family, events in corpus if family != 'clean']
    ranks = {name: dict(training=evaluate_ranking(model, training_gaps),
                        heldout=evaluate_ranking(model, heldout_cases()))
             for name, model in (('balanced', balanced),
                                 ('shuffled', shuffled))}
    full = evaluate_full(dict(balanced=balanced, shuffled=shuffled),
                         full_scene_cases())
    result = dict(training_episodes=len(corpus), training_seconds=trained,
                  ranking=ranks, full=full,
                  shared_reference=json.loads(SHARED_REFERENCE.read_text())[
                      'heldout']['full'],
                  ranking_reference=json.loads(RANKING_REFERENCE.read_text())[
                      'heldout'],
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        gap_exit={name: {group: dict(
            f1=round(row['gap_exit']['f1'], 3),
            auc=round(row['event_vs_quiet_auc'], 3))
            for group, row in scores['heldout'].items()}
            for name, scores in ranks.items()},
        full={family: {name: round(row['f1'], 3)
                       for name, row in rows.items()}
              for family, rows in full['groups'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
