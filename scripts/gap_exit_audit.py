"""Audit local forecasts while interrupted motion is still invisible."""

import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from gap_timing_transfer import MODEL_OUT, timing_cases
from generic_local_transition import LocalTripletLearner, accumulate, empty_score, finish


OUT = Path('docs/experiments/2026-09-25-gap-exit-audit-results.json')
COMPLEMENT = Path('docs/experiments/2026-09-25-sensory-history-complement-results.json')
NAMES = ('archived', 'fixed_raw', 'learned_history', 'fixed_plus_history')
PHASES = ('gap_exit', 'post_reappearance', 'followup')


def oracle_extra_f1(fixed, fused):
    """Best union score when only learned-only false positives are removed."""
    if fused['tp'] < fixed['tp'] or fused['fn'] > fixed['fn']:
        raise ValueError('fused forecast must contain the fixed forecast')
    return 2*fused['tp']/(2*fused['tp']+fixed['fp']+fused['fn'])


@torch.no_grad()
def audit(archived, model, cases):
    scores = {window: {phase: {name: empty_score() for name in NAMES}
                       for phase in PHASES}
              for window, *_ in cases}
    state = {window: dict(cases=0, blank_frames=0,
                          trace_present=0, trace_sum=0., active_sources=0)
             for window, *_ in cases}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for window, _, _, _, _, hidden, events in cases:
        codes = correlation_sequence(events)
        last_blank = hidden[-1]
        reappearance = last_blank+1
        phase_at = {last_blank: 'gap_exit',
                    reappearance: 'post_reappearance',
                    reappearance+1: 'followup'}
        archived.reset_state()
        model.reset_state()
        fixed.reset_state()
        state[window]['cases'] += 1
        for t in range(reappearance+2):
            baseline = primitive_to_events(archived.step(codes[t]))
            raw = fixed.step(events[t], learn=False)
            history = model.step(events[t], codes[t])
            if t == last_blank:
                trace_sum = float(model.code.trace.sum())
                state[window]['blank_frames'] += int(events[t].sum() == 0)
                state[window]['trace_present'] += int(trace_sum > 0)
                state[window]['trace_sum'] += trace_sum
                state[window]['active_sources'] += len(model.previous_sources)
            if t in phase_at:
                predictions = dict(
                    archived=baseline, fixed_raw=raw,
                    learned_history=history,
                    fixed_plus_history=torch.maximum(raw, history))
                for name, prediction in predictions.items():
                    accumulate(scores[window][phase_at[t]][name],
                               prediction, events[t+1])
    return dict(scores={window: {phase: {name: finish(row)
                                      for name, row in by_name.items()}
                                 for phase, by_name in by_phase.items()}
                        for window, by_phase in scores.items()},
                state=state)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    result = audit(saved['archived'], saved['models']['learned_split'],
                   timing_cases())
    complement = json.loads(COMPLEMENT.read_text())
    result['oracle_extra_f1'] = {
        family: oracle_extra_f1(rows['fixed_raw'],
                                rows['fixed_plus_learned'])
        for family, rows in complement['scores'].items()}
    result['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        phase_f1={window: {phase: {name: round(row['f1'], 3)
                                  for name, row in rows.items()}
                           for phase, rows in phases.items()}
                  for window, phases in result['scores'].items()},
        state=result['state'],
        oracle={family: round(value, 3)
                for family, value in result['oracle_extra_f1'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
