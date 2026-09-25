"""Evaluation-only local divisive normalization of frozen blank-frame trace."""

import json
import time
from pathlib import Path

import torch

from blank_trace_capacity import (fit, local_matrix, score,
                                  target_coverage, trace_at_exit)
from context_trace_capacity import readout_training_cases
from gap_timing_transfer import MODEL_OUT, timing_cases
from history_gated_prediction import expanded_interruption_cases


OUT = Path('docs/experiments/2026-09-25-normalized-blank-trace-results.json')
REFERENCE = Path('docs/experiments/2026-09-25-blank-trace-capacity-results.json')


def normalize_trace_sites(trace):
    total = trace.sum(0, keepdim=True)
    return torch.where(total > 0, trace/total.clamp(min=1e-8),
                       torch.zeros_like(trace))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    code = saved['models']['learned_split'].code
    groups = dict(
        training=[('train', events, (7, 8, 9))
                  for _, events in readout_training_cases()],
        familiar=[(family, events, (7, 8, 9))
                  for family, _, events in expanded_interruption_cases()],
        changed=[(f'{window}:{direction}', events, hidden)
                 for window, _, direction, _, _, hidden, events in timing_cases()
                 if window != 'familiar_3'])
    encoded = {}
    for name, cases in groups.items():
        raw = [trace_at_exit(code, events, hidden)
               for _, events, hidden in cases]
        traces = [normalize_trace_sites(trace) for trace in raw]
        targets = [events[hidden[-1]+1] for _, events, hidden in cases]
        labels = [label for label, _, _ in cases]
        encoded[name] = dict(traces=traces, targets=targets,
                             labels=labels,
                             coverage=target_coverage(traces, targets))
    training_matrix = local_matrix(encoded['training']['traces'])
    weights = fit(training_matrix, encoded['training']['targets'])
    results = {}
    for name, row in encoded.items():
        matrix = (training_matrix if name == 'training'
                  else local_matrix(row['traces']))
        results[name] = dict(
            cases=len(row['targets']), coverage=row['coverage'],
            decoder=score(matrix, weights, row['targets'], row['labels']))
    reference = json.loads(REFERENCE.read_text())
    result = dict(normalized=results,
                  raw_reference={name: reference[name]['decoder']
                                 for name in results},
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        normalized={name: {label: round(value['f1'], 3)
                           for label, value in row['decoder']['by_label'].items()}
                    for name, row in results.items()},
        overall={name: round(row['decoder']['overall']['f1'], 3)
                 for name, row in results.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
