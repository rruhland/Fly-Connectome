"""Evaluation-only spatial capacity of frozen motion history on blank frames."""

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
import torch
import torch.nn.functional as F

from context_trace_capacity import readout_training_cases
from correlation_input_latent import correlation_sequence
from gap_timing_transfer import MODEL_OUT, timing_cases
from generic_local_transition import accumulate, empty_score, finish
from history_gated_prediction import expanded_interruption_cases


OUT = Path('docs/experiments/2026-09-25-blank-trace-capacity-results.json')
HEIGHT, WIDTH = 32, 64
CHANNELS, RADIUS = 24, 8


@torch.no_grad()
def trace_at_exit(code, events, hidden):
    code.reset_state()
    correlations = correlation_sequence(events)
    for t in range(hidden[-1]+1):
        code.step(events[t], correlations[t])
    return code.trace.clone()


def local_matrix(traces, *, radius=RADIUS):
    """Shared local trace features, one row per output pixel and episode."""
    field = 2*radius+1
    dy, dx = np.meshgrid(np.arange(-radius, radius+1),
                         np.arange(-radius, radius+1), indexing='ij')
    dy, dx = dy.ravel(), dx.ravel()
    offsets = np.arange(field*field)
    rows, columns, values = [], [], []
    for episode, trace in enumerate(traces):
        for unit, y, x in torch.nonzero(trace > 0).tolist():
            yy, xx = y+dy, x+dx
            valid = (0 <= yy) & (yy < HEIGHT) & (0 <= xx) & (xx < WIDTH)
            rows.append(episode*HEIGHT*WIDTH+yy[valid]*WIDTH+xx[valid])
            columns.append(unit*field*field+offsets[valid])
            values.append(np.full(int(valid.sum()), float(trace[unit, y, x])))
    if not rows:
        return sparse.csr_matrix((len(traces)*HEIGHT*WIDTH,
                                  CHANNELS*field*field))
    return sparse.csr_matrix((np.concatenate(values),
                              (np.concatenate(rows),
                               np.concatenate(columns))),
                             shape=(len(traces)*HEIGHT*WIDTH,
                                    CHANNELS*field*field))


def target_coverage(traces, targets):
    result = {radius: dict(target_events=0, reachable_events=0)
              for radius in (2, 4, 8)}
    for trace, target in zip(traces, targets):
        active = (trace.sum(0) > 0).float()[None, None]
        for radius, row in result.items():
            reachable = F.max_pool2d(active, 2*radius+1,
                                     stride=1, padding=radius)[0, 0] > 0
            row['target_events'] += int(target.sum())
            row['reachable_events'] += int(target[:, reachable].sum())
    return {radius: dict(**row,
                         fraction=row['reachable_events']/row['target_events'])
            for radius, row in result.items()}


def fit(matrix, targets):
    labels = np.stack([target.reshape(2, -1).numpy().T
                       for target in targets]).reshape(-1, 2)
    weights = np.stack([lsqr(matrix, labels[:, polarity], damp=1.,
                             atol=1e-5, btol=1e-5, iter_lim=200)[0]
                        for polarity in range(2)], axis=1)
    return weights


def score(matrix, weights, targets, labels):
    predictions = np.clip(matrix @ weights, 0, 1).reshape(
        len(targets), HEIGHT, WIDTH, 2)
    overall = empty_score()
    by_label = {label: empty_score() for label in labels}
    for prediction, target, label in zip(predictions, targets, labels):
        forecast = torch.from_numpy(prediction).permute(2, 0, 1)
        accumulate(overall, forecast, target)
        accumulate(by_label[label], forecast, target)
    return dict(overall=finish(overall),
                by_label={label: finish(row)
                          for label, row in by_label.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    code = saved['models']['learned_split'].code
    training = [('train', events, (7, 8, 9))
                for _, events in readout_training_cases()]
    familiar = [(family, events, (7, 8, 9))
                for family, _, events in expanded_interruption_cases()]
    changed = [(f'{window}:{direction}', events, hidden)
               for window, _, direction, _, _, hidden, events in timing_cases()
               if window != 'familiar_3']
    groups = dict(training=training, familiar=familiar, changed=changed)
    encoded = {}
    for name, cases in groups.items():
        traces = [trace_at_exit(code, events, hidden)
                  for _, events, hidden in cases]
        targets = [events[hidden[-1]+1] for _, events, hidden in cases]
        encoded[name] = dict(traces=traces, targets=targets,
                             labels=[label for label, _, _ in cases],
                             coverage=target_coverage(traces, targets))
    train_matrix = local_matrix(encoded['training']['traces'])
    weights = fit(train_matrix, encoded['training']['targets'])
    results = {}
    for name, row in encoded.items():
        matrix = (train_matrix if name == 'training'
                  else local_matrix(row['traces']))
        results[name] = dict(cases=len(row['targets']),
                             matrix_nonzeros=matrix.nnz,
                             coverage=row['coverage'],
                             decoder=score(matrix, weights,
                                           row['targets'], row['labels']),
                             zero_trace=score(sparse.csr_matrix(matrix.shape),
                                              weights, row['targets'],
                                              row['labels']))
    result = dict(**results, trace_units=CHANNELS,
                  local_field=2*RADIUS+1,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        f1={name: {label: round(score['f1'], 3)
                   for label, score in row['decoder']['by_label'].items()}
            for name, row in results.items()},
        overall={name: round(row['decoder']['overall']['f1'], 3)
                 for name, row in results.items()},
        coverage={name: {radius: round(value['fraction'], 3)
                         for radius, value in row['coverage'].items()}
                  for name, row in results.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
