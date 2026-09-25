"""Threshold-free audit of the online continuous-trace event learner."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence
from gap_timing_transfer import MODEL_OUT, WINDOWS, timing_cases
from generic_local_transition import accumulate, empty_score, finish
from history_gated_prediction import expanded_interruption_cases
from online_blank_trace import train, training_cases


OUT = Path('docs/experiments/2026-09-25-online-trace-ranking-results.json')
TRAINED_OUT = Path('runs/2026-09-25-online-blank-trace-model.pt')


def ranking(positive, quiet):
    positive = np.asarray(positive)
    quiet = np.asarray(quiet)
    ranks = rankdata(np.concatenate((positive, quiet)))[:len(positive)]
    auc = (ranks.sum()-len(positive)*(len(positive)+1)/2)
    auc /= len(positive)*len(quiet)
    return dict(event_count=len(positive), quiet_candidates=len(quiet),
                event_mean=float(positive.mean()),
                event_median=float(np.median(positive)),
                quiet_mean=float(quiet.mean()),
                quiet_p99=float(np.quantile(quiet, .99)),
                event_vs_quiet_auc=float(auc))


def heldout_cases():
    cases = [('familiar_3', events, (7, 8, 9))
             for _, _, events in expanded_interruption_cases()]
    cases += [(window, events, hidden)
              for window, _, _, _, _, hidden, events in timing_cases()
              if window != 'familiar_3']
    return cases


@torch.no_grad()
def evaluate(model, cases):
    values = {group: dict(events=[], quiet=[], score=empty_score(),
                          quiet_frames=0, quiet_false_alarms=0)
              for group, _, _ in cases}
    for group, events, hidden in cases:
        correlations = correlation_sequence(events)
        last_blank = hidden[-1]
        model.reset_state()
        row = values[group]
        for t in range(last_blank+1):
            prediction = model.step(events[t], correlations[t])
            if t == last_blank-1:
                assert events[t+1].sum() == 0
                active = (model.code.trace.sum(0) > 0).float()[None, None]
                reachable = F.max_pool2d(active, 17, stride=1,
                                         padding=8)[0, 0] > 0
                row['quiet'].extend(prediction[:, reachable].flatten().tolist())
                row['quiet_frames'] += 1
                row['quiet_false_alarms'] += int((prediction >= .5).sum())
            if t == last_blank:
                target = events[t+1]
                row['events'].extend(prediction[target > 0].tolist())
                accumulate(row['score'], prediction, target)
    return {group: dict(**ranking(row['events'], row['quiet']),
                        gap_exit=finish(row['score']),
                        quiet_frames=row['quiet_frames'],
                        quiet_false_alarms=row['quiet_false_alarms'])
            for group, row in values.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    model = train(saved['models']['learned_split'].code, training_cases())
    TRAINED_OUT.parent.mkdir(exist_ok=True)
    torch.save(model, TRAINED_OUT)
    train_elapsed = time.perf_counter()-started
    print(f'online trace checkpoint saved after {train_elapsed:.1f}s',
          flush=True)
    training = [(family, events, WINDOWS[family])
                for family, events in training_cases()
                if family in WINDOWS]
    result = dict(training=evaluate(model, training),
                  heldout=evaluate(model, heldout_cases()),
                  training_seconds=train_elapsed,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training={group: dict(
            auc=round(row['event_vs_quiet_auc'], 3),
            f1=round(row['gap_exit']['f1'], 3),
            event_mean=round(row['event_mean'], 3),
            quiet_mean=round(row['quiet_mean'], 3))
            for group, row in result['training'].items()},
        heldout={group: dict(
            auc=round(row['event_vs_quiet_auc'], 3),
            f1=round(row['gap_exit']['f1'], 3),
            event_mean=round(row['event_mean'], 3),
            quiet_mean=round(row['quiet_mean'], 3))
            for group, row in result['heldout'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
