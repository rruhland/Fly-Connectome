"""Evaluation-only capacity of one trace readout across visual phases."""

import json
import time
from pathlib import Path

from scipy import sparse
import torch

from blank_trace_capacity import fit, local_matrix, score, trace_at_exit
from correlation_input_latent import correlation_sequence
from gap_timing_transfer import MODEL_OUT, timing_cases
from history_gated_prediction import expanded_interruption_cases
from online_blank_trace import training_cases
from gap_timing_transfer import WINDOWS


OUT = Path('docs/experiments/2026-09-25-trace-phase-conflict-results.json')
PHASES = ('exit', 'quiet', 'visible', 'post')


def phase_indices(hidden):
    return dict(exit=hidden[-1], quiet=hidden[-1]-1,
                visible=hidden[0]-2, post=hidden[-1]+1)


@torch.no_grad()
def collect_training(code, cases):
    samples = {phase: dict(traces=[], targets=[], labels=[])
               for phase in PHASES}
    for window, events in cases:
        hidden = WINDOWS[window]
        at = phase_indices(hidden)
        phases_at = {frame: phase for phase, frame in at.items()}
        correlations = correlation_sequence(events)
        code.reset_state()
        for t in range(max(at.values())+1):
            code.step(events[t], correlations[t])
            if t in phases_at:
                row = samples[phases_at[t]]
                row['traces'].append(code.trace.clone())
                row['targets'].append(events[t+1])
                row['labels'].append(window)
    return samples


@torch.no_grad()
def collect_heldout(code):
    cases = [(family, events, (7, 8, 9))
             for family, _, events in expanded_interruption_cases()]
    cases += [(f'{window}:{direction}', events, hidden)
              for window, _, direction, _, _, hidden, events in timing_cases()
              if window != 'familiar_3']
    traces = [trace_at_exit(code, events, hidden)
              for _, events, hidden in cases]
    return dict(traces=traces,
                targets=[events[hidden[-1]+1]
                         for _, events, hidden in cases],
                labels=[family for family, _, _ in cases])


def summarize_phase(samples):
    return {phase: dict(cases=len(row['targets']),
                        target_events=sum(int(target.sum())
                                          for target in row['targets']))
            for phase, row in samples.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    code = saved['models']['learned_split'].code
    training = [(family, events) for family, events in training_cases()
                if family in WINDOWS]
    samples = collect_training(code, training)
    heldout = collect_heldout(code)
    matrices = {phase: local_matrix(row['traces'])
                for phase, row in samples.items()}
    heldout_matrix = local_matrix(heldout['traces'])
    arms = {}
    definitions = dict(exit_only=('exit',),
                       exit_quiet=('exit', 'quiet'),
                       all_phases=PHASES)
    for name, phases in definitions.items():
        if (name == 'all_phases'
                and arms['exit_quiet']['training_exit']['overall']['f1'] < .20):
            arms[name] = dict(skipped='exit + quiet training exit F1 below 0.20')
            break
        matrix = sparse.vstack([matrices[phase] for phase in phases],
                               format='csr')
        targets = [target for phase in phases
                   for target in samples[phase]['targets']]
        weights = fit(matrix, targets)
        arms[name] = dict(
            phases=phases, fitted_episodes=len(targets),
            matrix_nonzeros=matrix.nnz,
            training_exit=score(matrices['exit'], weights,
                                samples['exit']['targets'],
                                samples['exit']['labels']),
            heldout_exit=score(heldout_matrix, weights,
                               heldout['targets'], heldout['labels']),
            training_quiet=score(matrices['quiet'], weights,
                                 samples['quiet']['targets'],
                                 samples['quiet']['labels']))
        print(f'{name}: train exit '
              f'{arms[name]["training_exit"]["overall"]["f1"]:.3f}, '
              f'heldout exit '
              f'{arms[name]["heldout_exit"]["overall"]["f1"]:.3f}',
              flush=True)
    result = dict(training_samples=summarize_phase(samples),
                  heldout_cases=len(heldout['targets']), arms=arms,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        arms={name: (dict(train=round(row['training_exit']['overall']['f1'], 3),
                          heldout=round(row['heldout_exit']['overall']['f1'], 3),
                          quiet_fp=row['training_quiet']['overall']['fp'])
                     if 'skipped' not in row else row)
              for name, row in arms.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
