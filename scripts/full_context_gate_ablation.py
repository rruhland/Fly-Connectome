"""Bounded always-open M1A gate ablation and frozen gated control replay."""
import json
from pathlib import Path

import torch

from full_context_pong import OpenLoopContextRun
from fly_connectome.data import checksum


SOURCE = Path('checkpoints/event-v1-combined-rate-initial.pt')
OUT = Path('runs/full-context-gate-ablation-v1')
FRAMES = 500
SOURCE_SHA = '8c4af80ff3904881bac9313b618dd9989bb51174b47606f28eca88865296005c'


def advance(name, payload, source_sha, seed, eta, *, components=None, always_open):
    path = OUT/f'{name}-{FRAMES}.pt'
    run = (OpenLoopContextRun.load(path, payload, source_sha) if path.exists()
           else OpenLoopContextRun(payload, seed, eta, components=components,
                                   always_open=always_open))
    if run.seed != seed or run.eta != eta or run.always_open != always_open:
        raise ValueError(f'{name}: checkpoint protocol mismatch')
    while run.frame < FRAMES:
        run.run(min(100, FRAMES-run.frame))
        run.save(path, source_sha)
        print('saved', name, 'frame', run.frame, 'fps',
              round(run.summary()['frames_per_second'], 2), flush=True)
    if run.frame != FRAMES:
        raise ValueError(f'{name}: checkpoint passed target frame count')
    return run, path


def delta(initial, trained):
    a, b = initial.summary(), trained.summary()
    changed = initial.spike_counts != trained.spike_counts
    return dict(changed_neuron_spike_counts=int(changed.sum()),
                total_spikes_initial=int(initial.spike_counts.sum()),
                total_spikes_trained=int(trained.spike_counts.sum()),
                absolute_neuron_spike_count_change=int(
                    (initial.spike_counts-trained.spike_counts).abs().sum()),
                on_anticipation_change=b['scores']['on']['anticipation']
                    - a['scores']['on']['anticipation'],
                off_anticipation_change=b['scores']['off']['anticipation']
                    - a['scores']['off']['anticipation'],
                on_mse_change=b['scores']['on']['mse']-a['scores']['on']['mse'],
                off_mse_change=b['scores']['off']['mse']-a['scores']['off']['mse'])


def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    if source_sha != SOURCE_SHA:
        raise ValueError('unexpected source checkpoint')
    payload = torch.load(SOURCE, weights_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    train, train_path = advance('always-open-train', payload, source_sha,
                                1101, 1., always_open=True)
    frozen = {}
    checkpoints = {}
    for name, components, always_open in (
            ('always-open-initial', None, True),
            ('always-open-trained', train.net.components.clone(), True),
            ('gated-initial-control', None, False),
            ('gated-trained-control', train.net.components.clone(), False)):
        # The gated trained control must use the original gated pilot weights.
        if name == 'gated-trained-control':
            original = torch.load('runs/full-context-m1a-v1/pilot-train-500.pt',
                                  weights_only=True)
            components = original['network']['components']
        run, path = advance(name, payload, source_sha, 1102, 0.,
                            components=components, always_open=always_open)
        frozen[name], checkpoints[name] = run, path
    original = {}
    for label in ('initial', 'trained'):
        state = torch.load(f'runs/full-context-m1a-v1/pilot-{label}-500.pt',
                           weights_only=True)
        control = frozen[f'gated-{label}-control']
        if control.events != state['events'] or control.metrics != state['metrics']:
            raise AssertionError(f'gated {label} control did not exactly replay')
        original[label] = dict(camera_events_and_scores_replayed=True,
                               original_checkpoint_sha256=checksum(
                                   Path(f'runs/full-context-m1a-v1/pilot-{label}-500.pt')))
    event_stream = frozen['always-open-initial'].events
    if any(run.events != event_stream for run in frozen.values()):
        raise AssertionError('frozen runs saw different camera targets')
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    results = {name: dict(checkpoint_sha256=checksum(checkpoints[name]),
                          **run.summary()) for name, run in frozen.items()}
    initial, trained = (frozen['always-open-initial'],
                        frozen['always-open-trained'])
    score = trained.summary()['scores']
    adjacent = trained.summary()['event_adjacent_quiet']
    initial_score = initial.summary()['scores']
    criteria = dict(on_anticipation_at_least_point_one=
                        score['on']['anticipation'] >= .1,
                    off_anticipation_at_least_point_one=
                        score['off']['anticipation'] >= .1,
                    on_mse_improved=score['on']['mse'] < initial_score['on']['mse'],
                    off_mse_improved=score['off']['mse'] < initial_score['off']['mse'],
                    event_adjacent_quiet_alarms_at_most_five_percent=
                        adjacent['fraction'] is not None and adjacent['fraction'] <= .05)
    report = dict(source_sha256=source_sha, train_seed=1101, evaluation_seed=1102,
                  frames=FRAMES, horizon_ticks=8, selected_eta=1.,
                  always_open_training=dict(checkpoint_sha256=checksum(train_path),
                                            **train.summary()),
                  frozen_evaluations=results,
                  original_gated_control=original,
                  always_open_initial_to_trained=delta(initial, trained),
                  gated_initial_to_trained=delta(
                      frozen['gated-initial-control'], frozen['gated-trained-control']),
                  criteria=criteria, all_criteria_pass=all(criteria.values()),
                  camera_event_sequences_equal=True,
                  final_seeds_untouched=[1201, 1202, 1203, 1204],
                  script_sha256=checksum(Path(__file__)))
    output = OUT/'pilot-500.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
