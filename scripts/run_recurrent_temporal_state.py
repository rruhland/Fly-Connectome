"""One registered test of a full-resolution locally recurrent visual state."""

import json
import random
import time
from pathlib import Path

import torch

from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from recurrent_temporal_state import RecurrentTemporalState
from run_absolute_refresh import contrast_frames
from run_local_observation_model import fit_observers
from run_multicadence_experience import training_sequences, transfer_groups
from run_observation_model_transfer import make_streams
from run_sparse_recurrent_field import empty_forecast, rank_hits
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-recurrent-temporal-state-results.json')
PRIOR = Path('docs/experiments/2026-09-26-local-temporal-mixture-results.json')
ARMS = ('aligned', 'frozen', 'shuffled', 'no_recurrence', 'state_reset')
GATES = dict(clean_generic=.617, known_noise=.572, speed=.376,
             separated=.307, pong_stride_1=.312)


@torch.no_grad()
def fit_models(episodes, observer):
    models = {name: RecurrentTemporalState()
              for name in ('aligned', 'frozen', 'shuffled')}
    for index, events in enumerate(episodes):
        observer.reset_state()
        filtered = []
        # The observer receives the same sparse intensity refresh as at test.
        absolute = contrast_frames(events)
        for frame, event in enumerate(events):
            filtered.append(observer.step(
                event, absolute=absolute[frame] if frame % 8 == 0 else None)[0])
        shuffled = filtered.copy()
        random.Random(9000+index).shuffle(shuffled)
        for name, model in models.items():
            model.reset_state()
            previous = None
            for event, sensory, shuffled_sensory in zip(
                    events, filtered, shuffled):
                if previous is not None:
                    model.credit_emission(previous, event)
                    if name != 'frozen':
                        model.credit_transition(
                            previous, sensory if name == 'aligned'
                            else shuffled_sensory)
                previous = model.step(sensory)
        if (index+1) % 64 == 0:
            print('trained episodes', index+1, flush=True)
    return models


def _finish(row):
    row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
    row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
    row['quiet_mass_per_frame'] = row['quiet_mass']/max(row['quiet_frames'], 1)
    row['mean_state_after_quiet'] = row['quiet_state_mass']/max(
        row['quiet_frames'], 1)


@torch.no_grad()
def evaluate(streams, models, observer):
    rows = {name: dict(empty_forecast(), cases=[], quiet_frames=0,
                       quiet_mass=0., quiet_state_mass=0.,
                       immediate_targets=0, immediate_top_32_hits=0)
            for name in ARMS}
    for clean, observed, _, absolute, _ in streams:
        observer.reset_state()
        for model in models.values():
            model.reset_state()
        no_recurrence = RecurrentTemporalState()
        state_reset = RecurrentTemporalState()
        for model in (no_recurrence, state_reset):
            model.transition = models['aligned'].transition
            model.emission = models['aligned'].emission
        local = {name: empty_forecast() for name in ARMS}
        pending_event = None
        pending_frame = None
        for frame, (target, event) in enumerate(zip(clean, observed)):
            if bool(target.any()) and pending_frame is not None:
                for name, forecast in pending_frame.items():
                    rows[name]['immediate_targets'] += int((target > 0).sum())
                    rows[name]['immediate_top_32_hits'] += rank_hits(
                        forecast, target, 32)
            if bool(target.any()) and pending_event is not None:
                for name, forecast in pending_event.items():
                    for row in (rows[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((target > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, target, 8)
                        row['top_32_hits'] += rank_hits(forecast, target, 32)
            if not bool(target.any()) and pending_frame is not None:
                for name, forecast in pending_frame.items():
                    rows[name]['quiet_frames'] += 1
                    rows[name]['quiet_mass'] += float(forecast.sum())
            intensity = absolute[frame] if frame % 8 == 0 else None
            filtered, _ = observer.step(event, absolute=intensity)
            states = {name: model.step(filtered)
                      for name, model in models.items()}
            # Ablations have independent state but share trained weights.
            states['no_recurrence'] = no_recurrence.step(
                filtered, recurrent=False)
            state_reset.reset_state()
            states['state_reset'] = state_reset.step(filtered)
            pending_frame = {}
            for name, state in states.items():
                model = (no_recurrence if name == 'no_recurrence' else
                         state_reset if name == 'state_reset' else models[name])
                pending_frame[name] = model.forecast(state)
                if not bool(event.any()):
                    rows[name]['quiet_state_mass'] += float(state.sum())
            if bool(event.any()):
                pending_event = pending_frame
        for name in ARMS:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        _finish(row)
        row['immediate_top_32_recall'] = row['immediate_top_32_hits']/max(
            row['immediate_targets'], 1)
    return rows


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observer = fit_observers(original)['aligned']
    _, diverse = training_sequences()
    models = fit_models(diverse, observer)
    training_seconds = time.perf_counter()-started
    print('training seconds', round(training_seconds, 1), flush=True)
    groups = transfer_groups()
    stationary_path = [None]*18
    for frame in range(2, 15):
        stationary_path[frame] = (16, 32)
    stationary = path_events([('square', stationary_path)])
    groups['stationary_noisy'] = make_streams(
        [stationary], [corrupt_events(stationary, seed=0)])
    scores = {}
    for split, streams in groups.items():
        scores[split] = evaluate(streams, models, observer)
        print(split, {name: round(row['top_32_recall'], 3)
                      for name, row in scores[split].items()}, flush=True)
    prior = json.loads(PRIOR.read_text())
    passed = all(scores[split]['aligned']['top_32_recall'] >= gate
                 for split, gate in GATES.items())
    passed &= scores['pong_stride_1']['aligned']['top_8_recall'] >= .072
    passed &= all(scores[split]['aligned']['top_32_recall'] >
                  scores[split][control]['top_32_recall']
                  for split in GATES for control in ('frozen', 'shuffled'))
    passed &= scores['speed']['aligned']['top_32_recall'] > scores[
        'speed']['state_reset']['top_32_recall']
    result = dict(training_episodes=len(diverse), training_seconds=training_seconds,
                  evaluation_seconds=time.perf_counter()-started-training_seconds,
                  gates=GATES, passed_joint_gate=bool(passed),
                  prior_baselines={split: {name: prior['scores'][split][name][
                      'top_32_recall'] for name in ('slow', 'diverse', 'equal')}
                      for split in groups if split in prior['scores']},
                  scores=scores, elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('joint gate', passed, 'elapsed', round(result['elapsed_seconds'], 1),
          flush=True)


if __name__ == '__main__':
    main()
