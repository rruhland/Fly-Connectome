"""One-shot local selector between two frozen learned temporal pathways."""

import json
import random
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_temporal_mixture import LocalTemporalMixture
from run_absolute_refresh import contrast_frames
from run_distributed_predictive_field import fit_fields
from run_local_observation_model import fit_observers
from run_multicadence_experience import training_sequences, transfer_groups
from run_sparse_recurrent_field import empty_forecast, rank_hits


OUT = Path('docs/experiments/2026-09-26-local-temporal-mixture-results.json')
PRIOR_SLOW = Path('docs/experiments/2026-09-26-observation-model-transfer-results.json')
PRIOR_DIVERSE = Path('docs/experiments/2026-09-26-multicadence-experience-results.json')
ARMS = ('slow', 'diverse', 'equal', 'aligned', 'shuffled_gate')


@torch.no_grad()
def fit_gates(episodes, experts, observer):
    gates = {name: LocalTemporalMixture()
             for name in ('aligned', 'shuffled')}
    for index, events in enumerate(episodes):
        observer.reset_state()
        for model in experts.values():
            model.reset_state()
        absolute = contrast_frames(events)
        targets = [event for event in events if bool(event.any())][1:]
        shuffled = targets.copy()
        random.Random(7000+index).shuffle(shuffled)
        pending = None
        target_index = 0
        for frame, event in enumerate(events):
            if bool(event.any()) and pending is not None:
                gates['aligned'].credit(*pending, event)
                gates['shuffled'].credit(*pending,
                                         shuffled[target_index])
                target_index += 1
            intensity = absolute[frame] if frame % 8 == 0 else None
            filtered, _ = observer.step(event, absolute=intensity)
            states = {name: model.step(filtered, absolute=intensity)
                      for name, model in experts.items()}
            if bool(event.any()):
                features = gates['aligned'].features(
                    states['slow'], states['diverse'])
                pending = (features, experts['slow'].forecast(),
                           experts['diverse'].forecast())
    observer.reset_state()
    for model in experts.values():
        model.reset_state()
    return gates


@torch.no_grad()
def evaluate_streams(streams, experts, observer, gates):
    scores = {name: dict(empty_forecast(), cases=[],
                         positive_sites=0) for name in ARMS}
    for clean, observed, _, absolute, _ in streams:
        observer.reset_state()
        for model in experts.values():
            model.reset_state()
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        for frame, (target, event) in enumerate(zip(clean, observed)):
            if bool(target.any()) and pending is not None:
                for name, forecast in pending.items():
                    for row in (scores[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((target > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, target, 8)
                        row['top_32_hits'] += rank_hits(forecast, target, 32)
                    scores[name]['positive_sites'] += int((forecast > 0).sum())
            intensity = absolute[frame] if frame % 8 == 0 else None
            filtered, _ = observer.step(event, absolute=intensity)
            states = {name: model.step(filtered, absolute=intensity)
                      for name, model in experts.items()}
            if bool(event.any()):
                slow = experts['slow'].forecast()
                diverse = experts['diverse'].forecast()
                features = gates['aligned'].features(
                    states['slow'], states['diverse'])
                pending = dict(
                    slow=slow, diverse=diverse,
                    equal=.5*(slow+diverse),
                    aligned=gates['aligned'].forecast(
                        features, slow, diverse),
                    shuffled_gate=gates['shuffled'].forecast(
                        features, slow, diverse))
        for name in ARMS:
            scores[name]['cases'].append(local[name])
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['positive_sites_per_event'] = row['positive_sites']/max(
            row['events'], 1)
    return scores


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observer = fit_observers(original)['aligned']
    _, diverse_training = training_sequences()
    experts = dict(
        slow=fit_fields([events for events, _ in original])['aligned'],
        diverse=fit_fields(diverse_training)['aligned'])
    expert_seconds = time.perf_counter()-started
    print('experts trained in', round(expert_seconds, 1), 'seconds',
          flush=True)
    gates = fit_gates(diverse_training, experts, observer)
    gate_seconds = time.perf_counter()-started-expert_seconds
    print('selectors trained in', round(gate_seconds, 1), 'seconds',
          flush=True)
    scores = {}
    for split, streams in transfer_groups().items():
        scores[split] = evaluate_streams(streams, experts, observer, gates)
        print(split, {name: round(row['top_32_recall'], 3)
                      for name, row in scores[split].items()}, flush=True)
    prior_slow = json.loads(PRIOR_SLOW.read_text())
    prior_diverse = json.loads(PRIOR_DIVERSE.read_text())
    for split in ('position', 'speed', 'shape', 'separated', 'crossing',
                  'pong_stride_1', 'pong_stride_4', 'known_noise'):
        assert scores[split]['slow']['top_32_hits'] == prior_slow[
            'scores'][split]['learned_aligned']['top_32_hits']
        assert scores[split]['diverse']['top_32_hits'] == prior_diverse[
            'scores']['diverse'][split]['learned_aligned']['top_32_hits']
    result = dict(training_episodes=len(diverse_training),
                  expert_seconds=expert_seconds,
                  gate_seconds=gate_seconds,
                  scores=scores,
                  aligned_gate_weights=gates['aligned'].weights.tolist(),
                  aligned_gate_bias=gates['aligned'].bias.flatten().tolist(),
                  shuffled_gate_weights=gates['shuffled'].weights.tolist(),
                  shuffled_gate_bias=gates['shuffled'].bias.flatten().tolist(),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={split: {name: round(row['top_32_recall'], 3)
                        for name, row in arms.items()}
                for split, arms in scores.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
