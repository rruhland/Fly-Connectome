"""Post hoc fixed spatial probe of slow assembly state."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generic_native_cadence import scene_events
from run_decisive_representation_audit import (ACTIVE, fit_centroids,
                                               make_cases,
                                               score as probe_score)
from run_local_motion_assemblies import case_events, target_center
from run_sparse_recurrent_field import fit_dictionary, fit_transitions
from run_two_timescale_assemblies import (SLOW_ARMS, fit_slow_dictionary,
                                          fit_slow_transitions, step_fast)


OUT = Path('docs/experiments/2026-09-26-spatial-state-audit-results.json')


def spatial_feature(state, center, *, is_fast=False):
    if is_fast:
        state = F.max_pool2d(state[None], 4, stride=4)[0]
    y, x = (int(round(coordinate/4)) for coordinate in center)
    padded = F.pad(state, (1, 1, 1, 1))
    return padded[:, y:y+3, x:x+3].flatten()


@torch.no_grad()
def evaluate(fast, models, cases, events):
    names = ('fast_only', *SLOW_ARMS)
    rows = {name: {} for name in names}
    for case, episode in zip(cases, events):
        fast.reset_state()
        for model in models.values():
            model.reset_state()
        features = {name: [torch.zeros((fast.units if name ==
                                        'fast_only' else models[name].units)*9)
                           for _ in case['objects']]
                    for name in names}
        for frame, event in enumerate(episode):
            incoming = step_fast(fast, event)
            for model in models.values():
                model.step(incoming)
            if frame not in ACTIVE:
                continue
            for index, item in enumerate(case['objects']):
                center = target_center(item, frame)
                features['fast_only'][index] += spatial_feature(
                    fast.state, center, is_fast=True)
                for name, model in models.items():
                    features[name][index] += spatial_feature(
                        model.state, center)
        for name in names:
            split = case['split']
            rows[name].setdefault(split, [])
            for item, feature in zip(case['objects'], features[name]):
                rows[name][split].append(dict(direction=item['direction'],
                                              polarity=case['polarity'],
                                              feature=feature))
    result = {}
    for name, groups in rows.items():
        centroids = fit_centroids(groups['calibration'], 'feature')
        result[name] = {split: probe_score(samples, centroids, 'feature')
                        for split, samples in groups.items()}
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_dictionary(training)
    fast = fit_transitions(fast, training)['aligned']
    slow = fit_slow_dictionary(fast, training)
    models = fit_slow_transitions(fast, slow, training)
    training_seconds = time.perf_counter()-started
    print(f'frozen circuits reconstructed in {training_seconds:.1f}s',
          flush=True)
    cases = make_cases()
    events = [case_events(case) for case in cases]
    probe = evaluate(fast, models, cases, events)
    result = dict(training_episodes=len(training),
                  training_seconds=training_seconds,
                  probe=probe,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {
        split: f"{row['correct']}/{row['total']}"
        for split, row in groups.items()}
        for name, groups in probe.items()}), flush=True)


if __name__ == '__main__':
    main()
