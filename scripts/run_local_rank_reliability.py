"""Evaluate direct-error source reliability for two visual predictors."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from local_rank_reliability import fit_reliability
from run_complementary_prediction import collect_samples, score_samples
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_sparse_recurrent_field import (empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)


OUT = Path('docs/experiments/2026-09-26-local-rank-reliability-results.json')
BASELINE = Path('docs/experiments/2026-09-26-complementary-prediction-results.json')


@torch.no_grad()
def score_learned(episodes, models):
    rows = {name: dict(empty_forecast(), cases=[])
            for name in models}
    for samples in episodes:
        local = {name: empty_forecast() for name in models}
        for event, fast, files in samples:
            for name, model in models.items():
                forecast = model.forecast(fast, files)
                for row in (rows[name], local[name]):
                    row['events'] += 1
                    row['targets'] += int((event > 0).sum())
                    row['positive_sites'] = row.get('positive_sites', 0)+int(
                        (forecast > 0).sum())
                    row['top_8_hits'] += rank_hits(forecast, event, 8)
                    row['top_32_hits'] += rank_hits(forecast, event, 32)
        for name in models:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
        row['mean_positive_sites'] = row.get('positive_sites', 0)/max(
            row['events'], 1)
    return rows


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_transitions(fit_dictionary(training), training)['aligned']
    links = fit_diagonal_affinity(training)['aligned']
    training_samples = collect_samples(fast, training, links)
    models = {name: fit_reliability(training_samples, shuffled=shuffled)
              for name, shuffled in (('aligned', False), ('shuffled', True))}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case in make_cases():
        groups.setdefault(case['split'], []).append(case_events(case))
    results = {}
    baseline = json.loads(BASELINE.read_text())
    for split, episodes in groups.items():
        samples = collect_samples(fast, episodes, links)
        matched = score_samples(samples)
        assert matched['equal_fusion']['top_32_hits'] == baseline[
            'results'][split]['equal_fusion']['top_32_hits']
        matched.update(score_learned(samples, models))
        results[split] = matched
    result = dict(training_episodes=len(training),
                  training_events=sum(len(row) for row in training_samples),
                  weights={name: model.weight.tolist()
                           for name, model in models.items()},
                  counts={name: model.count.tolist()
                          for name, model in models.items()},
                  results=results,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={split: {name: round(row['top_32_recall'], 3)
                        for name, row in arms.items()}
                for split, arms in results.items()},
        top_8={split: {name: round(row['top_8_recall'], 3)
                       for name, row in arms.items()}
               for split, arms in results.items()},
        training_events=result['training_events'],
        mean_positive_sites={split: {
            name: round(arms[name]['mean_positive_sites'], 1)
            for name in models} for split, arms in results.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
