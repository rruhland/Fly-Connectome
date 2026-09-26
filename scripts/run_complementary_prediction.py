"""Check whether generic fast-field and entity-file futures complement."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from persistent_entity_files import PersistentEntityFiles
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_sparse_recurrent_field import (empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)


OUT = Path('docs/experiments/2026-09-26-complementary-prediction-results.json')
BASELINE = Path('docs/experiments/2026-09-26-local-affinity-binding-results.json')
ARMS = ('fast_only', 'entity_files', 'equal_fusion',
        'shuffled_entity_fusion')


def reciprocal_rank_fusion(fast, files):
    def rank_field(forecast):
        flat = forecast.flatten()
        order = flat.argsort(descending=True)
        score = torch.empty_like(flat)
        score[order] = 1/torch.arange(1, flat.numel()+1,
                                      dtype=flat.dtype)
        return score.reshape(forecast.shape)
    return rank_field(fast)+rank_field(files)


@torch.no_grad()
def collect_samples(fast, episodes, links):
    result = []
    for events in episodes:
        fast.reset_state()
        files = PersistentEntityFiles(diagonal_links=links)
        samples = []
        pending = None
        for event in events:
            if bool(event.any()) and pending is not None:
                samples.append((event, *pending))
            fast.step(event)
            files.step(event)
            if bool(event.any()):
                pending = (fast.forecast(), files.forecast())
        result.append(samples)
    return result


@torch.no_grad()
def score_samples(episodes):
    rows = {name: dict(empty_forecast(), cases=[]) for name in ARMS}
    flat = [sample for samples in episodes for sample in samples]
    positions = []
    for case_index, samples in enumerate(episodes):
        positions.extend([case_index]*len(samples))
        for name in ARMS:
            rows[name]['cases'].append(empty_forecast())
    for index, (event, fast, files) in enumerate(flat):
        shuffled = flat[(index+max(len(flat)//3, 1)) % len(flat)][2]
        forecasts = dict(fast_only=fast, entity_files=files,
                         equal_fusion=reciprocal_rank_fusion(fast, files),
                         shuffled_entity_fusion=
                         reciprocal_rank_fusion(fast, shuffled))
        for name, forecast in forecasts.items():
            for row in (rows[name], rows[name]['cases'][positions[index]]):
                row['events'] += 1
                row['targets'] += int((event > 0).sum())
                row['top_8_hits'] += rank_hits(forecast, event, 8)
                row['top_32_hits'] += rank_hits(forecast, event, 32)
        first = set(fast.flatten().topk(32).indices.tolist())
        second = set(files.flatten().topk(32).indices.tolist())
        rows['equal_fusion']['source_overlap_top32'] = (
            rows['equal_fusion'].get('source_overlap_top32', 0)
            +len(first & second))
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    return rows


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_transitions(fit_dictionary(training), training)['aligned']
    links = fit_diagonal_affinity(training)['aligned']
    cases = make_cases()
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case in cases:
        groups.setdefault(case['split'], []).append(case_events(case))
    results = {name: score_samples(collect_samples(fast, episodes, links))
               for name, episodes in groups.items()}
    baseline = json.loads(BASELINE.read_text())
    for split, arms in results.items():
        assert arms['fast_only']['top_32_hits'] == baseline[
            'prior_fast_only'][split]['fast_only']['top_32_hits']
        assert arms['entity_files']['top_32_hits'] == baseline[
            'forecasts'][split]['aligned']['top_32_hits']
    result = dict(training_episodes=len(training), results=results,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={split: {name: round(row['top_32_recall'], 3)
                        for name, row in arms.items()}
                for split, arms in results.items()},
        top_8={split: {name: round(row['top_8_recall'], 3)
                       for name, row in arms.items()}
               for split, arms in results.items()},
        generic_overlap=round(results['generic_heldout'][
            'equal_fusion']['source_overlap_top32']/results[
                'generic_heldout']['equal_fusion']['events'], 2),
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
