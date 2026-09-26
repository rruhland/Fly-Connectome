"""Compare provisional and fast-conditioned entity files without labels."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from persistent_entity_files import PersistentEntityFiles
from predictive_file_evidence import PredictiveEvidenceFiles
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_pong_camera_transfer import pong_events
from run_sparse_recurrent_field import (empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-predictive-file-evidence-results.json')
PRIOR_CLEAN = Path('docs/experiments/2026-09-26-local-affinity-binding-results.json')
PRIOR_TRANSFER = Path('docs/experiments/2026-09-26-zero-shot-visual-transfer-results.json')
FILE_ARMS = ('base', 'confirmation', 'conditioned',
             'combined', 'shuffled')
ARMS = ('fast_only', *FILE_ARMS)


def make_models(links):
    return dict(
        base=PersistentEntityFiles(diagonal_links=links),
        confirmation=PredictiveEvidenceFiles(
            diagonal_links=links, condition_forecast=False),
        conditioned=PredictiveEvidenceFiles(
            diagonal_links=links, confirm_birth=False),
        combined=PredictiveEvidenceFiles(diagonal_links=links),
        shuffled=PredictiveEvidenceFiles(diagonal_links=links))


@torch.no_grad()
def fast_evidence(fast, episodes):
    evidence = []
    for _, observed in episodes:
        fast.reset_state()
        rows = []
        prior = None
        for event in observed:
            fast.step(event)
            current = fast.forecast()
            rows.append((prior, current))
            if bool(event.any()):
                prior = current
        evidence.append(rows)
    return evidence


@torch.no_grad()
def evaluate_episodes(fast, episodes, links):
    evidence = fast_evidence(fast, episodes)
    scores = {name: dict(empty_forecast(), cases=[]) for name in ARMS}
    counts = {name: [] for name in FILE_ARMS}
    births = {name: [] for name in FILE_ARMS}
    for index, (clean, observed) in enumerate(episodes):
        models = make_models(links)
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        foreign = evidence[(index+1) % len(evidence)]
        for frame, (target, event) in enumerate(zip(clean, observed)):
            prior, current = evidence[index][frame]
            foreign_prior, foreign_current = foreign[frame]
            if bool(target.any()) and pending is not None:
                for name, forecast in pending.items():
                    for row in (scores[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((target > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, target, 8)
                        row['top_32_hits'] += rank_hits(forecast, target, 32)
            models['base'].step(event)
            models['confirmation'].step(event, support=prior)
            models['conditioned'].step(event, support=prior)
            models['combined'].step(event, support=prior)
            models['shuffled'].step(event, support=foreign_prior)
            if bool(event.any()):
                for name, model in models.items():
                    counts[name].append(len(model.live_slots))
                pending = dict(
                    fast_only=current,
                    base=models['base'].forecast(),
                    confirmation=models['confirmation'].forecast(),
                    conditioned=models['conditioned'].forecast(current),
                    combined=models['combined'].forecast(current),
                    shuffled=models['shuffled'].forecast(foreign_current))
        for name in ARMS:
            scores[name]['cases'].append(local[name])
        for name, model in models.items():
            births[name].append(model.next_id)
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    activity = {name: dict(active_frames=len(counts[name]),
                           mean_live_files=sum(counts[name])/max(
                               len(counts[name]), 1),
                           max_live_files=max(counts[name], default=0),
                           mean_total_births=sum(births[name])/max(
                               len(births[name]), 1))
                for name in FILE_ARMS}
    return scores, activity


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_transitions(fit_dictionary(training), training)['aligned']
    links = fit_diagonal_affinity(training)['aligned']
    groups = dict(generic_heldout=[(events, events)
                                   for events in (scene_events(
                                       1000+seed, heldout=True)
                                       for seed in range(16))])
    for case in make_cases():
        events = case_events(case)
        groups.setdefault(case['split'], []).append((events, events))
    for stride in (1, 4):
        groups[f'pong_stride_{stride}'] = [(events, events)
            for events in (pong_events(seed, stride=stride, frames=120)
                           for seed in (1101, 1102, 1103, 1104))]
    groups['generic_corrupted'] = []
    for seed in (2000, 2001, 2002, 2003):
        clean = scene_events(seed, heldout=True)
        groups['generic_corrupted'].append(
            (clean, corrupt_events(clean, seed=seed)))
    results = {}
    activity = {}
    for split, episodes in groups.items():
        results[split], activity[split] = evaluate_episodes(
            fast, episodes, links)
    clean_prior = json.loads(PRIOR_CLEAN.read_text())
    transfer_prior = json.loads(PRIOR_TRANSFER.read_text())
    for split, arms in results.items():
        if split in clean_prior['forecasts']:
            expected = clean_prior['forecasts'][split]['aligned']
        else:
            expected = transfer_prior['results'][split]['entity_files']
        assert arms['base']['top_32_hits'] == expected['top_32_hits']
    result = dict(training_episodes=len(training),
                  results=results, file_activity=activity,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={split: {name: round(row['top_32_recall'], 3)
                        for name, row in arms.items()}
                for split, arms in results.items()},
        top_8={split: {name: round(row['top_8_recall'], 3)
                       for name, row in arms.items()}
               for split, arms in results.items()},
        activity={split: {name: dict(
            live=round(row['mean_live_files'], 2),
            max=row['max_live_files'],
            births=round(row['mean_total_births'], 2))
            for name, row in arms.items()}
            for split, arms in activity.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
