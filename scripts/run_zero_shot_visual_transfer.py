"""Zero-shot visual forecast transfer to Pong and corrupted generic cameras."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from local_rank_reliability import fit_reliability
from persistent_entity_files import PersistentEntityFiles
from run_complementary_prediction import collect_samples, score_samples
from run_local_rank_reliability import score_learned
from run_pong_camera_transfer import pong_events
from run_sparse_recurrent_field import fit_dictionary, fit_transitions


OUT = Path('docs/experiments/2026-09-26-zero-shot-visual-transfer-results.json')
PRIOR = Path('docs/experiments/2026-09-26-local-rank-reliability-results.json')
PONG_SEEDS = (1101, 1102, 1103, 1104)
CORRUPTED_SEEDS = (2000, 2001, 2002, 2003)


@torch.no_grad()
def corrupt_events(events, *, seed, dropout=.1, false_rate=.001):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for event in events:
        retained = event*(torch.rand(event.shape,
                                     generator=generator) >= dropout)
        false = (torch.rand(event.shape,
                            generator=generator) < false_rate).float()
        result.append(torch.maximum(retained, false))
    return result


@torch.no_grad()
def collect_corrupted_samples(fast, episodes, links):
    result = []
    for clean, observed in episodes:
        fast.reset_state()
        files = PersistentEntityFiles(diagonal_links=links)
        pending = None
        samples = []
        for target, event in zip(clean, observed):
            if bool(target.any()) and pending is not None:
                samples.append((target, *pending))
            fast.step(event)
            files.step(event)
            if bool(event.any()):
                pending = (fast.forecast(), files.forecast())
        result.append(samples)
    return result


@torch.no_grad()
def file_activity(episodes, links):
    counts = []
    births = []
    for events in episodes:
        files = PersistentEntityFiles(diagonal_links=links)
        for event in events:
            files.step(event)
            if bool(event.any()):
                counts.append(len(files.live_slots))
        births.append(files.next_id)
    return dict(active_frames=len(counts),
                mean_live_files=sum(counts)/max(len(counts), 1),
                max_live_files=max(counts, default=0),
                mean_total_births=sum(births)/max(len(births), 1))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_transitions(fit_dictionary(training), training)['aligned']
    links = fit_diagonal_affinity(training)['aligned']
    training_samples = collect_samples(fast, training, links)
    readouts = {name: fit_reliability(training_samples,
                                      shuffled=shuffled)
                for name, shuffled in (('aligned', False), ('shuffled', True))}
    groups = {f'pong_stride_{stride}': [
        pong_events(seed, stride=stride, frames=120)
        for seed in PONG_SEEDS] for stride in (1, 4)}
    results = {}
    activity = {}
    for name, episodes in groups.items():
        samples = collect_samples(fast, episodes, links)
        scores = score_samples(samples)
        scores.update(score_learned(samples, readouts))
        results[name] = scores
        activity[name] = file_activity(episodes, links)
    clean = [scene_events(seed, heldout=True)
             for seed in CORRUPTED_SEEDS]
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip(CORRUPTED_SEEDS, clean)]
    samples = collect_corrupted_samples(fast, list(zip(clean, noisy)), links)
    scores = score_samples(samples)
    scores.update(score_learned(samples, readouts))
    results['generic_corrupted'] = scores
    activity['generic_corrupted'] = file_activity(noisy, links)
    prior = json.loads(PRIOR.read_text())
    assert readouts['aligned'].weight.tolist() == prior['weights']['aligned']
    result = dict(training_episodes=len(training),
                  pong_seeds=PONG_SEEDS,
                  corrupted_seeds=CORRUPTED_SEEDS,
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
        file_activity=activity,
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
