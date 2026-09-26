"""Evaluate autonomously selected local assemblies with learned motion priors."""

import json
import random
import time
from pathlib import Path

import torch

from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from learned_assembly_consolidation import (LocalDisplacementExpectation,
                                            consolidate_paths)
from multi_hypothesis_assemblies import MultiHypothesisAssemblies
from run_decisive_representation_audit import make_cases
from run_event_centric_competition import train_state
from run_local_motion_assemblies import (ARMS, candidate_episode,
                                         case_events, score_tracks)


OUT = Path('docs/experiments/2026-09-25-learned-assembly-consolidation-results.json')


def fit_expectation(episodes, *, shuffled=False):
    learned = LocalDisplacementExpectation(features=26, radius=3)
    for index, episode in enumerate(episodes):
        active = [(frame, candidates) for frame, candidates in
                  enumerate(episode) if candidates]
        targets = active[1:].copy()
        if shuffled:
            random.Random(1000+index).shuffle(targets)
        for (source_frame, previous), (target_frame, current) in zip(
                active[:-1], targets):
            if not shuffled and target_frame-source_frame > 8:
                continue
            learned.observe(previous, current)
    learned.finalize()
    return learned


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    episodes = [candidate_episode(model, events) for events in training]
    learned = fit_expectation(episodes)
    shuffled = fit_expectation(episodes, shuffled=True)
    expectations = dict(learned=learned, shuffled_credit=shuffled)
    training_seconds = time.perf_counter()-started
    print(f'local displacement expectations trained in '
          f'{training_seconds:.1f}s', flush=True)
    scores = {split: {name: dict(correct=0, matched=0, total=0,
                                 coverage=0., tracks=0, false_tracks=0,
                                 top_k_correct=0, top_k_matched=0,
                                 cases=[])
                      for name in ARMS}
              for split in ('calibration', 'position', 'speed', 'shape',
                            'separated', 'crossing')}
    for case in make_cases():
        candidates = candidate_episode(model, case_events(case))
        for name in ARMS:
            bank = MultiHypothesisAssemblies(
                affinity=expectations.get(name))
            for frame, proposals in enumerate(candidates):
                bank.step(frame, proposals)
            tracks = consolidate_paths(bank.all_hypotheses)
            row = score_tracks(case, tracks)
            top_k = score_tracks(case, tracks[:len(case['objects'])])
            aggregate = scores[case['split']][name]
            for key in ('correct', 'matched', 'total', 'coverage'):
                aggregate[key] += row[key]
            aggregate['tracks'] += len(tracks)
            aggregate['false_tracks'] += len(tracks)-row['matched']
            aggregate['top_k_correct'] += top_k['correct']
            aggregate['top_k_matched'] += top_k['matched']
            aggregate['cases'].append(dict(polarity=case['polarity'],
                                           objects=row['objects'],
                                           tracks=len(tracks)))
    for group in scores.values():
        for row in group.values():
            row['accuracy'] = row['correct']/max(row['total'], 1)
            row['matched_fraction'] = row['matched']/max(row['total'], 1)
            row['mean_coverage'] = row['coverage']/max(row['total'], 1)
            row['tracks_per_case'] = row['tracks']/max(len(row['cases']), 1)
            row['false_tracks_per_case'] = row['false_tracks']/max(
                len(row['cases']), 1)
            row['top_k_accuracy'] = row['top_k_correct']/max(row['total'], 1)
            row['top_k_matched_fraction'] = row['top_k_matched']/max(
                row['total'], 1)
    result = dict(training_episodes=len(training),
                  aligned_feature_mass=float(learned.mass.sum()),
                  shuffled_feature_mass=float(shuffled.mass.sum()),
                  aligned_expectation_mean_norm=float(
                      learned.vector.norm(dim=1).mean()),
                  shuffled_expectation_mean_norm=float(
                      shuffled.vector.norm(dim=1).mean()),
                  training_seconds=training_seconds, scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({split: {name: dict(
        accuracy=round(row['accuracy'], 3),
        top_k_accuracy=round(row['top_k_accuracy'], 3),
        coverage=round(row['mean_coverage'], 3),
        tracks=round(row['tracks_per_case'], 2),
        false_tracks=round(row['false_tracks_per_case'], 2))
        for name, row in arms.items()}
        for split, arms in scores.items()}), flush=True)


if __name__ == '__main__':
    main()
