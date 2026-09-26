"""Matched delayed local-path comparison on the frozen visual test suite."""

import json
import time
from pathlib import Path

import torch

from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from multi_hypothesis_assemblies import MultiHypothesisAssemblies
from run_decisive_representation_audit import make_cases
from run_event_centric_competition import train_state
from run_local_motion_assemblies import (ARMS, candidate_episode,
                                         case_events, fit_affinity,
                                         score_tracks)


OUT = Path('docs/experiments/2026-09-25-multi-hypothesis-assembly-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    episodes = [candidate_episode(model, events) for events in training]
    affinities = dict(learned=fit_affinity(episodes),
                      shuffled_credit=fit_affinity(episodes, shuffled=True))
    trained_seconds = time.perf_counter()-started
    print(f'local transition affinities trained in '
          f'{trained_seconds:.1f}s', flush=True)
    scores = {split: {name: dict(correct=0, matched=0, total=0,
                                 coverage=0., tracks=0, cases=[])
                      for name in ARMS}
              for split in ('calibration', 'position', 'speed', 'shape',
                            'separated', 'crossing')}
    for case in make_cases():
        candidates = candidate_episode(model, case_events(case))
        for name in ARMS:
            assemblies = MultiHypothesisAssemblies(
                affinity=affinities.get(name))
            for frame, proposals in enumerate(candidates):
                assemblies.step(frame, proposals)
            tracks = assemblies.tracks()
            row = score_tracks(case, tracks)
            aggregate = scores[case['split']][name]
            for key in ('correct', 'matched', 'total', 'coverage'):
                aggregate[key] += row[key]
            aggregate['tracks'] += len(tracks)
            aggregate['cases'].append(dict(polarity=case['polarity'],
                                           objects=row['objects'],
                                           tracks=len(tracks)))
    for group in scores.values():
        for row in group.values():
            row['accuracy'] = row['correct']/max(row['total'], 1)
            row['matched_fraction'] = row['matched']/max(row['total'], 1)
            row['mean_coverage'] = row['coverage']/max(row['total'], 1)
            row['tracks_per_case'] = row['tracks']/max(len(row['cases']), 1)
    result = dict(training_episodes=len(training),
                  training_seconds=trained_seconds, scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({split: {name: dict(
        accuracy=round(row['accuracy'], 3),
        matched=round(row['matched_fraction'], 3),
        coverage=round(row['mean_coverage'], 3),
        tracks=round(row['tracks_per_case'], 2))
        for name, row in arms.items()}
        for split, arms in scores.items()}), flush=True)


if __name__ == '__main__':
    main()
