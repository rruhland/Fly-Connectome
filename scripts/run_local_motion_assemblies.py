"""One-shot no-oracle inference test of learned local motion assemblies."""

import json
import random
import time
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera
from gap_timing_transfer import MODEL_OUT
from generic_motion_probe import events_map
from generic_native_cadence import scene_events as generic_scene_events
from local_motion_assemblies import (LocalTransitionAffinity,
                                     MotionAssemblies, local_candidates)
from run_decisive_representation_audit import (ACTIVE, VELOCITY,
                                               case_frames, make_cases)
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import sensory_sequence


OUT = Path('docs/experiments/2026-09-25-local-motion-assembly-results.json')
ARMS = ('learned', 'proximity', 'shuffled_credit')


@torch.no_grad()
def candidate_episode(model, events):
    model.reset_state()
    result = []
    for event, code, surface in sensory_sequence(events):
        model.step((code, surface))
        result.append(local_candidates(event, model.observed))
    return result


def fit_affinity(episodes, *, shuffled=False):
    affinity = LocalTransitionAffinity(features=26, radius=3)
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
            affinity.observe(previous, current)
    affinity.finalize()
    return affinity


def direction_from_history(history):
    visible = [(t, y, x) for t, y, x in history if t in ACTIVE]
    if len(visible) < 2:
        return None
    dy = visible[-1][1]-visible[0][1]
    dx = visible[-1][2]-visible[0][2]
    if max(abs(dy), abs(dx)) < 2 or abs(dy) == abs(dx):
        return None
    if abs(dy) > abs(dx):
        return 'down' if dy > 0 else 'up'
    return 'right' if dx > 0 else 'left'


def target_center(item, frame):
    dy, dx = VELOCITY[item['direction']]
    y, x = item['center']
    return (y+dy*item['speed']*(frame-8),
            x+dx*item['speed']*(frame-8))


def score_tracks(case, tokens):
    choices = []
    for entity, item in enumerate(case['objects']):
        for token_index, token in enumerate(tokens):
            hits = 0
            for frame, y, x in token['history']:
                if frame not in ACTIVE:
                    continue
                ty, tx = target_center(item, frame)
                hits += max(abs(y-ty), abs(x-tx)) <= 3
            if hits:
                choices.append((hits, entity, token_index))
    choices.sort(reverse=True)
    assigned = {}
    used_tokens = set()
    for hits, entity, token_index in choices:
        if entity not in assigned and token_index not in used_tokens:
            assigned[entity] = (hits, token_index)
            used_tokens.add(token_index)
    details = []
    for entity, item in enumerate(case['objects']):
        hits, token_index = assigned.get(entity, (0, None))
        matched = hits >= 3
        predicted = (direction_from_history(tokens[token_index]['history'])
                     if matched else None)
        details.append(dict(direction=item['direction'], prediction=predicted,
                            matched=matched, correct=predicted ==
                            item['direction'], coverage=hits/len(ACTIVE)))
    return dict(total=len(details), matched=sum(row['matched'] for row in details),
                correct=sum(row['correct'] for row in details),
                coverage=sum(row['coverage'] for row in details),
                objects=details)


@torch.no_grad()
def case_events(case):
    images, _ = case_frames(case)
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(case['polarity'] == 'dark')
    return [events_map(camera.observe(image)) for image in images]


@torch.no_grad()
def evaluate(model, cases, affinities):
    results = {split: {name: dict(correct=0, matched=0, total=0,
                                  coverage=0., cases=[])
                       for name in ARMS}
               for split in ('calibration', 'position', 'speed', 'shape',
                             'separated', 'crossing')}
    source_sites = 0
    source_frames = 0
    for case in cases:
        episodes = candidate_episode(model, case_events(case))
        source_sites += sum(len(candidates) for candidates in episodes)
        source_frames += len(episodes)
        for name in ARMS:
            tracker = MotionAssemblies(affinity=affinities.get(name))
            for frame, candidates in enumerate(episodes):
                tracker.step(frame, candidates)
            row = score_tracks(case, tracker.tokens)
            result = results[case['split']][name]
            for key in ('correct', 'matched', 'total', 'coverage'):
                result[key] += row[key]
            result['cases'].append(dict(polarity=case['polarity'],
                                        objects=row['objects'],
                                        tokens=len(tracker.tokens)))
    for split in results.values():
        for result in split.values():
            result['accuracy'] = result['correct']/max(result['total'], 1)
            result['matched_fraction'] = result['matched']/max(
                result['total'], 1)
            result['mean_coverage'] = result['coverage']/max(result['total'], 1)
    return dict(scores=results,
                source_sites_per_frame=source_sites/max(source_frames, 1))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [generic_scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    episode_candidates = [candidate_episode(model, events)
                          for events in training]
    learned = fit_affinity(episode_candidates)
    shuffled = fit_affinity(episode_candidates, shuffled=True)
    train_seconds = time.perf_counter()-started
    print(f'local assemblies trained in {train_seconds:.1f}s', flush=True)
    groups = evaluate(model, make_cases(),
                      dict(learned=learned, shuffled_credit=shuffled))
    result = dict(training_episodes=len(training),
                  aligned_local_pair_mass=float(learned.counts.sum()),
                  shuffled_local_pair_mass=float(shuffled.counts.sum()),
                  training_seconds=train_seconds, **groups,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({split: {name: dict(
        accuracy=round(row['accuracy'], 3),
        matched=round(row['matched_fraction'], 3),
        coverage=round(row['mean_coverage'], 3))
        for name, row in arms.items()}
        for split, arms in groups['scores'].items()}), flush=True)


if __name__ == '__main__':
    main()
