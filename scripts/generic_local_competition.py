"""Opt-in local competition test for unlabeled event-transition learning."""

import json
import random
import time
from pathlib import Path

from generic_local_transition import LocalTripletLearner, event_sequence, evaluate, item
from generic_motion_probe import SHAPES


OUT = Path('docs/experiments/2026-09-23-generic-local-competition-results.json')


def main():
    started = time.perf_counter()
    learner = LocalTripletLearner(eta=.3, local_competition=True)
    training = []
    for direction in ('up', 'down', 'left', 'right'):
        centers = ((16, 20), (16, 44)) if direction in ('up', 'down') else (
            (10, 32), (22, 32))
        for center in centers:
            for speed in (1, 2):
                for background in (True, False):
                    training.append(event_sequence(
                        [item('square', center, direction, speed)],
                        background=background))
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(training)
        for sequence in training:
            learner.reset_state()
            for event in sequence:
                learner.step(event)

    cases = []
    for shape in SHAPES:
        for direction in ('up', 'down', 'left', 'right'):
            for speed in (1, 2):
                for background in (True, False):
                    cases.append((f'{shape}:{direction}:s{speed}:'
                                  f'{"dark" if background else "bright"}',
                        event_sequence([item(shape, (16, 32), direction, speed)],
                                       background=background)))
    for background in (True, False):
        static = dict(shape='ell', center=(16, 32),
                      velocity=(0, 0), moving=False)
        cases.append((f'static:{"dark" if background else "bright"}',
                      event_sequence([static], background=background)))
    cases.append(('two_objects_same_speed', event_sequence([
        item('square', (16, 20), 'up', 1),
        item('plus', (16, 44), 'down', 1)], background=True)))
    cases.append(('two_objects_mixed_speed', event_sequence([
        item('square', (16, 20), 'up', 1),
        item('plus', (16, 44), 'down', 2)], background=True)))

    result = dict(eta=learner.eta, competition_window=7,
                  training_episodes=3*len(training),
                  confirmed_local_updates=learner.confirmed_updates,
                  weights=learner.weights.tolist(),
                  evaluation=evaluate(learner, cases),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(weights=result['weights'],
                          overall={name: dict(f1=row['f1'],
                                              candidate_mse=row['candidate_mse'])
                                   for name, row in result['evaluation']['overall'].items()},
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
