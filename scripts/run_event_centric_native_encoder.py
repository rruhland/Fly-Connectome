"""Matched event-centric forecast with a native-cadence learned code."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from learned_transition_units import TransitionPopulation
from run_event_budget_probe import probe
from run_event_centric_competition import rank_fixed, train_heads, train_state
from run_native_trace_support import trace_correlation_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-event-centric-native-encoder-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    motion = TransitionPopulation(channels=16, seed=0)
    for events in training:
        motion.reset_state()
        for code in trace_correlation_sequence(events):
            motion.step(code, learn_dictionary=True)
    model = train_state(motion, training)
    heads = train_heads(model, training)['four_frame']
    trained_seconds = time.perf_counter()-started
    print(f'native-cadence code and local heads trained in '
          f'{trained_seconds:.1f}s', flush=True)
    calibration = probe(model, heads, training)
    gains = {name: row['target_events']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    scores = {group: probe(model, heads, cases, gains=gains)
              for group, cases in groups.items()}
    ranks = {group: {name: rank_fixed(model, head, cases)
                     for name, head in heads.items()}
             for group, cases in groups.items()}
    result = dict(training_episodes=len(training),
                  dictionary_updates=motion.dictionary_updates,
                  trained_seconds=trained_seconds,
                  gains=gains, scores=scores, ranks=ranks,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        scores={group: {name: round(row['score']['f1'], 3)
                        for name, row in arms.items()}
                for group, arms in scores.items()},
        rank={group: {name: round(row['top_8_recall'], 3)
                     for name, row in arms.items()}
              for group, arms in ranks.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
