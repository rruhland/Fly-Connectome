"""Locally learn the event code on diverse generic fractional motion."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from generic_native_cadence import scene_events
from learned_transition_units import TransitionPopulation
from run_event_forecast_ranking import empty as empty_rank, rank_episode
from run_native_trace_support import trace_correlation_sequence
from run_observation_horizon_forecast import (evaluate, train_heads,
                                              train_recurrent)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-native-cadence-encoder-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    heldout = [scene_events(1000+seed, heldout=True)
               for seed in range(16)]
    native = [pong_events(seed, stride=1, frames=120) for seed in SEEDS]
    motion = TransitionPopulation(channels=16, seed=0)
    for events in training:
        motion.reset_state()
        for code in trace_correlation_sequence(events):
            motion.step(code, learn_dictionary=True)
    dictionary_seconds = time.perf_counter()-start
    print(f'generic local event code trained in {dictionary_seconds:.1f}s',
          flush=True)
    recurrent = train_recurrent(motion, training)
    heads = train_heads(recurrent, training)
    groups = dict(generic_heldout=heldout, native=native)
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: evaluate(recurrent, heads, cases)
              for name, cases in groups.items()}
    ranks = {}
    for name in ('generic_heldout', 'native'):
        row = empty_rank()
        for events in groups[name]:
            rank_episode(recurrent, heads['next_event']['learned'],
                         events, row)
        ranks[name] = {**row,
                       'top_8_recall': row['top_8_hits']/max(row['targets'], 1),
                       'top_32_recall': row['top_32_hits']/max(row['targets'], 1)}
    result = dict(training_episodes=len(training),
                  heldout_episodes=len(heldout),
                  dictionary_updates=motion.dictionary_updates,
                  dictionary_seconds=dictionary_seconds,
                  scores=scores, ranks=ranks,
                  elapsed_seconds=time.perf_counter()-start)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(scores={group: {mode: {
        name: round(row['f1'], 3) for name, row in arms.items()}
        for mode, arms in modes.items()} for group, modes in scores.items()},
        ranks={name: round(row['top_8_recall'], 3)
               for name, row in ranks.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
