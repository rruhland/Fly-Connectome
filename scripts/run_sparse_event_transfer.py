"""Test whether sparse generic event training closes the native transfer gap."""

import json
import time
from pathlib import Path

import torch

from gap_timing_transfer import MODEL_OUT
from history_gated_code import DIRECTIONS
from run_observation_horizon_forecast import (evaluate, train_heads,
                                              train_recurrent)
from run_pong_camera_transfer import SEEDS, pong_events
from run_variable_cadence_transition import continuous_events, training_cases


OUT = Path('docs/experiments/2026-09-25-sparse-event-transfer-results.json')
SPEEDS = (.25, .5, 1, 2)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def dots(centers, directions):
    return [continuous_events('dot', center, direction, speed, background)
            for center in centers for direction in directions
            for speed in SPEEDS for background in (False, True)]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    recurrent = train_recurrent(saved['models']['learned_split'].code.motion,
                                training_cases())
    sparse_training = dots(((10, 20), (14, 28), (16, 32), (18, 36),
                            (22, 44)), DIRECTIONS.values())
    heads = train_heads(recurrent, sparse_training)
    heldout_cardinal = dots(((12, 24), (20, 40)), DIRECTIONS.values())
    heldout_diagonal = dots(((12, 24), (20, 40)), DIAGONALS)
    native = [pong_events(seed, stride=1, frames=120) for seed in SEEDS]
    result = dict(training_episodes=len(sparse_training),
                  heldout_cardinal=evaluate(recurrent, heads,
                                            heldout_cardinal),
                  heldout_diagonal=evaluate(recurrent, heads,
                                            heldout_diagonal),
                  native=evaluate(recurrent, heads, native),
                  elapsed_seconds=time.perf_counter()-start)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({domain: {mode: {name: round(row['f1'], 3)
                                       for name, row in arms.items()}
                                 for mode, arms in scores.items()}
                      for domain, scores in result.items()
                      if isinstance(scores, dict)}), flush=True)


if __name__ == '__main__':
    main()
