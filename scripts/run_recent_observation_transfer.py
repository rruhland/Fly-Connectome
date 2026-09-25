"""Opt-in test of local event recency beside persistent signed evidence."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from gap_timing_transfer import MODEL_OUT
from history_gated_code import DIRECTIONS
from run_observation_horizon_forecast import (evaluate, train_heads,
                                              train_recurrent)
from run_pong_camera_transfer import SEEDS, pong_events
from run_sparse_event_transfer import DIAGONALS, dots
from run_variable_cadence_transition import training_cases


OUT = Path('docs/experiments/2026-09-25-recent-observation-transfer-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    recurrent = train_recurrent(saved['models']['learned_split'].code.motion,
                                training_cases(), recent=True)
    sparse = dots(((10, 20), (14, 28), (16, 32), (18, 36),
                   (22, 44)), DIRECTIONS.values())
    heads = train_heads(recurrent, sparse, recent=True)
    groups = dict(heldout_cardinal=dots(((12, 24), (20, 40)),
                                      DIRECTIONS.values()),
                  heldout_diagonal=dots(((12, 24), (20, 40)), DIAGONALS),
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    result = dict(training_episodes=len(sparse),
                  groups={name: evaluate(recurrent, heads, cases, recent=True)
                          for name, cases in groups.items()},
                  elapsed_seconds=time.perf_counter()-start)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({group: {mode: {name: round(row['f1'], 3)
                                      for name, row in arms.items()}
                                for mode, arms in modes.items()}
                      for group, modes in result['groups'].items()}),
          flush=True)


if __name__ == '__main__':
    main()
