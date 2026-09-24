"""Bounded extra-experience check for the same opt-in visual circuit."""

import json
import time
from pathlib import Path

import torch

from diverse_visual_experience import (diverse_training_sequences, train_arm,
                                       unseen_shape_sequences)


OUT = Path('docs/experiments/2026-09-24-expanded-visual-exposure-results.json')
BASELINE = Path('docs/experiments/2026-09-24-diverse-visual-experience-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    baseline = json.loads(BASELINE.read_text())['diverse']
    expanded = train_arm(diverse_training_sequences(epochs=12),
                         unseen_shape_sequences())
    result = dict(baseline=baseline, expanded=expanded,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: dict(
        episodes=arm['training_episodes'],
        code_probe=arm['code_probe']['accuracy'],
        local_event_f1=arm['local_prediction']['overall']['learned']['event']['f1'],
        local_train_f1=arm['local_training']['event']['f1'],
        capacity_event_f1=arm['capacity']['overall']['learned']['f1'],
        capacity_train_f1=arm['capacity_training']['learned']['f1'],
        random_capacity_f1=arm['capacity']['overall']['random_dictionary']['f1'])
        for name, arm in (('baseline', baseline), ('expanded', expanded))}),
        flush=True)


if __name__ == '__main__':
    main()
