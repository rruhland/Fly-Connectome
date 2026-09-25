"""Opt-in online local-learning test with balanced interrupted motion."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import TRAIN_SHAPES, unseen_shape_sequences
from history_gated_code import interrupted_cases
from history_gated_prediction import expanded_interruption_cases, matched_pair
from polarity_local_prediction import evaluate, train_candidates
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-25-balanced-interruption-exposure-results.json')


def make_balanced_cases():
    original = make_training_cases()
    added = [('interruption', shape, events)
             for shape, _, _, _, events in interrupted_cases(TRAIN_SHAPES)]
    return original, added


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original, added = make_balanced_cases()
    archived, models = train_candidates(original+added)
    added_scores = evaluate(archived, models, added)
    heldout = [('single', shape, events)
               for shape, _, _, _, events in unseen_shape_sequences()]
    heldout += [(family, 'unseen', events)
                for family, _, events in make_robust_cases()]
    heldout_scores = evaluate(archived, models, heldout)
    expanded_scores = evaluate(archived, models,
                               expanded_interruption_cases())
    pair = matched_pair(archived, models)
    result = dict(original_episodes=len(original), added_episodes=len(added),
                  added=added_scores, heldout=heldout_scores,
                  expanded=expanded_scores, matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        added_exact={name: round(row['f1'], 3)
                     for name, row in added_scores['exact'].items()},
        single={name: round(row['f1'], 3)
                for name, row in heldout_scores['groups']['single'].items()},
        expanded_exact={name: round(row['f1'], 3)
                        for name, row in expanded_scores['exact'].items()},
        expanded_by_direction={family: {name: dict(
            f1=round(row['f1'], 3), tp=row['tp'])
            for name, row in rows.items()}
            for family, rows in expanded_scores['exact_by_family'].items()},
        quiet=heldout_scores['quiet'],
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
