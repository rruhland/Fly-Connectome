"""Replay a fixed low-budget view of recorded strict branch outcomes."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_branch_motor import OutcomeScorer
from .tt22_branch_motor_trial import OUTPUT

COUNT = 8
SCHEDULE = 22968193
SOURCE = OUTPUT/'branch-motor-practice.json.gz'
CHECKPOINT = OUTPUT/'branch-motor-sparse-practice.json.gz'
RECEIPT = OUTPUT/'branch-motor-sparse-receipt.json.gz'


def sample(rows, seed):
    """Choose eligible motor IDs from a blind permutation before revealing outcomes."""
    sparse = []
    for row in rows:
        trials = row['outcomes']
        random = np.random.default_rng([seed, row['seed'], row['tick']])
        slots = random.choice(len(trials), min(COUNT, len(trials)), replace=False)
        sparse.append({key: value for key, value in row.items() if key != 'physical_frames'}
                      | {'outcomes': [trials[i] for i in slots]})
    return sparse


def leave_donor_out(training, validation):
    counts = dict(contextual=0, marginal=0)
    for row in validation:
        others = [r for r in training if r['seed'] != row['seed']]
        truth = {trial[0]: trial[1] for trial in row['outcomes']}
        eligible = np.array(list(truth))
        features, alignment = np.asarray(row['features']), np.zeros(len(eligible))
        for name in counts:
            scorer = OutcomeScorer(others, contextual=name=='contextual')
            scores = scorer.ranking(features, True, eligible, alignment)
            counts[name] += int(truth[eligible[np.argmax(scores)]])
    return counts


def main():
    base = _read(SOURCE)
    opening = [row for row in base['contexts'] if row['leg'] == 0]
    later = [row for row in base['contexts'] if row['leg'] > 0]
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=COUNT).tolist()
    receipt = dict(schema='tt22-branch-motor-sparse-1', source=SOURCE.name, source_hash=_hash(SOURCE),
                   schedule=SCHEDULE, fixed_checkpoint_seed=seeds[0], budget_per_later_cue=COUNT,
                   candidates=len(later), full=leave_donor_out(later, later), repeats=[])
    for seed in seeds:
        partial = sample(later, seed)
        count = sum(len(row['outcomes']) for row in partial)
        if seed == seeds[0]:
            checkpoint = dict(base, schema='tt22-branch-practice-sparse-1', sparse_seed=seed,
                              dense_source_hash=receipt['source_hash'], contexts=opening+partial,
                              later_outcomes=count)
            _write(CHECKPOINT, checkpoint)
            receipt['checkpoint_hash'] = _hash(CHECKPOINT)
        scores = leave_donor_out(partial, later)
        receipt['repeats'].append(dict(seed=seed, later_outcomes=count, **scores))
    _write(RECEIPT, receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
