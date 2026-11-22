"""Fit regularized return surfaces from physical stroke attempts."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_branch_motor import FEATURE_SCALE, OutcomeScorer
from .tt22_branch_motor_student import CHECKPOINT, HELD_COUNT, HELD_SCHEDULE, OUTPUT, record


WIDTH = .50
RIDGE = .06
DIAGNOSTIC = OUTPUT/'tt23-motor-surface-validation.json.gz'
HELD = OUTPUT/'tt23-motor-surface-held.json.gz'


class ReturnSurface(OutcomeScorer):
    """Fit each stroke's residual from its prior across previously seen approaches."""

    def __init__(self, corpus):
        super().__init__(corpus)
        self.approaches = self.context[self.later]
        observed = self.available[self.later].T
        outcomes = self.reward[self.later].T
        differences = self.approaches[:, None] - self.approaches
        kernel = np.exp(-.5*np.sum(differences**2, axis=-1)/WIDTH**2)
        # Only physically attempted strokes enter their fit. The same layout
        # lets NumPy solve all independent, small systems together.
        systems = kernel * observed[:, :, None] * observed[:, None, :]
        systems += RIDGE*np.eye(len(self.approaches))
        residual = (outcomes-self.priors[1][:, None])*observed
        self.fit = np.linalg.solve(systems, residual[:, :, None])[:, :, 0]

    def ranking(self, features, later, indices, errors):
        if not later:
            return super().ranking(features, later, indices, errors)
        kernel = np.exp(-.5*np.sum((self.approaches-features/FEATURE_SCALE)**2, axis=1)/WIDTH**2)
        return self.priors[1][indices] + self.fit[indices] @ kernel - .28*errors


def validate(checkpoint, output):
    saved = _read(checkpoint)
    rows = saved['contexts']
    donors = {row['seed'] for row in rows if row['leg'] > 0}
    results = []
    for donor in sorted(donors):
        train = [row for row in rows if row['seed'] != donor]
        models = {'local': OutcomeScorer(train), 'ridge': ReturnSurface(train)}
        for row in rows:
            if row['seed'] != donor or row['leg'] == 0:
                continue
            indices = np.asarray([trial[0] for trial in row['outcomes']])
            errors = np.zeros(len(indices))
            decisions = {}
            for name, model in models.items():
                index = int(np.argmax(model.ranking(row['features'], True, indices, errors)))
                motor, legal, _, _ = row['outcomes'][index]
                decisions[name] = dict(motor=motor, legal=legal)
            results.append(dict(seed=donor, tick=row['tick'], leg=row['leg'], decisions=decisions))
    summary = {name: sum(row['decisions'][name]['legal'] for row in results) for name in ('local', 'ridge')}
    _write(output, dict(schema='tt23-motor-surface-validation-1', practice=checkpoint.name,
                        practice_hash=_hash(checkpoint), width=WIDTH, ridge=RIDGE,
                        held_donors=len(donors), rows=results, summary=summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--diagnostic', type=Path, default=DIAGNOSTIC)
    parser.add_argument('--output', type=Path, default=HELD)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--game', type=Path)
    args = parser.parse_args()
    if args.game is not None:
        from .tt23_motor_game import run

        run(args.checkpoint, args.game, scorer=ReturnSurface, policy='ridge')
        return
    print(validate(args.checkpoint, args.diagnostic), flush=True)
    if not args.validate_only:
        record(args.checkpoint, args.output, HELD_COUNT, HELD_SCHEDULE, scorer=ReturnSurface, policy='ridge')


if __name__ == '__main__':
    main()
