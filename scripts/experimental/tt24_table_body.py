"""Carry motor exploration statistics from physical table tennis to paddle bouncing."""

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import play, search, seeds, verify_replay
from .tt24_motor import MotorEffects


DIR = Path('artifacts/tt24')
OUTPUT = DIR / 'table-body.json.gz'
SOURCE = (('wrist-match-held.json.gz', 'wrist'), ('rally-bank-held.json.gz', 'broad'),
          ('rally-improve-combined-held.json.gz', 'combined'))
STUDY = 261101
ASSESS = 261102
RNG_SEED = 261103
PRACTICE = 24
HELD = 16
CONTEXTS = 2
CONTEXT_REPEATS = 2


def return_credit(tape):
    credit = np.zeros(len(tape['actions']))
    # A direct return pays after the ball has travelled, so retain a broad
    # eligibility window over the commands the learner actually issued.
    for event in tape['events']:
        if event['kind'] != 'legal_return' or event['side'] != 1:
            continue
        reward = event['step']
        beginning = max(0, reward - 30)
        indices = np.arange(beginning, min(len(credit), reward))
        delay = reward - indices
        credit[indices] += np.exp(-.5 * ((delay - 16.) / 7.)**2)
    return credit


def source_covariance():
    total = np.zeros((5, 5))
    receipts = []
    for name, mode in SOURCE:
        path = DIR / name
        raw = path.read_bytes()
        source = json.loads(gzip.decompress(raw))
        verified = sum(len(row['results']) for row in source['cases'])
        if source['verified_tapes'] != verified:
            raise ValueError(f'physical game archive {name} was not completely verified')
        ticks, turns, eligible = 0, 0, 0
        for case in source['cases']:
            tape = case['results'][mode]
            action = np.asarray(tape['actions'])[:, :5]
            credit = return_credit(tape)
            total += action.T @ (action * credit[:, None])
            ticks += len(action)
            turns += tape['outcome']['legal_returns']
            eligible += int(np.count_nonzero(credit))
        receipts.append(dict(file=name, modality=mode, sha256=sha256(raw).hexdigest(), axes=5,
                             physical_games=len(source['cases']), ticks=ticks, legal_returns=turns,
                             credit_windows=eligible, verified=verified))

    covariance = .32 * np.eye(5) + .68 * total / (np.trace(total) / 5.)
    values, vector = np.linalg.eigh(covariance)
    prior = (vector * np.sqrt(values)) @ vector.T
    rng = np.random.default_rng(RNG_SEED + 823)
    rotation, signs = np.linalg.qr(rng.normal(size=(5, 5)))
    rotation = rotation * np.sign(np.diag(signs))
    shuffled = rotation @ prior @ rotation.T
    return dict(table=prior, shuffled=shuffled, cold=np.eye(5)), receipts, values


def summary(rows):
    longest = np.array([row['longest_consecutive_rebounds'] for row in rows])
    rebounds = np.array([row['rebounds'] for row in rows])
    return dict(rebounds=rebounds.tolist(), consecutive=longest.tolist(), mean=float(rebounds.mean()),
                mean_consecutive=float(longest.mean()), maximum=int(longest.max()),
                at_least_five=int(np.sum(longest >= 5)), at_least_ten=int(np.sum(longest >= 10)))


def assess(arena, held, weights, effects, *, control=None):
    rows = [play(arena, start, weights, deepcopy(effects), split='held', keep_actions=True,
                 hold=control == 'hold', blind=control is not None, readout='rate', stride=2, retinal='relative')
            for start in held]
    for row in rows:
        verify_replay(arena, row)
    return rows


def store(path, result):
    with gzip.open(path, 'wt') as stream:
        json.dump(result, stream, separators=(',', ':'))


def run(output):
    modes, receipt, spectrum = source_covariance()
    contexts = seeds(STUDY, CONTEXTS)
    practice = contexts * (PRACTICE // len(contexts))
    held = seeds(ASSESS, HELD)
    result = dict(schema='tt24-table-body-1', source_type='physical table tennis', sources=receipt,
                  covariance_spectrum=spectrum.tolist(), modes={key: val.tolist() for key, val in modes.items()},
                  practice=practice, held_starts=held, training={}, held={}, summary={},
                  target_demonstrations=0, target_reward='scalar rebound with .24 capped apex bonus')
    print('physical table-tennis source', receipt, 'spectrum', spectrum.tolist(), flush=True)
    arena = BounceArena(view='wrist', width=96, height=72, max_steps=300, apex_reward_weight=.24)
    try:
        for condition, mode in modes.items():
            effects = MotorEffects(arena.action_size)
            weights, training = search(arena, practice, None, effects, seed=RNG_SEED, radius=.37,
                                       log=True, stride=2, readout='rate', retinal='relative', repeats=CONTEXT_REPEATS,
                                       motor_modes=mode)
            rows = assess(arena, held, weights, effects)
            result['training'][condition] = dict(rows=training, weights=weights.tolist(),
                                                 effects=effects.readout.tolist(), samples=effects.samples)
            result['held'][condition] = rows
            result['summary'][condition] = summary(rows)
            print('table motor held', condition, result['summary'][condition], flush=True)
            store(output, result)
            if condition == 'table':
                for control in ('blind', 'hold'):
                    rows = assess(arena, held, weights, effects, control=control)
                    result['held'][control] = rows
                    result['summary'][control] = summary(rows)
                    print('table motor held', control, result['summary'][control], flush=True)
        result['verified_physical_replays'] = sum(len(rows) for rows in result['held'].values())
        store(output, result)
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
