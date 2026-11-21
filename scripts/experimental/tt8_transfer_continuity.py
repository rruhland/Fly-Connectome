"""Carry an unlabeled table-tennis belief into a screened catch session."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt7_surface_trial import DAMPING
from .tt8_transfer_model import RESTITUTION, Catcher
from .tt8_transfer_rebound import ReboundCourt
from .tt8_transfer_trial import play, verify


BOUNDARY_STRIDE = 2
SOURCE_NOISE = .075
SOURCE_CHANGE = .17
SESSIONS = 8
ROUNDS = 4


def observe_source(prior, mass, sightings):
    """Update one running belief from a single table point's RGB bounce fits."""
    mass = (1. - SOURCE_CHANGE) * mass + SOURCE_CHANGE * prior
    if sightings:
        error = (RESTITUTION - np.median(sightings)) / SOURCE_NOISE
        mass *= np.exp(-.5 * np.minimum(16., error**2))
    return mass / mass.sum()


def generic_controls():
    hard = Catcher(.94, .055, learn=False).probability
    soft = Catcher(.65, .070, learn=False).probability
    return dict(generic_mixture=.5 * (hard + soft),
                generic_hard=Catcher(.94, .025, learn=False, diffuse=.40).probability)


def snapshots(points, rng):
    """Take fixed chronological snapshots without reading source phase labels."""
    prior = generic_controls()['generic_mixture']
    belief = prior
    history = []
    for count, optical in enumerate(points, 1):
        history.append(optical)
        belief = observe_source(prior, belief, optical)
        if count % BOUNDARY_STRIDE:
            continue
        # The control gets the same evidence budget. Shuffle only the prefix
        # that has actually been observed by this handoff.
        permutation = rng.permutation(count)
        shuffled = prior
        for index in permutation:
            shuffled = observe_source(prior, shuffled, history[index])
        yield count, dict(chronological=belief, shuffled=shuffled), permutation.tolist()


def catcher(mass):
    player = Catcher(.75, .15, learn=True)
    player.log_probability = np.log(mass)
    return player


def summarize(rows):
    result = {}
    for label in ('all', 'persisted', 'changed'):
        cases = [row for row in rows if label == 'all' or row['continuity'] == label]
        result[label] = dict(falls=len(cases), first_falls=sum(row['round'] == 0 for row in cases), scores={})
        for name in cases[0]['players']:
            result[label]['scores'][name] = dict(
                catches=sum(r['players'][name]['catch'] for r in cases),
                first=sum(r['players'][name]['catch'] for r in cases if r['round'] == 0),
                first_two=sum(r['players'][name]['catch'] for r in cases if r['round'] <= 1),
                mean_miss_m=float(np.mean([r['players'][name]['miss_m'] for r in cases])),
            )
    return result


def run(source, visit, seed):
    practice = _read(source)['visits'][visit]
    points = practice['points']
    optical = [[sample['ratio'] for sample in row['table_bounce']['samples']] for row in points]
    optical_rng = np.random.default_rng([seed, 57915])
    target_rng = np.random.default_rng(seed)
    courts = {name: ReboundCourt(name) for name in DAMPING}
    rows = []
    handoffs = []
    for count, donors, order in snapshots(optical, optical_rng):
        material = points[count - 1]['phase']
        alternates = [name for name in courts if name != material]
        controls = generic_controls()
        handoffs.append(dict(count=count, optical_points=sum(bool(r) for r in optical[:count]),
                             shuffled_source_order=order, source_material=material,
                             priors={k: v.tolist() for k, v in donors.items()}))
        became_different = target_rng.permutation([False, True] * (SESSIONS // 2))
        for session, change in enumerate(became_different):
            target = str(target_rng.choice(alternates)) if change else material
            players = {name: catcher(mass) for name, mass in {**donors, **controls}.items()}
            for t, fall in enumerate(target_rng.integers(0, 2**31 - 1, size=ROUNDS).tolist()):
                court = courts[target]
                opening = court.opening(fall)
                tapes = {name: play(court, opening, player, visibility='curtain')
                         for name, player in players.items()}
                rows.append(dict(handoff=count, session=session, round=t, material=target,
                                 continuity='changed' if change else 'persisted', fall=fall, visibility='curtain',
                                 crossing_m=opening.endpoint, time_s=opening.crossing_time, players=tapes))
    return dict(schema='tt8-chronological-rebound-continuity-v1', seed=seed, visit=visit, body=practice['body'],
                source_sha256=_hash(source), handoffs=handoffs, rows=rows,
                implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                       ('tt4_arena.xml', 'tt7_surface_trial.py', 'tt8_transfer_rebound.py',
                                        'tt8_transfer_model.py', 'tt8_transfer_trial.py', 'tt8_transfer_continuity.py')},
                summary=summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt7/surface-read-only.json.gz')
    parser.add_argument('--seed', type=int, default=86337015)
    parser.add_argument('--visit', type=int, default=0)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        receipt = _read(args.output)
        print(f'Verified {verify(receipt)} rebound catch action tapes.')
        return
    if args.visit < 0:
        parser.error('use a nonnegative visit index')
    receipt = run(args.source, args.visit, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, receipt)
    print(receipt['summary'])


if __name__ == '__main__':
    main()
