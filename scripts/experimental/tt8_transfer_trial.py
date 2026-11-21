"""Test whether optical table-tennis rebound experience improves RGB catch scores."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_transfer_model import Catcher, optical_prior
from .tt8_transfer_rebound import DT, MOUTH, ReboundCourt, steer


MATERIALS = ('lively', 'low')
CONDITIONS = ('matched', 'mismatched', 'target_hard', 'target_only', 'matched_frozen',
              'fixed_generic', 'fixed_regulation', 'fixed_hard')
EARLY_FRAMES = 8


def player_group(donors, material):
    matched = donors[material]
    other = donors[next(name for name in MATERIALS if name != material)]
    return dict(matched=Catcher(matched['center'], matched['spread'], learn=True),
                mismatched=Catcher(other['center'], other['spread'], learn=True),
                target_hard=Catcher(.94, .025, learn=True, diffuse=.40),
                target_only=Catcher(.75, .15, learn=True),
                matched_frozen=Catcher(matched['center'], matched['spread'], learn=False),
                fixed_generic=Catcher(.75, .15, learn=False),
                fixed_regulation=Catcher(.86, .035, learn=False),
                fixed_hard=Catcher(.94, .025, learn=False))


def play(court, opening, player, *, visibility):
    player.reset()
    cart = opening.cart_start
    times = len(opening.positions) - 1
    commands = []
    initial = player.restitution
    for t, ball in enumerate(opening.positions[:-1]):
        visible = visibility == 'open' or t < EARLY_FRAMES
        rgb = court.frame(ball if visible else None, cart)
        action = player.act(rgb)
        commands.append(action)
        elapsed = opening.crossing_time - t * DT if t == times - 1 else DT
        cart = steer(cart, action, elapsed)
    distance = abs(cart - opening.endpoint)
    player.finish(court.frame(opening.positions[-1], cart))
    return dict(catch=distance <= MOUTH, miss_m=distance, commands=commands,
                initial_ratio=initial, final_ratio=player.restitution)


def _summary(rows, stations, rounds):
    scores = {}
    cells = {}
    for visibility in ('curtain', 'open'):
        selected = [row for row in rows if row['visibility'] == visibility]
        cells[visibility] = {}
        scores[visibility] = {}
        for condition in CONDITIONS:
            end = np.array([row['players'][condition]['catch'] for row in selected], dtype=int)
            matrix = end.reshape(stations, len(MATERIALS), rounds)
            dist = [row['players'][condition]['miss_m'] for row in selected]
            scores[visibility][condition] = matrix
            cells[visibility][condition] = dict(
                catches=int(end.sum()), mean_absolute_miss_m=float(np.mean(dist)),
                first_three=int(matrix[:, :, :3].sum()), first=int(matrix[:, :, 0].sum()),
                by_material={name: int(matrix[:, i].sum()) for i, name in enumerate(MATERIALS)})
    rng = np.random.default_rng(49158701)
    bootstrap = rng.integers(0, stations, (5000, stations))
    comparisons = {}
    for visibility, group in scores.items():
        comparisons[visibility] = {}
        for name in CONDITIONS[1:]:
            difference = group['matched'] - group[name]
            first_three = difference[:, :, :3]
            comparisons[visibility][name] = dict(
                catch_difference=int(difference.sum()),
                first_three_difference=int(first_three.sum()),
                interval_pp=np.percentile(100 * difference[bootstrap].mean(axis=(1, 2, 3)), [2.5, 97.5]).tolist(),
                first_three_interval_pp=np.percentile(100 * first_three[bootstrap].mean(axis=(1, 2, 3)),
                                                       [2.5, 97.5]).tolist())
    return dict(scores=cells, matched_comparisons=comparisons,
                reachable=sum(row['reachable'] for row in rows),
                catches_per_condition=stations * len(MATERIALS) * rounds)


def run(source, seed, stations, rounds):
    receipt = _read(source)
    donors = {name: optical_prior(receipt, name) for name in MATERIALS}
    rng = np.random.default_rng(seed)
    rows = []
    courts = {name: ReboundCourt(name) for name in MATERIALS}
    for station, seeds in enumerate(rng.integers(0, 2**31 - 1, size=(stations, rounds)).tolist()):
        for material, court in courts.items():
            groups = {name: player_group(donors, material) for name in ('open', 'curtain')}
            for fall in seeds:
                opening = court.opening(fall)
                left = steer(opening.cart_start, -1., opening.crossing_time)
                right = steer(opening.cart_start, 1., opening.crossing_time)
                reachable = left - MOUTH <= opening.endpoint <= right + MOUTH
                for visibility, players in groups.items():
                    tapes = {name: play(court, opening, player, visibility=visibility)
                             for name, player in players.items()}
                    rows.append(dict(
                        station=station, fall=fall, material=material, visibility=visibility,
                        crossing_m=opening.endpoint, time_s=opening.crossing_time,
                        reachable=reachable, players=tapes))
    return dict(schema='tt8-rgb-bank-catch-v1', seed=seed, stations=stations, rounds=rounds,
                source_sha256=_hash(source), donors=donors, rows=rows,
                implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                       ('tt4_arena.xml', 'tt7_surface_trial.py', 'tt8_transfer_rebound.py',
                                        'tt8_transfer_model.py', 'tt8_transfer_trial.py')},
                summary=_summary(rows, stations, rounds))


def verify(receipt):
    """Replay the applied cart commands through the independent physical scoring rule."""
    courts = {name: ReboundCourt(name) for name in MATERIALS}
    checked = 0
    for row in receipt['rows']:
        opening = courts[row['material']].opening(row['fall'])
        if (row['crossing_m'], row['time_s']) != (opening.endpoint, opening.crossing_time):
            raise ValueError(f'ball replay diverged for {row["fall"]}')
        for name, result in row['players'].items():
            cart = opening.cart_start
            commands = result['commands']
            if len(commands) + 1 != len(opening.positions) or max(map(abs, commands)) > 1.:
                raise ValueError(f'invalid commands for {name} on {row["fall"]}')
            for t, command in enumerate(commands):
                elapsed = opening.crossing_time - t * DT if t == len(commands) - 1 else DT
                cart = steer(cart, command, elapsed)
            distance = abs(cart - opening.endpoint)
            if (distance, distance <= MOUTH) != (result['miss_m'], result['catch']):
                raise ValueError(f'cart replay diverged for {name} on {row["fall"]}')
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt7/surface-read-only.json.gz')
    parser.add_argument('--seed', type=int, default=56438819)
    parser.add_argument('--stations', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify(_read(args.output))} rebound catch action tapes.')
        return
    if min(args.stations, args.rounds) < 1:
        parser.error('stations and rounds must be positive')
    receipt = run(args.source, args.seed, args.stations, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, receipt)
    for visibility, scores in receipt['summary']['scores'].items():
        print(visibility, receipt['summary']['catches_per_condition'], 'falls')
        for name, values in scores.items():
            print(f'  {name:17} {values["catches"]:3} catches, {values["first_three"]:3} early, '
                  f'{values["mean_absolute_miss_m"]:.3f} m mean absolute miss')


if __name__ == '__main__':
    main()
