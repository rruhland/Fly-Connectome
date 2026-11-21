"""Test timing learned in table tennis against paired RGB fruit-catching games."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_timing import DelayedWiring, Disturbance, TimingMemory
from .tt7_transfer_catch import CatchPlayer, FruitCourt, RADIUS
from .tt7_transfer_timing import train_dynamics


PRACTICE = ROOT / 'artifacts/tt6'
EVIDENCE = ROOT / 'artifacts/tt7'
CONDITIONS = ('varied', 'ordinary', 'cold', 'uniform', 'varied_frozen')
DYNAMIC_CONDITIONS = ('varied', 'ordinary', 'cold', 'varied_prior', 'uniform_visual')


def learned_prior(receipt):
    """Pool source visits' camera-only latency estimates with a symmetric prior."""
    bins = np.arange(len(TimingMemory().probabilities))
    points = [point['timing'] for visit in receipt['visits'] for point in visit['points']]
    votes = [np.maximum(0., 1. - np.abs(bins - point['estimated']))
             for point in points if point['informative'] >= 3]
    counts = np.ones_like(bins, dtype=float) + np.sum(votes, axis=0)
    return dict(points=len(points), informative_points=len(votes), prior=TimingMemory(counts).state())


def schedule(seed, stations, rounds, *, occlusions=False):
    """Balance all delay pairs without consulting any donor or test outcome."""
    rng = np.random.default_rng(seed)
    pairs = np.array([(start, end, shade) for start in range(6) for end in range(6)
                      for shade in range(2 if occlusions else 1)])
    n = stations * rounds
    delays = np.concatenate([rng.permutation(pairs) for _ in range((n + len(pairs) - 1) // len(pairs))])
    entries = []
    for index, (first, later, shade) in enumerate(delays[:n]):
        fall, noise = rng.integers(0, 2**31 - 1, 2).tolist()
        switch = int(rng.integers(5, 12))
        entries.append(dict(fall=fall, noise=noise, delays=((0, int(first)), (switch, int(later))),
                            station=index // rounds, occlusion=bool(shade)))
    return entries


def match(specification):
    court = FruitCourt(occlusion=specification['occlusion'])
    change = Disturbance(delays=tuple(map(tuple, specification['delays'])), noise=((0, .04),))
    transport = DelayedWiring(court, change, seed=specification['noise'])
    return court, transport


def play(player, specification):
    court, transport = match(specification)
    rgb = transport.reset(seed=specification['fall'])
    player.reset(rgb)
    start_belief = player.memory.state()
    preceding, commands = None, []
    while True:
        preceding = player.act(rgb, preceding)
        rgb, reward, done, _, _ = transport.step(preceding)
        commands.append(float(preceding[0]))
        if done:
            break
    player.finish()
    return dict(catch=bool(reward), miss_distance=abs(court.basket - court.x), commands=commands,
                applied=[float(row[0]) for row in transport.applied], start_belief=start_belief,
                delay_estimates=player.observer.trajectory, informative=player.observer.informative)


def reachability(specification):
    """Calculate whether the evaluator can reach the fruit under the action cap."""
    endpoints = []
    for direction in (-1., 1.):
        court, transport = match(specification)
        transport.reset(seed=specification['fall'])
        for _ in range(court.flight):
            transport.step(np.array([direction]), render=False)
        endpoints.append(court.basket)
    return endpoints[0] - RADIUS <= court.x <= endpoints[1] + RADIUS


def verify(receipt):
    """Replay recorded public actions and check actuator commands and catch scores."""
    verified = 0
    for row in receipt['rows']:
        specification = row['specification']
        for name, tape in row['players'].items():
            court, transport = match(specification)
            transport.reset(seed=specification['fall'])
            reward = 0.
            for index, command in enumerate(tape['commands']):
                _, reward, done, _, _ = transport.step(np.array([command]), render=False)
                if done != (index == len(tape['commands']) - 1):
                    raise ValueError(f'catch time changed for {name} on fall {specification["fall"]}')
            if (bool(reward) != tape['catch'] or abs(court.basket - court.x) != tape['miss_distance']
                    or not np.array_equal(np.ravel(transport.applied), tape['applied'])):
                raise ValueError(f'physical replay diverged for {name} on fall {specification["fall"]}')
            verified += 1
    return verified


def summarize(rows, stations, conditions):
    result = {}
    estimates = {}
    for condition in conditions:
        trials = [row['players'][condition] for row in rows]
        catches = np.asarray([row['catch'] for row in trials], dtype=int).reshape(stations, -1)
        estimates[condition] = catches
        result[condition] = dict(catches=int(catches.sum()), first_three=int(catches[:, :3].sum()),
                                 catches_per_station=catches.sum(axis=1).tolist(),
                                 mean_miss_distance=float(np.mean([row['miss_distance'] for row in trials])))
        for boundary, label in enumerate(('by_start_delay', 'by_end_delay')):
            result[condition][label] = [sum(int(trial['catch']) for row, trial in zip(rows, trials, strict=True)
                                            if row['specification']['delays'][boundary][1] == delay)
                                        for delay in range(6)]
        result[condition]['by_visibility'] = {label: sum(int(trial['catch'])
                                                         for row, trial in zip(rows, trials, strict=True)
                                                         if row['specification']['occlusion'] == hidden)
                                              for label, hidden in [('open', False), ('occluded', True)]}
        errors = []
        for row, trial in zip(rows, trials, strict=True):
            (_, first), (switch, last) = row['specification']['delays']
            observed = np.asarray(trial['delay_estimates'])
            delays = np.where(np.arange(len(observed)) <= switch, first, last)
            errors.extend(np.abs(observed - delays))
        result[condition]['delay_mae_frames'] = float(np.mean(errors))
    groups = np.random.default_rng(2185819).integers(0, stations, (5000, stations))
    paired = {}
    for condition in conditions[1:]:
        delta = estimates['varied'] - estimates[condition]
        resamples = 100 * delta[groups].mean(axis=(1, 2))
        paired[condition] = dict(catch_difference=int(delta.sum()),
                                 first_three_difference=int(delta[:, :3].sum()),
                                 paired_mean_percentage_points=float(100 * delta.mean()),
                                 paired_interval_percentage_points=np.percentile(resamples, [2.5, 97.5]).tolist())
    return dict(by_condition=result, varied_comparisons=paired,
                reachable=sum(row['reachable'] for row in rows))


def player_group(donors, dynamics):
    varied, ordinary = donors['varied'], donors['ordinary']
    if dynamics:
        return dict(varied=CatchPlayer(varied['dynamics'], dynamics=True),
                    ordinary=CatchPlayer(ordinary['dynamics'], dynamics=True),
                    cold=CatchPlayer(dynamics=True), varied_prior=CatchPlayer(varied['prior']),
                    uniform_visual=CatchPlayer(np.ones(6)))
    return dict(varied=CatchPlayer(varied['prior']), ordinary=CatchPlayer(ordinary['prior']),
                cold=CatchPlayer(), uniform=CatchPlayer(np.ones(6)),
                varied_frozen=CatchPlayer(varied['prior'], learn=False))


def run(seed, stations, rounds, varied_path, ordinary_path, *, dynamics=False):
    donors = {}
    for name, path in [('varied', varied_path), ('ordinary', ordinary_path)]:
        receipt = _read(path)
        donors[name] = dict(**learned_prior(receipt), sha256=_hash(path))
        if dynamics:
            donors[name]['dynamics'] = train_dynamics(receipt)
    if donors['varied']['points'] != donors['ordinary']['points']:
        raise ValueError('the table-tennis donors must have equal practice budgets')
    rows = []
    for specification in schedule(seed, stations, rounds, occlusions=dynamics):
        if len(rows) % rounds == 0:
            players = player_group(donors, dynamics)
        results = {name: play(player, specification) for name, player in players.items()}
        rows.append(dict(specification=specification, players=results, reachable=reachability(specification)))
    return dict(schema='tt7-rgb-catch-transfer-v1', seed=seed, stations=stations, rounds_per_station=rounds,
                dynamics=dynamics, donors=donors, rows=rows,
                implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                       ('tt7_transfer_catch.py', 'tt7_transfer_timing.py',
                                        'tt7_transfer_trial.py', 'tt6_timing.py')},
                summary=summarize(rows, stations, DYNAMIC_CONDITIONS if dynamics else CONDITIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=30171967)
    parser.add_argument('--stations', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=18)
    parser.add_argument('--dynamics', action='store_true', help='fit optical transition models and vary basket visibility')
    parser.add_argument('--verify', action='store_true', help='replay and verify the existing receipt')
    parser.add_argument('--varied', type=Path, help='override the varied table-tennis source receipt')
    parser.add_argument('--ordinary', type=Path, help='override the ordinary table-tennis source receipt')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify(_read(args.output))} catch game action tapes.')
        return
    if args.stations < 1 or args.rounds < 1:
        parser.error('stations and rounds must be positive')
    varied = EVIDENCE / 'catch-source-varied.json.gz' if args.dynamics else PRACTICE / 'combined-carry.json.gz'
    ordinary = EVIDENCE / 'catch-source-ordinary.json.gz' if args.dynamics else PRACTICE / 'ordinary-practice.json.gz'
    report = run(args.seed, args.stations, args.rounds, args.varied or varied, args.ordinary or ordinary,
                 dynamics=args.dynamics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    for name, outcome in report['summary']['by_condition'].items():
        print(f'{name:14} {outcome["catches"]:3}/{len(report["rows"])} catches, '
              f'{outcome["first_three"]:2}/{args.stations * min(3, args.rounds)} early')


if __name__ == '__main__':
    main()
