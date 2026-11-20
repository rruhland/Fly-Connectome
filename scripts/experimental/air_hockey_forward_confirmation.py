"""Confirm whether a Pong-derived wall estimate changes actual air-hockey scoring."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

from air_hockey_experiment import (
    COLD_COEFFICIENT, ROOT, finish_round, paired, read, sha, summarize, write,
)
from air_hockey_learning import GoalMemory
from arcade_wall_learn import WallMemory


STYLES = ('bank', 'fast-bank', 'direct', 'neutral')
ENTROPY = 5071601703


def confirm(path, output, *, count, workers):
    state = read(path)
    prior = read(path.with_name('registration.json'))
    original = prior['seeds']
    used = set(original['pong'])
    used.update(round_['seed'] for kind in ('hockey_practice', 'hockey_heldout') for round_ in original[kind])
    rng = random.Random(ENTROPY)
    schedule = []
    for style in STYLES:
        for _ in range(count):
            seed = rng.randrange(1, 2**31 - 1)
            while seed in used:
                seed = rng.randrange(1, 2**31 - 1)
            used.add(seed)
            schedule.append(dict(seed=seed, style=style, opponent='laneguard'))
    files = [Path(__file__), *(ROOT / name for name in prior['hashes'])]
    output.mkdir(parents=True, exist_ok=False)
    registration = {'checkpoint_sha256': sha(path), 'schedule': schedule, 'seed': ENTROPY,
                    'component_sha256': {str(file.relative_to(ROOT)): sha(file) for file in files},
                    'primary': 'Pong forecaster minus broad reset, paired last-touch goals on all starts',
                    'physical_control': 'Same controller with elastic e=1 forecast prior',
                    'agent': 'Fixed center aim and frozen policy in all arms'}
    write(output / 'registration.json', registration)
    blank = GoalMemory().to_state()
    controls = {'pong_center': state['wall'],
                'broad_center': WallMemory(default=COLD_COEFFICIENT).to_state(),
                'elastic_center': WallMemory(default=1.).to_state()}
    jobs = [(label, case, wall, blank) for case in schedule for label, wall in controls.items()]
    groups = {label: [] for label in controls}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label, row in pool.map(finish_round, jobs, chunksize=4):
            del row['events'], row['visual_bounces'], row['visual_shots']
            groups[label].append(row)
    first = [row['initial_sha256'] for row in groups['pong_center']]
    if any(first != [row['initial_sha256'] for row in rows] for rows in groups.values()):
        raise AssertionError('matched starts differ')
    summary = {}
    for style in ('all', *STYLES):
        selected = {arm: [row for row in rows if style == 'all' or row['style'] == style]
                    for arm, rows in groups.items()}
        summary[style] = dict(arms={arm: summarize(rows) for arm, rows in selected.items()},
                              paired={arm: {metric: paired(selected['pong_center'], selected[arm], metric)
                                            for metric in ('last_touch_goal', 'goal_after_own_hit',
                                                           'opening_return')}
                                      for arm in ('broad_center', 'elastic_center')})
    write(output / 'results.json', dict(summary=summary, rounds=groups,
                                        registration_sha256=sha(output / 'registration.json')))
    for style, entry in summary.items():
        print(style, {arm: data['last_touch_goals'] for arm, data in entry['arms'].items()}, entry['paired'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rounds-per-style', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.rounds_per_style < 1 or args.workers < 1:
        parser.error('positive round count and worker count required')
    confirm(args.checkpoint, args.output, count=args.rounds_per_style, workers=args.workers)


if __name__ == '__main__':
    main()
