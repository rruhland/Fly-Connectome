"""Run paired stock-arm and alternate-arm controls for the paddle benchmark."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .tt24_bounce import BounceArena, MIN_AIRBORNE, MIN_RISE, mujoco


PRACTICE_SEEDS = tuple(range(240_001, 240_049))
HELD_SEEDS = tuple(range(241_001, 241_049))
RANDOM_HOLD_STEPS = 4


def _run(arena, seed, split, regime, random_rates):
    arena.reset(seed=seed, split=split, regime=regime)
    rng = np.random.default_rng(np.random.SeedSequence((31_415, seed)))
    command = np.zeros(arena.action_size)
    for step in range(arena.max_steps):
        if random_rates and step % RANDOM_HOLD_STEPS == 0:
            command = rng.uniform(-1., 1., arena.action_size)
        _, _, done = arena.step(command)
        if done:
            break
    return arena.judge_metrics()


def _summarize(results):
    rebounds = [row['rebounds'] for row in results]
    arcs = [arc for row in results for arc in row['rebound_events']]
    return dict(episodes=len(results), mean_rebounds=float(np.mean(rebounds)),
                max_rebounds=max(rebounds), at_least_five=sum(count >= 5 for count in rebounds),
                max_consecutive_rebounds=max(row['longest_consecutive_rebounds'] for row in results),
                mean_elapsed_s=float(np.mean([row['elapsed_s'] for row in results])),
                rebound_histogram=dict(sorted(Counter(rebounds).items())),
                terminations=dict(Counter(row['termination'] for row in results)),
                mean_rise_m=float(np.mean([arc['rise_m'] for arc in arcs])) if arcs else None,
                min_time_to_apex_s=min((arc['time_to_apex_s'] for arc in arcs), default=None))


def _fixture_image(path):
    width, height = 224, 168
    arena = BounceArena(width=width, height=height)
    observation = arena.reset(seed=0)
    frames = [('Release', 0., observation['rgb'], arena.render_external())]
    labels = iter(('First contact', 'First apex', 'Second contact'))
    previous_contacts = 0
    try:
        for _ in range(arena.max_steps):
            observation, reward, done = arena.step(np.zeros(arena.action_size))
            result = arena.judge_metrics()
            if result['paddle_contacts'] > previous_contacts or reward:
                frames.append((next(labels), result['elapsed_s'], observation['rgb'], arena.render_external()))
                previous_contacts = result['paddle_contacts']
            if done or len(frames) == 4:
                break
    finally:
        arena.close()

    board = Image.new('RGB', (len(frames) * width, 2 * height + 43), '#11191f')
    draw = ImageDraw.Draw(board)
    draw.text((8, 23 + height), 'External view', fill='#eff2f3')
    for index, (caption, time, wrist, external) in enumerate(frames):
        left = index * width
        draw.text((left + 8, 4), f'{caption}    {time:.3f} s', fill='#eff2f3')
        board.paste(Image.fromarray(wrist), (left, 21))
        board.paste(Image.fromarray(external), (left, 21 + height + 22))
    board.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    comparisons = {}
    stock = BounceArena(view='none')
    alternate = BounceArena(view='none', arm='alternate')
    for arena, split, regime, seeds in (
        (stock, 'practice', 'standard', PRACTICE_SEEDS),
        (stock, 'held', 'standard', HELD_SEEDS),
        (stock, 'held', 'changed', HELD_SEEDS),
        (alternate, 'held', 'standard', HELD_SEEDS),
    ):
        methods = {}
        for test in ('stationary', 'random'):
            rows = [_run(arena, seed, split, regime, test == 'random') for seed in seeds]
            methods[test] = dict(summary=_summarize(rows), episodes=rows)
        comparisons[f'{arena.arm}/{split}/{regime}'] = methods

    record = dict(mujoco=mujoco.__version__, action_axes=BounceArena.action_axes, control_hz=60,
                  min_rise_m=MIN_RISE, min_airborne_s=MIN_AIRBORNE,
                  random_hold_steps=RANDOM_HOLD_STEPS, comparisons=comparisons)
    metrics = args.out / 'bounce-baselines.json.gz'
    with gzip.open(metrics, 'wt', encoding='utf-8') as output:
        json.dump(record, output, indent=2)
    _fixture_image(args.out / 'bounce-fixture.png')
    print(json.dumps({key: {mode: row['summary'] for mode, row in value.items()}
                      for key, value in comparisons.items()}, indent=2))


if __name__ == '__main__':
    main()
