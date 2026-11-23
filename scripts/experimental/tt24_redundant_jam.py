"""Test six-joint redundancy after mechanically jamming the legacy pitch axis."""

import argparse
from contextlib import ExitStack, closing
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt24_bounce_trial import _write, seeds, verify_replay
from .tt24_redundant import MarkerBounce, RedundantBounce
from .tt24_redundant_reach import move_for, practice_goal, reach
from .tt24_redundant_trial import POLICY, aggregate, grounding, measure, play


ROOT = Path('artifacts/tt24')
SOURCE = ROOT / 'redundant-game.json.gz'
OUTPUT = ROOT / 'redundant-jam.json.gz'
LAYOUT = {'normal': {}, 'jammed': {'jammed': True}, 'both_jammed': {'jammed': True, 'pinned': True}}


def describe(episodes):
    summary = aggregate(episodes)
    travel = [row['max_encoder_travel'][3] for row in episodes]
    fractions = [row['sixth_stop_fraction'] for row in episodes]
    return summary | dict(original_pitch_max_rad=float(np.max(travel)),
                          original_pitch_median_rad=float(np.median(travel)),
                          sixth_mean_stop_fraction=float(np.mean(fractions)))


def motion(stack, source, body, grounding_maps, seed, count):
    flags = dict(view='none', width=192, height=144, calibration=True)
    stock = stack.enter_context(closing(MarkerBounce(**flags)))
    target = {name: stack.enter_context(closing(RedundantBounce(**flags, **extra))) for name, extra in LAYOUT.items()}
    rng = np.random.default_rng(seed + 11)
    rows = []
    for start in seeds(seed, count):
        goal = practice_goal(stock, start, rng)
        if goal is None:
            rows.append(dict(seed=start, missing_stock=True))
            continue
        donor = move_for(source['command'], goal)
        request = donor @ source['command']
        choices = {name: (target[name], move_for(body[name]['command'], request)) for name in LAYOUT}
        choices['jammed_padded'] = target['jammed'], np.r_[donor, 0.]
        naive = donor @ grounding_maps['normal']['optical_transmit']
        choices['jammed_frozen'] = target['jammed'], naive / max(1., np.max(np.abs(naive)) / .61)
        choices['jammed_stationary'] = target['jammed'], np.zeros(6)
        outcomes = {name: reach(arena, start, action, goal) for name, (arena, action) in choices.items()}
        rows.append(dict(seed=start, goal=goal.tolist(), result=outcomes))
    paired = [row for row in rows if 'result' in row and all(v['rms_pixels'] is not None for v in row['result'].values())]
    if not paired:
        raise ValueError('no complete paired reaches in the selected held cohort')
    summary = {}
    for name in paired[0]['result']:
        error = np.asarray([row['result'][name]['rms_pixels'] for row in paired])
        excursions = [row['result'][name]['sixth_rad'] for row in paired]
        summary[name] = dict(mean_error_px=float(error.mean()), within_pixel=int(np.sum(error <= 1.)),
                             sixth_median_excursion_rad=float(np.median(excursions)))
    return dict(requested=count, paired=len(paired), summary=summary, held=rows)


def run(args):
    with gzip.open(args.source, 'rt') as stream:
        stored = json.load(stream)
    if sha256(args.policy.read_bytes()).hexdigest() != stored['source_sha256']:
        raise ValueError('source policy and source effects must share the same frozen receipt')
    source = {key: np.asarray(stored['calibration']['stock'][key]) for key in ('encoder', 'command', 'actuator')}
    marker = dict(view='none', width=192, height=144, calibration=True, max_steps=400)
    game = dict(view='wrist', width=192, height=144, max_steps=300)
    launches = seeds(args.seed + 1, args.held)
    if args.seed in launches or set(launches) & set(stored['launch_seeds']):
        raise ValueError('jam check must use fresh held drops')
    report = dict(schema='tt24-redundant-physical-jam-1', source=str(args.source), seed=args.seed,
                  held_seeds=launches, source_sha256=sha256(args.source.read_bytes()).hexdigest(),
                  policy_sha256=stored['source_sha256'], calibration=dict(), outcome=dict(),
                  observed_joint_correspondence='five legacy encoder indices and reset angles')
    with ExitStack() as stack:
        def arena(**kwargs):
            return stack.enter_context(closing(RedundantBounce(**kwargs)))

        data = {name: measure(arena(**marker, **extra), args.seed, active=6, repetitions=1)
                for name, extra in LAYOUT.items()}
        maps = {name: grounding(source, target) for name, target in data.items()}
        report['calibration'] = {name: value['evidence'] | {key: value[key].tolist() for key in source}
                                 for name, value in data.items()}
        report['grounding'] = {name: {key: x.tolist() if isinstance(x, np.ndarray) else x for key, x in value.items()}
                               for name, value in maps.items()}
        active = {name: arena(**game, **extra) for name, extra in LAYOUT.items()}
        conditions = {name: (active[name], drive['transmit'], drive['virtual'], False) for name, drive in maps.items()}
        conditions['jammed_frozen'] = active['jammed'], maps['normal']['transmit'], maps['normal']['virtual'], False
        conditions['jammed_padded'] = active['jammed'], np.eye(5, 6), np.eye(6, 5), False
        conditions['jammed_stationary'] = active['jammed'], np.zeros((5, 6)), np.eye(6, 5), True
        for name, (world, drive, virtual, still) in conditions.items():
            episodes = [play(world, start, args.policy, drive, virtual, stationary=still) for start in launches]
            for episode in episodes:
                verify_replay(world, episode)
            report['outcome'][name] = dict(summary=describe(episodes), episodes=episodes)
            print(name, report['outcome'][name]['summary'], flush=True)
            _write(args.output, report)
        if args.reaches:
            report['optical_reaching'] = motion(stack, source, data, maps, args.seed + 2, args.reaches)
            print('optical reaching', report['optical_reaching']['summary'], flush=True)
            _write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=91360112)
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--reaches', type=int, default=16)
    args = parser.parse_args()
    if args.held < 1 or args.reaches < 0:
        parser.error('need a positive held sample and nonnegative reaching sample')
    run(args)


if __name__ == '__main__':
    main()
