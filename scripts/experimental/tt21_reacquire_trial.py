"""Compare causal single-eye reply continuation on matched full physical points."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_skill_course import install_feeder
from .tt20_camera_carry import SETTINGS, Station
from .tt20_curriculum import OUTPUT as FACTORY
from .tt20_fixture import TT20Arena
from .tt20_opponent import install_opponent
from .tt21_coach import install_coach
from .tt21_reacquire import ReplyBlock, ReplyTrack, early_reply


OUTPUT = ROOT / 'artifacts' / 'tt21'
SEED = 1172507762
COACH_SEED = 1649047322
DEVELOPMENT = (2107586298, 1392057730, 1794195003, 958455482)


def point(station, seed, continuation, recorded=None, *, lead=False, block=None):
    arena, player = station.arena, station.player
    arena.reset(seed=seed, server=-1, render=False)
    player.reset(-1)
    eye = station.eye
    station.tracker.registry.known.clear()
    tracker = station.tracker.tracker()
    reacquire = ReplyTrack(player.player.law, tracker.transport.bounds)
    actions, samples, images = [], [], []
    prefix = recorded is not None
    for tick in range(arena.max_steps):
        if prefix and tick < len(recorded[0]):
            rgb = recorded[0][tick]
        else:
            frames = arena.render_cameras()
            if tuple(frames) != (eye,):
                raise ValueError('expected exactly the chosen stream')
            rgb = frames[eye]
        if recorded is None:
            images.append(rgb)
        tracker.update({eye: rgb}, tick=tick)
        xyz = None if tracker.ball is None or not tracker.ball.observed else tracker.ball.position
        speed = tracker.ball_velocity
        seen = xyz is not None
        continued = None
        if continuation:
            continued = reacquire.update(tick, eye, tracker, xyz, len(player.player.plans))
            if continued is not None and not seen:
                xyz, speed = continued.position, continued.velocity
        action = player.command(tick, xyz, estimated_velocity=speed)
        if not np.any(action) and (lead or block is not None):
            earlier = (early_reply(player.player, tick, continued) if block is None
                       else block.command(player.player, tick, continued))
            if earlier is None and block is not None:
                earlier = early_reply(player.player, tick, continued)
            if earlier is not None:
                action = earlier
        command = action.tolist()
        actions.append(command)
        if prefix and (tick >= len(recorded[1]) or command != recorded[1][tick]):
            prefix = False

        # Ground truth travels to the receipt after the command has been chosen.
        actual = arena.ball.copy() - [0., 0., TABLE_HEIGHT]
        samples.append(dict(t=tick, seen=seen, estimated=None if xyz is None else list(xyz),
                            actual=actual.tolist(), filled_from=None if continued is None or seen else continued.observed_tick,
                            turn=reacquire.turn))
        _, reward, done, capped, _ = arena.step(command, render=False)
        if done or capped:
            break
    row = dict(seed=seed, role=-1, reward=reward, actions=actions, samples=samples,
               plans=player.player.plans, events=arena.events.copy(), rgb=tracker.digest.hexdigest(),
               outcome=arena.point_result())
    return row, images


def summarize(pairs):
    summary = {}
    for mode in ('frozen', 'continued'):
        rows = [row[mode] for row in pairs]
        filled = [sample for row in rows for sample in row['samples'] if sample['filled_from'] is not None]
        errors = [np.linalg.norm(np.subtract(s['estimated'], s['actual'])) for s in filled]
        summary[mode] = dict(first_returns=sum(row['outcome']['legal_returns'] >= 1 for row in rows),
                             second_returns=sum(row['outcome']['legal_returns'] >= 2 for row in rows),
                             rival_legal_replies=sum(row['outcome']['rival_returns'] > 0 for row in rows),
                             central_reply_windows=sum(central_reply(row) for row in rows),
                             second_plans=sum(len(row['plans']) > 1 for row in rows),
                             filled_frames=len(filled), median_fill_error_m=float(np.median(errors)) if errors else None,
                             median_frames_advanced=float(np.median([s['t'] - s['filled_from'] for s in filled]))
                             if errors else None, wins=sum(row['reward'] > 0 for row in rows))
    return summary


def central_reply(row):
    """Check the physical playability corridor from the evaluator trace."""
    landed = next((event['step'] for event in row['events']
                   if event['kind'] == 'legal_return' and event.get('side') == -1), None)
    return landed is not None and any(
        sample['t'] >= landed and abs(x) < .14 and -.99 < y < -.77 and .07 < z < .32
        for sample in row['samples'] for x, y, z in (sample['actual'],))


def setup_opponent(arena, coach):
    if coach:
        install_coach(arena)
        install_feeder(arena, 'center')
    else:
        install_opponent(arena)


def replay(info):
    arena = TT20Arena(**info['arena'], view_mode=info['eye'])
    setup_opponent(arena, info.get('coach', False))
    count = 0
    try:
        for pair in info['pairs']:
            for mode in ('frozen', 'continued'):
                row = pair[mode]
                arena.reset(seed=row['seed'], server=-1, render=False)
                for action in row['actions']:
                    _, reward, _, _, _ = arena.step(action, render=False)
                if (arena.point_result() != row['outcome'] or reward != row['reward']
                        or json.dumps(arena.events) != json.dumps(row['events'])):
                    raise ValueError(f'physical replay changed at {row["seed"]}')
                count += 1
    finally:
        arena.close()
    return count


def run(folder, eye, count, development, lead, block, coach):
    rng = np.random.default_rng(COACH_SEED if coach else SEED)
    seeds = list(DEVELOPMENT) if development else rng.integers(2**31, size=count).tolist()
    if not development and (len(set(seeds)) != count or set(seeds) & (previous_seeds(FACTORY) | previous_seeds(FACTORY.parent / 'tt19'))):
        raise ValueError('a held start overlaps earlier trials')
    variant = '-block' if block else '-lead' if lead else ''
    suffix = '-coach' if coach else '-development' if development else ''
    path = folder / f'reacquire-{eye}{variant}{suffix}.json.gz'
    station = Station(eye, strong=False, ranging='temporal')
    setup_opponent(station.arena, coach)
    controller = ReplyBlock(station.tracker.bounds) if block else None
    tapes = station.player.player.tapes
    pairs = []
    try:
        for seed in seeds:
            station.player.player.tapes = tapes
            a, images = point(station, seed, False)
            b, _ = point(station, seed, True, (images, a['actions']), lead=lead, block=controller)
            pairs.append(dict(seed=seed, frozen=a, continued=b))
            print(eye, seed, ([row['outcome']['legal_returns'] for row in (a, b)]),
                  'rival', a['outcome']['rival_returns'], 'plans', [len(row['plans']) for row in (a, b)],
                  'filled', sum(s['filled_from'] is not None for s in b['samples']), flush=True)
            _write(path, dict(schema='tt21-reacquire-v1', eye=eye, arena=SETTINGS, strong=True, coach=coach,
                              split='development' if development else 'coaching_pilot' if coach else 'held',
                              lead=lead, block=block,
                              seeds=seeds, pairs=pairs,
                              complete=False))
    finally:
        station.close()
    info = _read(path)
    info['metrics'] = summarize(pairs)
    info['replayed_points'] = replay(info)
    info['complete'] = True
    _write(path, info)
    print(info['metrics'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--eye', choices=('wrist', 'left'), default='wrist')
    cli.add_argument('--starts', type=int, default=16)
    course = cli.add_mutually_exclusive_group()
    course.add_argument('--development', action='store_true')
    course.add_argument('--coach', action='store_true')
    mechanisms = cli.add_mutually_exclusive_group()
    mechanisms.add_argument('--lead', action='store_true')
    mechanisms.add_argument('--block', action='store_true')
    cli.add_argument('--folder', type=Path, default=OUTPUT)
    cli.add_argument('--verify', type=Path)
    args = cli.parse_args()
    if args.verify is not None:
        print('replayed', replay(_read(args.verify)))
        raise SystemExit
    if args.starts < 1:
        cli.error('choose a positive number of starts')
    args.folder.mkdir(parents=True, exist_ok=True)
    run(args.folder, args.eye, args.starts, args.development, args.lead, args.block, args.coach)
