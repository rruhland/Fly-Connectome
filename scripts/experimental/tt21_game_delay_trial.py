"""Paired strict single-eye games with an unannounced actuator delay change."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_camera_carry import EYES, SETTINGS, summarize
from .tt20_curriculum import CHECKPOINT, OUTPUT as TT20, configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe
from .tt20_natural import NaturalViewTransport
from .tt21_game_delay import GameDelayPlayer, NominalMotion, PassiveDelay, motion, motion_sample


OUTPUT = ROOT / 'artifacts' / 'tt21' / 'game-delay.json.gz'
SEEDS = (1920024739, 384469573)
SWITCHBACK_SEED = 1692022069
SWITCH_AFTER = 4
EVALUATOR_DELAY = 4
SWITCHBACK_SCHEDULE = ((0, 0), (SWITCH_AFTER, 2), (8, 0))
STEP_SCHEDULE = ((0, 0), (SWITCH_AFTER, EVALUATOR_DELAY))
BRANCHES = ('frozen', 'adaptive')


def transport_delay(schedule, completed_points):
    return next(delay for first, delay in reversed(schedule) if completed_points >= first)


def play_point(arena, player, transport, eye, seed, role, candidate, pipe, learner):
    arena.reset(seed=seed, server=role, render=False)
    player.serves.force = candidate
    player.reset(role)
    lead = player.lead
    tracker = transport.tracker()
    commands, applied, observed, measured = [], [], [], []
    previous = None
    clock = perf_counter()
    for tick in range(arena.max_steps):
        images = arena.render_cameras()
        if tuple(images) != (eye,):
            raise ValueError('unexpected stream in the single-eye test')
        if learner is not None:
            sample = motion_sample(eye, images[eye])
            if previous is not None:
                measured.append(motion(previous, sample))
            previous = sample
        tracker.update(images, tick=tick)
        ball = tracker.ball
        xyz = None if ball is None or not ball.observed else list(map(float, ball.position))
        request = player.command(tick, xyz, estimated_velocity=tracker.ball_velocity)
        action = pipe.apply(request)
        commands.append(request.tolist())
        applied.append(action.tolist())
        observed.append(xyz)
        _, reward, done, capped, _ = arena.step(action, render=False)
        if done or capped:
            break
    learning = None if learner is None else learner.observe(np.asarray(commands), measured)
    if learner is not None:
        player.lead = learner.delay
    return dict(seed=seed, role=role, candidate=candidate, lead=lead, actions=commands, applied=applied,
                optical=observed, measured=measured, learning=learning, rgb_sha256=tracker.digest.hexdigest(),
                outcome=arena.point_result(), events=arena.events.copy(), plans=player.player.plans, reward=reward,
                wall_seconds=perf_counter() - clock)


def summarize_game(rows):
    result = summarize(rows)
    result.update(after_change=summarize(rows[SWITCH_AFTER:]),
                  nominal_frames=sum(row['learning']['rendered'] for row in rows if row['learning'] is not None),
                  delay_inference_seconds=sum(row['learning']['wall_seconds'] for row in rows if row['learning'] is not None),
                  additional_target_frames=0)
    return result


def play_game(arena, player, eye, seed, candidate, nominal, branch, schedule):
    adaptive = branch == 'adaptive'
    learner = PassiveDelay(nominal) if adaptive else None
    player.lead = 0
    lens = arena.lens_intrinsics()[eye]
    transport = NaturalViewTransport(eye, lens_fov={eye: lens}, law=player.player.law)
    score = ServiceScore(first_server=-1)
    rng = np.random.default_rng(seed)
    points = []
    while not score.finished and len(points) < 70:
        hidden = transport_delay(schedule, len(points))
        point_seed, role = int(rng.integers(2**31)), score.server
        point = play_point(arena, player, transport, eye, point_seed, role, candidate, CommandPipe(hidden), learner)
        score.record(point['outcome'])
        points.append(dict(point, evaluator_delay=hidden, score=list(score.score)))
        print(eye, seed, branch, len(points), tuple(score.score), role, point['lead'],
              None if not adaptive else (point['learning']['status'], point['learning']['delay']),
              point['outcome']['legal_returns'], flush=True)
    if not score.finished:
        raise RuntimeError('game exceeded the declared point cap')
    return dict(eye=eye, seed=seed, branch=branch, score=list(score.score), winner=score.winner,
                points=points, result=summarize_game(points))


def replay_game(arena, game, schedule):
    score = ServiceScore(first_server=-1)
    lead = 0
    for index, point in enumerate(game['points']):
        if point['lead'] != lead or point['evaluator_delay'] != transport_delay(schedule, index):
            raise ValueError('saved lead or transport schedule diverged')
        if game['branch'] == 'adaptive':
            if point['learning']['previous'] != lead:
                raise ValueError('saved estimate diverged at the point boundary')
            lead = point['learning']['delay']
        if score.server != point['role']:
            raise ValueError('service changed on replay')
        arena.reset(seed=point['seed'], server=point['role'], render=False)
        queue = CommandPipe(point['evaluator_delay'])
        for request, applied in zip(point['actions'], point['applied'], strict=True):
            action = queue.apply(request)
            if action.tolist() != applied:
                raise ValueError('timing transport changed on replay')
            _, reward, _, _, _ = arena.step(action, render=False)
        score.record(arena.point_result())
        if (arena.point_result() != point['outcome'] or reward != point['reward'] or
                list(score.score) != point['score'] or json.dumps(arena.events) != json.dumps(point['events'])):
            raise ValueError(f"physical replay diverged at {point['seed']}")
    if not score.finished:
        raise ValueError('game replay stopped early')
    return len(game['points'])


def paired_games(games):
    frozen, adaptive = (games[branch]['points'] for branch in BRANCHES)
    prefix = 0
    paired = []
    for index, (base, learned) in enumerate(zip(frozen, adaptive)):
        if (base['seed'], base['role']) != (learned['seed'], learned['role']):
            continue
        if (index == prefix and base['actions'] == learned['actions'] and
                base['rgb_sha256'] == learned['rgb_sha256'] and base['events'] == learned['events']):
            prefix += 1
        paired.append((index, base, learned))
    disagreement = [(a, b) for i, a, b in paired if i >= SWITCH_AFTER and a['reward'] != b['reward']]
    return dict(common_starts=len(paired), identical_rgb_action_prefix=prefix,
                changed_point_wins=[sum(pair[i]['reward'] > 0 for pair in disagreement) for i in (0, 1)])


def run(path, eyes, seeds, *, resume=False, schedule=STEP_SCHEDULE):
    programs, catalogue, law, response, candidate = configuration(TT20)
    metadata = dict(schema='tt21-game-delay-1', arena=SETTINGS, eyes=list(eyes), game_seeds=list(seeds), switch_after=SWITCH_AFTER,
                evaluator_delay=schedule[1][1], candidate=int(candidate), factory_sha256=_hash(TT20/'curriculum-factory.json.gz'),
                source_sha256=_hash(CHECKPOINT), atlas_sha256=_hash(TT20/'curriculum-dry-catalog.npz'))
    if schedule != STEP_SCHEDULE:
        metadata['transport_schedule'] = [list(phase) for phase in schedule]
    data = dict(metadata, results={}, replayed_points=0)
    if resume and path.exists():
        data = _read(path)
        if any(data.get(key) != value for key, value in metadata.items()):
            raise ValueError('saved matches used a different fixture or schedule')
    path.parent.mkdir(parents=True, exist_ok=True)
    player = GameDelayPlayer(programs, catalogue, law, response, model_arena())
    try:
        for index, seed in enumerate(seeds):
            for eye in eyes:
                finished = data['results'].setdefault(eye, [])
                if any(pair['seed'] == seed for pair in finished):
                    continue
                arena = TT20Arena(**SETTINGS, view_mode=eye)
                nominal = NominalMotion(eye, SETTINGS)
                try:
                    branches = BRANCHES if index % 2 == 0 else tuple(reversed(BRANCHES))
                    games = {}
                    for branch in branches:
                        game = games[branch] = play_game(arena, player, eye, seed, candidate, nominal, branch, schedule)
                        data['replayed_points'] += replay_game(arena, game, schedule)
                    finished.append(dict(seed=seed, games=games, matched=paired_games(games)))
                    _write(path, data)
                    print(eye, seed, {branch: (game['score'], game['result']) for branch, game in games.items()}, flush=True)
                finally:
                    nominal.close()
                    arena.close()
    finally:
        player.player.close()


def verify(path):
    data = _read(path)
    schedule = data.get('transport_schedule', STEP_SCHEDULE)
    points = 0
    for eye, rows in data['results'].items():
        arena = TT20Arena(**data['arena'], view_mode=eye)
        try:
            for row in rows:
                for game in row['games'].values():
                    points += replay_game(arena, game, schedule)
        finally:
            arena.close()
    if points != data['replayed_points']:
        raise ValueError('the saved replay count changed')
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--eyes', nargs='+', choices=EYES)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--switchback', action='store_true', help='test 0 to 2 to 0 frames on the outside camera')
    args = parser.parse_args()
    path = args.output or (OUTPUT.with_name('game-delay-switchback.json.gz') if args.switchback else OUTPUT)
    if args.verify:
        print('replayed', verify(path))
    else:
        eyes = args.eyes or (('left',) if args.switchback else EYES)
        seeds = args.seeds or ((SWITCHBACK_SEED,) if args.switchback else SEEDS)
        schedule = SWITCHBACK_SCHEDULE if args.switchback else STEP_SCHEDULE
        run(path, eyes, seeds, resume=args.resume, schedule=schedule)
