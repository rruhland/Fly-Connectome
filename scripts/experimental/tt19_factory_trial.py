"""Matched calibrated-controller trials with disjoint scored SO-101 holdouts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt19_factory import FactoryModel, FactoryPlayer, StereoBall
from .tt19_flight import FlightLaw
from .tt19_view_trial import SO101ViewArena


DONOR = ROOT / 'artifacts/tt17/motor-gantry-coupling-travel-player.json.gz'
SOURCE = ROOT / 'artifacts/tt19/flight-source.json.gz'
MODES = ('gantry', 'nominal')
MASTER = 431589201
SCAFFOLD = dict(width=192, height=144, venue='hall', view_mode='fixed', detail='preview',
                strict_endline=True, max_steps=150)
BASE = {-1: np.array((0., .85, 0., 0.)), 1: np.array((-.03, .60, .05, .0))}
SPREAD = {-1: np.array((.043, .22, .095, 1.2)), 1: np.array((.034, .24, .12, 1.3))}
LOW = np.array((-.089, .26, -.25, -2.6))
HIGH = np.array((.089, 1.28, .28, 2.6))


def make_player(arena, mode, *, refine=False, reach=False):
    carried = mode == 'gantry'
    law = FlightLaw.from_state(_read(SOURCE)['source_law']) if carried else FlightLaw()
    state = _read(DONOR)['series'] if carried else {}
    response = FlightMemory('impulse', state.get('flight_history', ()), state.get('flight_impulses', ()), frozen=True)
    return FactoryPlayer(FactoryModel(), StereoBall(arena.width, arena.height), law, response,
                         stabilize=refine, reachable=reach)


def play_point(arena, player, server, seed, knobs, stride=2):
    images = arena.reset(seed=seed, server=server)
    player.reset(server, knobs)
    tape, encoders, rewards = [], [], []
    image_hash = sha256()
    while True:
        if images is not None:
            for pixels in images.values():
                image_hash.update(pixels.tobytes())
        q = arena.joints
        encoder = q.tolist()
        action = player.choose(images, q)
        tape.append(action.tolist())
        encoders.append(encoder)
        images, reward, done, capped, _ = arena.step(action, render=len(tape) % stride == 0)
        rewards.append(reward)
        if done or capped:
            break
    return dict(seed=seed, server=server, knobs=knobs.tolist(), actions=tape, optical=player.samples,
                encoder_frames=encoders, rgb_sha256=image_hash.hexdigest(), optical_reversals=player.reversals,
                reward=rewards[-1], evaluator=arena.point_result(), events=arena.events)


def verify(report):
    points = [*report.get('learning', ()), *(row for game in report.get('games', ()) for row in game['points'])]
    arena = SO101ViewArena(**report['arena'])
    try:
        for row in points:
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            for q, action in zip(row['encoder_frames'], row['actions'], strict=True):
                if q != arena.joints.tolist():
                    raise ValueError('factory encoder replay diverged')
                arena.step(action, render=False)
            if arena.point_result() != row['evaluator'] or json.dumps(arena.events) != json.dumps(row['events']):
                raise ValueError('factory trial physical score or event replay diverged')
    finally:
        arena.close()
    for game in report.get('games', ()):
        score = ServiceScore(first_server=-1)
        for row in game['points']:
            score.record(row['evaluator'])
        if not score.finished or list(score.score) != game['score']:
            raise ValueError('replayed factory game score differs')
    return len(points)


def verify_rgb(report, indices):
    """Replay the complete observation-to-command path on selected held points."""
    arena = SO101ViewArena(**report['arena'])
    try:
        player = make_player(arena, report['mode'], refine=report.get('stabilized', False),
                             reach=report.get('reachable', False))
        for index in indices:
            saved = report['games'][0]['points'][index]
            repeated = play_point(arena, player, saved['server'], saved['seed'], np.asarray(saved['knobs']),
                                  report['camera_stride'])
            if json.dumps(saved, sort_keys=True) != json.dumps(repeated, sort_keys=True):
                raise ValueError(f'camera-to-command replay differs on point {index}')
    finally:
        arena.close()
    return list(indices)


def scoring_examples(report):
    first = report['games'][0]['points']
    selected = []
    for role, field in ((-1, 'legal_returns'), (1, 'legal_serve')):
        choices = [i for i, row in enumerate(first) if row['server'] == role]
        if choices:
            selected.append(next((i for i in choices if first[i]['evaluator'][field]), choices[0]))
    return tuple(selected)


def learning_score(player, reward):
    """Use scored reward and room-coordinate RGB progress to rank metered attempts."""
    tracked = [pose for tick, pose in player.history if pose is not None and player.reversals
               and tick > player.reversals[0] + 1]
    far = [point for point in tracked if point[1] > 0]
    closest = min((float(np.linalg.norm(point - [0., .39, .05])) for point in far), default=.95)
    return float(reward + .3 * bool(tracked) + .6 * (1. - min(closest, 1.)))


class JointSearch:
    def __init__(self, rng, role):
        self.rng, self.role = rng, role
        self.rows = []

    def choose(self):
        if not self.rows:
            return BASE[self.role].copy()
        p = self.best()
        if len(self.rows) % 5 == 4:
            return p.copy()
        if len(self.rows) < 7:
            p = BASE[self.role]
        return np.clip(p + self.rng.normal(size=4) * SPREAD[self.role], LOW, HIGH)

    def best(self):
        if not self.rows:
            return BASE[self.role].copy()
        phases = np.array([row[0] for row in self.rows])
        quality = np.array([row[1] for row in self.rows])
        near = (phases[:, None] - phases[None, :]) / SPREAD[self.role]
        weights = np.exp(-.30 * np.sum(near**2, axis=-1))
        averages = (weights @ quality - .25) / (.30 + weights.sum(axis=1))
        return phases[np.argmax(averages)]


def name(mode, refine, reach):
    return f'factory-{mode}' + ('-reachable' if reach else '-stabilized' if refine else '')


def train(folder, mode, attempts, *, refine=False, reach=False):
    arena = SO101ViewArena(**SCAFFOLD)
    prefix = name(mode, refine, reach)
    path = folder / f'{prefix}-experience.json.gz'
    stride = 1 if refine else 2
    report = dict(schema='tt19-factory-1', mode=mode, arena=SCAFFOLD, camera_stride=stride,
                  stabilized=refine, reachable=reach, master_seed=MASTER,
                  source_flight=_hash(SOURCE) if mode == 'gantry' else None,
                  donor=_hash(DONOR) if mode == 'gantry' else None, learning=[], attempts_per_role=attempts)
    rng = np.random.default_rng([MASTER, 92433, 2 if reach else 1] if refine else [MASTER, 92433])
    search = {role: JointSearch(np.random.default_rng([MASTER, role + 117]), role) for role in (-1, 1)}
    try:
        player = make_player(arena, mode, refine=refine, reach=reach)
        for index in range(attempts):
            for role in (-1, 1):
                seed = int(rng.integers(2**31))
                knobs = search[role].choose()
                row = play_point(arena, player, role, seed, knobs, stride)
                score = learning_score(player, row['reward'])
                search[role].rows.append((knobs, score))
                report['learning'].append(dict(row, public_quality=score))
                r = row['evaluator']
                print(mode, 'learn', index + 1, role, r['reason'], row['reward'], r['legal_serve'],
                      r['legal_returns'], np.round(knobs, 3).tolist(), flush=True)
            report['policy'] = {str(role): search[role].best().tolist() for role in search}
            report['learning_frames'] = sum(len(row['actions']) for row in report['learning'])
            _write(path, report)
    finally:
        arena.close()
    report['replayed_points'] = verify(report)
    _write(path, report)


def held_games(folder, mode, count, *, trained=True, refine=False, reach=False):
    arena = SO101ViewArena(**SCAFFOLD)
    prefix = name(mode, refine, reach)
    experience = folder / f'{prefix}-experience.json.gz'
    policy = ({int(key): np.asarray(row) for key, row in _read(experience)['policy'].items()}
              if trained else BASE)
    path = folder / f'{prefix}-{"held" if trained else "baseline"}.json.gz'
    stride = 1 if refine else 2
    report = dict(schema='tt19-factory-1', mode=mode, arena=SCAFFOLD, camera_stride=stride,
                  stabilized=refine, reachable=reach, master_seed=MASTER,
                  experience_sha256=_hash(experience) if trained else None,
                  policy={str(key): row.tolist() for key, row in policy.items()}, games=[])
    try:
        player = make_player(arena, mode, refine=refine, reach=reach)
        for number in range(count):
            rng = np.random.default_rng([MASTER, 375192, number + (80 if reach else 40 if refine else 0)])
            score = ServiceScore(first_server=-1)
            rows = []
            while not score.finished and len(rows) < 65:
                server = score.server
                row = play_point(arena, player, server, int(rng.integers(2**31)), policy[server], stride)
                r = row['evaluator']
                rows.append(row)
                score.record(r)
                print(mode, 'held', number + 1, len(rows), score.score, server, r['reason'],
                      r['legal_serve'], r['legal_returns'], flush=True)
            if not score.finished:
                raise ValueError('held factory game did not finish within 65 scheduled point attempts')
            game = dict(number=number + 1, score=list(score.score), points=rows,
                        legal_returns=sum(r['evaluator']['legal_returns'] for r in rows),
                        legal_serves=sum(r['server'] == 1 and r['evaluator']['legal_serve'] for r in rows),
                        frames=sum(len(r['actions']) for r in rows))
            report['games'].append(game)
            _write(path, report)
    finally:
        arena.close()
    report['replayed_points'] = verify(report)
    _write(path, report)


def pilot(folder, mode, limit):
    arena = SO101ViewArena(**SCAFFOLD)
    rows = []
    try:
        player = make_player(arena, mode)
        for index, server in enumerate(np.tile([-1, 1], limit)):
            seed = int(np.random.default_rng([MASTER, index]).integers(2**31))
            row = play_point(arena, player, int(server), seed, np.array((0., 1., 0., 0.)))
            rows.append(row)
            shown = sum(r['ball'] is not None for r in row['optical'])
            planned = sum(r['plan'] is not None for r in row['optical'])
            result = row['evaluator']
            print(mode, index, server, seed, result['reason'], result['contacts'], result['legal_returns'],
                  result['legal_serve'], result['steps'], shown, planned, row['optical_reversals'], flush=True)
    finally:
        arena.close()
    path = folder / f'factory-{mode}-pilot.json.gz'
    report = dict(arena=SCAFFOLD, learning=rows, mode=mode)
    report['replayed'] = verify(report)
    _write(path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--pilot', type=int)
    parser.add_argument('--train', type=int)
    parser.add_argument('--held', type=int)
    parser.add_argument('--untrained', action='store_true')
    parser.add_argument('--stabilized', action='store_true')
    parser.add_argument('--reachable', action='store_true')
    parser.add_argument('--verify-rgb', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    if args.pilot:
        pilot(args.folder, args.mode, args.pilot)
    refine = args.stabilized or args.reachable
    if args.train:
        train(args.folder, args.mode, args.train, refine=refine, reach=args.reachable)
    if args.held:
        held_games(args.folder, args.mode, args.held, trained=not args.untrained, refine=refine, reach=args.reachable)
    if args.verify_rgb:
        path = args.folder / f'{name(args.mode, refine, args.reachable)}-held.json.gz'
        receipt = _read(path)
        receipt['replayed_rgb_indices'] = verify_rgb(receipt, scoring_examples(receipt))
        _write(path, receipt)
