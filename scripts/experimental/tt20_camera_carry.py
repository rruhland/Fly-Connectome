"""Single-camera practice and cross-camera carry in complete strict SO-101 games."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_contextual import optical_placement
from .tt20_curriculum import OUTPUT, configuration, receiver
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport

SETTINGS = dict(mount='clamp', wrist_optics='outboard', marked=False, role_palette='natural',
                width=480, height=360, venue='hall', detail='preview', strict_endline=True, max_steps=190)
PRACTICE = 73402511
GAME = 40582930
EYES = ('left', 'wrist')


def practice_schedule(seed, blocks):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(blocks):
        for candidate in rng.permutation(4):
            draws.append((int(rng.integers(2**31)), int(candidate)))
    return draws


class Station:
    def __init__(self, eye, *, strong=False, ranging='temporal', settings=None):
        self.ranging = ranging
        self.eye = eye
        self.arena = TT20Arena(**(SETTINGS if settings is None else settings), view_mode=eye)
        if strong:
            from .tt20_opponent import install_opponent
            install_opponent(self.arena)
        self.player = receiver(False, OUTPUT)
        self.law = self.player.player.law
        lens = self.arena.lens_intrinsics()[eye]
        self.tracker = NaturalViewTransport(eye, lens_fov={eye: lens}, law=self.law, flight=ranging != 'size')

    def close(self):
        self.arena.close()
        self.player.player.close()

    def point(self, seed, role, candidate):
        self.arena.reset(seed=seed, server=role, render=False)
        self.player.serves.force = candidate
        self.player.reset(role)
        camera = self.tracker.tracker()
        actions, sightings = [], []
        direct_ranges = 0
        digest = hashlib.sha256()
        for tick in range(self.arena.max_steps):
            images = self.arena.render_cameras()
            if tuple(images) != (self.eye,):
                raise ValueError('single-camera policy received an unexpected stream')
            digest.update(images[self.eye])
            camera.update(images)
            xyz = None
            speed = camera.ball_velocity
            # A resolved silhouette can range very near objects more accurately
            # than a fitted trajectory. Both values come from the same image.
            use_size = (self.ranging == 'adaptive' and camera.raw_ball is not None
                        and max(camera.raw_ball_sigma) <= .020)
            if use_size:
                xyz = list(map(float,camera.raw_ball))
                speed = None
                direct_ranges += 1
            elif camera.ball is not None and camera.ball.observed:
                xyz = list(map(float, camera.ball.position))
            if xyz is not None:
                sightings.append((tick, xyz))
            action = self.player.command(tick, xyz, estimated_velocity=speed)
            actions.append(action.tolist())
            _, reward, done, capped, _ = self.arena.step(action, render=False)
            if done or capped:
                break
        return dict(seed=seed, role=role, candidate=candidate, reward=reward, actions=actions,
                    optical=sightings, preferred_size_frames=direct_ranges, rgb_sha256=digest.hexdigest(), outcome=self.arena.point_result(),
                    events=self.arena.events.copy(), merit=optical_placement(sightings, role, reward, self.law, None))


def summarize(rows):
    serves = [r for r in rows if r['role'] == 1]
    receives = [r for r in rows if r['role'] == -1]
    return dict(points=len(rows), wins=sum(r['reward'] > 0 for r in rows), own_serves=len(serves),
                legal_serves=sum(r['outcome']['legal_serve'] for r in serves), receives=len(receives),
                first_returns=sum(r['outcome']['legal_returns'] > 0 for r in receives),
                all_returns=sum(r['outcome']['legal_returns'] for r in rows),
                multi_return_points=sum(r['outcome']['legal_returns'] >= 2 for r in rows),
                frames=sum(len(r['actions']) for r in rows))


def train(eye, blocks):
    data = OUTPUT / f'camera-carry-{eye}-practice.json.gz'
    station = Station(eye)
    _, _, _, _, prior = configuration(OUTPUT)
    rows, picks = [], []
    try:
        for seed, candidate in practice_schedule(PRACTICE, blocks):
            rows.append(station.point(seed, 1, candidate))
            if len(rows) % 4 == 0:
                merits = np.array([np.mean([r['merit'] for r in rows if r['candidate'] == i]) for i in range(4)])
                pick = int(np.argmax(merits + .03 * (np.arange(4) == prior)))
                picks.append(dict(points=len(rows), merits=merits.tolist(), selected=pick))
                print(eye, picks[-1], summarize(rows[-4:]), flush=True)
        _write(data, dict(schema='camera-carry-practice-1', eye=eye, arena=SETTINGS,
                          factory_sha256=_hash(OUTPUT/'curriculum-factory.json.gz'),
                          seed=PRACTICE, points=rows, learning=picks, factory_pick=int(prior),
                          result=summarize(rows)))
    finally:
        station.close()


def verify(data):
    arena = TT20Arena(**data['arena'], view_mode=data['eye'])
    if data.get('strong'):
        from .tt20_opponent import install_opponent
        install_opponent(arena)
    try:
        for row in data['points']:
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            for action in row['actions']:
                _, reward, _, _, _ = arena.step(action, render=False)
            if (arena.point_result() != row['outcome'] or reward != row['reward']
                    or json.dumps(arena.events,sort_keys=True) != json.dumps(row['events'],sort_keys=True)):
                raise ValueError(f"physical replay diverged at {row['seed']}")
    finally:
        arena.close()
    return len(data['points'])


def evaluate(eye, game_seed, strong, pick_from=None, checkpoint=None, ranging='temporal'):
    picks, sources = {}, {}
    checkpoint_suffix = ''
    if checkpoint is None:
        for source in EYES:
            path = OUTPUT / f'camera-carry-{source}-practice.json.gz'
            data = _read(path)
            sources[source] = _hash(path)
            picks[source] = data['learning'][-1]['selected']
            picks['factory'] = data['factory_pick']
    else:
        data = _read(checkpoint)
        factory_digest = data.get('factory_sha256')
        if factory_digest is None and 'source' in data:
            original = checkpoint.parent / data['source']
            if _hash(original) != data['source_sha256']:
                raise ValueError('the original practice checkpoint changed')
            factory_digest = _read(original)['factory_sha256']
        if factory_digest != _hash(OUTPUT/'curriculum-factory.json.gz'):
            raise ValueError('checkpoint references a different factory stroke library')
        sources[checkpoint.name] = _hash(checkpoint)
        picks[checkpoint.name] = (data['state']['pick'] if 'state' in data else data['learning'][-1]['selected'])
        checkpoint_suffix = '-' + sources[checkpoint.name][:12]
    # A ranked stroke index is the only source state given to a destination.
    # Identical checkpoints on identical initial conditions need just one game.
    for candidate in sorted({picks[pick_from]} if pick_from else set(picks.values())):
        station = Station(eye, strong=strong, ranging=ranging)
        rows = []
        score = ServiceScore(first_server=-1)
        rng = np.random.default_rng(game_seed)
        try:
            while not score.finished and len(rows) < 80:
                seed = int(rng.integers(2**31))
                row = station.point(seed, score.server, candidate)
                score.record(row['outcome'])
                rows.append(row)
            if not score.finished:
                raise RuntimeError('game exceeded the predeclared cap')
        finally:
            station.close()
        data = dict(schema='camera-carry-game-1', eye=eye, arena=SETTINGS, seed=game_seed, strong=strong, ranging=ranging,
                    candidate=candidate, selected_by=[name for name, i in picks.items() if i == candidate],
                    training_sha256=sources, points=rows, score=list(score.score), result=summarize(rows))
        data['replayed_points'] = verify(data)
        range_suffix = '' if ranging == 'temporal' else '-' + ranging
        path = OUTPUT / f'camera-carry-{eye}-game-{game_seed}-{candidate}{"-strong" if strong else ""}{checkpoint_suffix}{range_suffix}.json.gz'
        _write(path, data)
        print(eye, data['selected_by'], data['score'], data['result'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eye', choices=EYES)
    parser.add_argument('--train', type=int, metavar='BLOCKS')
    parser.add_argument('--seed', type=int, default=GAME)
    parser.add_argument('--strong', action='store_true')
    parser.add_argument('--ranging', choices=('temporal','adaptive','size'), default='temporal')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--pick-from', choices=EYES)
    source.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.train is not None:
        train(args.eye, args.train)
    else:
        evaluate(args.eye, args.seed, args.strong, args.pick_from, args.checkpoint, args.ranging)
