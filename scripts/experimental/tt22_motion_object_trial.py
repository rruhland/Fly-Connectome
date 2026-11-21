"""Validate anonymous continuous stereo on a native rally and paired fresh points."""

from argparse import ArgumentParser
from colorsys import hsv_to_rgb
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt20_natural import NaturalViewTransport
from .tt20_natural_trial import ORIGIN
from .tt22_motion_object import MotionObject, OPENING, playable
from .tt22_opponent_trial import FarTape
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, RallyArena, alternating, opponent


FOLDER = ROOT / 'artifacts/tt22'
ARCHIVE = FOLDER / 'rally-anchor-612423769.json.gz'
ARCHIVE_SEED = 593435140
ARCHIVE_RESULT = FOLDER / 'motion-object-archive.json.gz'
LIVE_RESULT = FOLDER / 'motion-object-live.json.gz'
SCHEDULE = (783616777, 1208696531, 1480718242, 354768949, 1571798639, 177210998)
HUE_SEED = 190338517


class AppearanceArena(RallyArena):
    """Change a surface material exclusively for the blinded optical evaluation."""

    def __init__(self, spread, pace):
        self.material = None
        super().__init__(spread, pace)

    def _color_roles(self):
        super()._color_roles()
        if self.material is not None:
            self.model.geom('ball').rgba[:3] = self.material


def sensors(arena, variant):
    intrinsics = arena.lens_intrinsics()
    if variant == 'anonymous':
        return MotionObject(intrinsics)
    return NaturalViewTransport('fixed', lens_fov=intrinsics, flight=False, associate=True).tracker()


def step_sensor(tracker, variant, frames, tick):
    if variant == 'anonymous':
        return tracker.update(frames, tick)
    tracker.update(frames, tick=tick)
    estimate = tracker.ball
    if estimate is not None and estimate.observed and tracker.sources.get('ball_range') == 'stereo':
        return np.asarray(estimate.position)
    return None


def sample(position, truth):
    return None if position is None else float(np.linalg.norm(position-truth))


def statistics(records):
    eligible = [row for row in records if row['audit_volume']]
    result = dict(evaluated=len(eligible), first=[row for row in records[:OPENING]])
    for variant in ('named', 'anonymous'):
        errors = np.array([row[variant] for row in eligible if row[variant] is not None])
        result[variant] = dict(observed=len(errors), within_15mm=int(np.sum(errors <= .015)),
                               within_30mm=int(np.sum(errors <= .03)),
                               median_mm=float(1000 * np.median(errors)) if len(errors) else None,
                               p90_mm=float(1000 * np.quantile(errors, .9)) if len(errors) else None)
    return result


def audit_archive(output):
    archive = _read(ARCHIVE)
    row = next(row for row in archive['rows'] if row['seed'] == ARCHIVE_SEED)
    arena = AppearanceArena(archive['spread'], archive['pace'])
    rival = opponent(arena, archive['rival'])
    tape = FarTape(arena)
    observations = []
    tracker = {variant: sensors(arena, variant) for variant in ('named', 'anonymous')}
    digest = sha256()
    try:
        arena.reset(seed=row['seed'], server=row['role'], render=False)
        for tick, command in enumerate(row['actions']):
            frames = arena.render_cameras()
            truth = arena.ball.copy() - ORIGIN
            record = dict(tick=tick, truth=truth.tolist(), audit_volume=playable(truth))
            for key, frame in frames.items():
                digest.update(key.encode())
                digest.update(frame)
            for variant, sensor in tracker.items():
                estimate = step_sensor(sensor, variant, frames, tick)
                record[variant] = sample(estimate, truth)
            observations.append(record)
            _, reward, done, capped, _ = arena.step(command, render=False)
            if bool(done or capped) != (tick + 1 == len(row['actions'])):
                raise ValueError(f'archived replay changed point duration at {tick}')
        if (reward != row['reward'] or arena.point_result() != row['outcome'] or
                digest_json(arena.events) != digest_json(row['events']) or digest.hexdigest() != row['rgb']):
            raise ValueError('archived RGB or outcomes did not replay exactly')
        anonymous = tracker['anonymous']
        for tick, position in anonymous.prefix_positions:
            observations[tick]['anonymous'] = sample(position, np.array(observations[tick]['truth']))
        result = dict(schema='tt22-anonymous-object-archive-1', archive_sha256=_hash(ARCHIVE), seed=row['seed'],
                      outcome=row['outcome'], length=row['longest'], rgb_sha256=digest.hexdigest(),
                      rival_actions=tape.tape(len(row['actions'])), events=arena.events.copy(),
                      prefix_score=anonymous.seed_score, rival_plans=rival.decisions.copy(),
                      agency='archived usual clearing ramp', observations=observations, statistics=statistics(observations))
        _write(output, result)
        print('archive', result['statistics'], flush=True)
    finally:
        arena.close()


def play(arena, rival, tape, player, seed, variant, material, zero):
    arena.material = material
    arena.reset(seed=seed, server=-1, render=False)
    player.reset()
    tape.steps.clear()
    observer = sensors(arena, variant)
    digest = sha256()
    opening_digest = sha256()
    prefix, records, commands = [], [], []
    for tick in range(arena.max_steps):
        frames = arena.render_cameras()
        truth = arena.ball.copy() - ORIGIN
        for key, frame in frames.items():
            digest.update(key.encode())
            digest.update(frame)
            if tick < OPENING:
                opening_digest.update(key.encode())
                opening_digest.update(frame)
        observation = step_sensor(observer, variant, frames, tick)
        records.append(dict(tick=tick, pos=None if observation is None else observation.tolist(),
                            error=sample(observation, truth), audit_volume=playable(truth)))
        if tick < OPENING:
            if zero:
                action = np.zeros(arena.action_size)
                player.history.append((tick, None))
                player.sim.twin.advance(action)
            else:
                action = player.command(tick, None)
            prefix.append(observation)
            if tick == OPENING - 1:
                if variant == 'anonymous':
                    for time, point in observer.prefix_positions:
                        prefix[time] = point
                player.history[:] = list(enumerate(prefix))
        else:
            action = player.command(tick, observation)
        commands.append(action.tolist())
        _, reward, done, capped, _ = arena.step(action, render=False)
        if done or capped:
            break
    estimates = [r['error'] for r in records[OPENING:] if r['error'] is not None and r['audit_volume']]
    return dict(seed=seed, role=-1, rgb_sha256=digest.hexdigest(), actions=commands,
                opening_rgb_sha256=opening_digest.hexdigest(),
                rival_actions=tape.tape(len(commands)), rival=rival.decisions.copy(), reward=reward,
                outcome=arena.point_result(), longest=alternating(arena.events), shots=player.shots,
                prefix_seed=observer.seed_score if variant == 'anonymous' else None, observations=records,
                events=arena.events.copy(), seen=len(estimates),
                correct_15mm=sum(p <= .015 for p in estimates),
                median_mm=float(1000 * np.median(estimates)) if estimates else None)


def verify(rows):
    arena = AppearanceArena(.30, .82)
    verified = 0
    try:
        for pair in rows:
            for variant in ('named', 'anonymous'):
                row = pair[variant]
                arena.material = pair['material']
                arena.reset(seed=row['seed'], server=-1, render=False)
                digest = sha256()
                for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                    for key, frame in arena.render_cameras().items():
                        digest.update(key.encode())
                        digest.update(frame)
                    arena._rival_attempt = True
                    arena._aim = lambda *args: np.asarray(far)
                    _, reward, done, capped, _ = arena.step(near, render=False)
                    if bool(done or capped) != (tick + 1 == len(row['actions'])):
                        raise ValueError('the direct two-arm replay changed the point duration')
                if (reward != row['reward'] or arena.point_result() != row['outcome'] or
                        digest_json(arena.events) != digest_json(row['events']) or
                        digest.hexdigest() != row['rgb_sha256']):
                    raise ValueError(f'{variant} replay changed an actual point')
                verified += 1
    finally:
        arena.close()
    return verified


def live(output, count, zero):
    if not 1 <= count <= len(SCHEDULE):
        raise ValueError('invalid number of paired starts')
    arena = AppearanceArena(.30, .82)
    rival = opponent(arena, 'anchor')
    tape = FarTape(arena)
    player = RallyPlayer(FlightLaw())
    result = dict(schema='tt22-anonymous-object-live-1', arena=CONFIG, opening=.82, spread=.30,
                  zero_opening=zero, first_control=OPENING, hue_seed=HUE_SEED, points=[], complete=False)
    hues = np.random.default_rng(HUE_SEED).uniform(.18, .93, len(SCHEDULE)//2)
    try:
        for i, seed in enumerate(SCHEDULE[:count]):
            material = None if i % 2 == 0 else hsv_to_rgb(float(hues[i//2]), .79, .96)
            pair = {variant: play(arena, rival, tape, player, seed, variant, material, zero)
                    for variant in ('named', 'anonymous')}
            named, anonymous = pair['named'], pair['anonymous']
            if (named['actions'][:OPENING] != anonymous['actions'][:OPENING]
                    or named['rival_actions'][:OPENING] != anonymous['rival_actions'][:OPENING]
                    or named['opening_rgb_sha256'] != anonymous['opening_rgb_sha256']):
                raise ValueError('opening controls or RGB differ between the optical comparisons')
            result['points'].append(dict(seed=seed, material=material, **pair))
            _write(output, result)
            print('live', seed, material, {key: (row['outcome']['legal_returns'], row['longest'],
                                                 row['seen'], row['median_mm'], row['prefix_seed'])
                                          for key, row in pair.items()}, flush=True)
    finally:
        player.close()
        arena.close()
    result['replayed'] = verify(result['points'])
    result['complete'] = True
    _write(output, result)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('archive', 'live'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--points', type=int, default=4)
    parser.add_argument('--zero', action='store_true')
    args = parser.parse_args()
    if args.kind == 'archive':
        audit_archive(args.output or ARCHIVE_RESULT)
    else:
        live(args.output or LIVE_RESULT, args.points, args.zero)
