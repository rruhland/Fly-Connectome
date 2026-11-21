"""A cooperative physical rival for optional SO-101 receiving practice."""

from functools import partial
from itertools import product

import numpy as np

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT, _racket_loft
from .tt20_opponent import RivalSearch, aim_angles, forecast, install_opponent, trial_quality


def receive_window(ball, speed, landed, *, lateral=.26, depth=(-1.11, -.77), height=(.068, .43)):
    """Find reachable-height, central incoming samples after a receiving bounce."""
    return np.flatnonzero((np.arange(len(ball)) > landed)
                          & (ball[:, 1] < depth[1]) & (ball[:, 1] > depth[0])
                          & (np.abs(ball[:, 0]) <= lateral)
                          & (ball[:, 2] > TABLE_HEIGHT + height[0]) & (ball[:, 2] < TABLE_HEIGHT + height[1])
                          & (speed[:, 1] < -.30))


def _targets(opponent):
    arena = opponent.arena
    frames = np.arange(52) * arena.control_dt
    court = arena.table_half_length
    hits = forecast(arena, tuple(court + depth for depth in (.028, .078, .135)))
    goals, descriptions = [], []
    for depth, (when, center, speed) in hits.items():
        x = center[0]
        sweep = (0., -.65 * np.sign(x), -1.35 * np.sign(x)) if abs(x) > .045 else (0., -.32, .32)
        for pace, lead, loft, lateral in product((.13, .82, 1.52), (.033, .078), (-.11, .005, .12, .25), sweep):
            pitch = float(np.clip(_racket_loft(-speed[1], speed[2], center[2], depth + .84 * court,
                                               pace) + loft, -.49, .81))
            motion = np.clip(pace * (frames - when + lead), -.15, .16)
            depth_goal = depth + (BALL_RADIUS + .006) * np.cos(pitch) - motion
            height = center[2] - (BALL_RADIUS + .006) * np.sin(pitch)
            sweep_x = x + np.clip(lateral * (frames - when + lead), -.16, .16)
            goals.append(aim_angles(arena.rival_base_depth, sweep_x, depth_goal, height, pitch))
            descriptions.append((depth, pace, lead, pitch, lateral))
    return np.asarray(goals), descriptions


def _teaching_quality(opponent, paths, rows):
    arena = opponent.arena
    position = paths[..., opponent.position:opponent.position + 3]
    speeds = paths[..., opponent.velocity:opponent.velocity + 3]
    rng = np.random.default_rng((arena.seed, arena.rival_contacts, arena.contacts, 7351))
    x_target, z_target, pace_target = rng.uniform((-.13, .13, 1.8), (.13, .29, 3.4))
    scored = {}
    for index, row in enumerate(rows):
        if row is None:
            continue
        centers, v = position[index], speeds[index]
        landed = np.flatnonzero((v[:-1, 2] < 0.) & (v[1:, 2] > 0.) & (centers[1:, 1] < -.023)
                                & (centers[1:, 2] < TABLE_HEIGHT + BALL_RADIUS + .026)
                                & (np.abs(centers[1:, 0]) < arena.table_half_width + .005)
                                & (centers[1:, 1] > -arena.table_half_length - .005)) + 1
        if not len(landed):
            continue
        first = int(landed[0])
        window = receive_window(centers, v, first, lateral=.39, depth=(-1.15, -.75), height=(.052, .44))
        if len(landed) > 1:
            window = window[window < landed[1]]
        if not len(window):
            continue
        point = window[np.argmin(np.abs(centers[window, 1] + .91))]
        x, y, z = centers[point] - [0., 0., TABLE_HEIGHT]
        vx, vy, _ = v[point]
        # Seeded variations are secondary to providing a usable strike window.
        loss = (2.6 * abs(x - x_target) + 4. * abs(y + .91) + 2.7 * abs(z - z_target)
                + .19 * abs(-vy - pace_target) + .08 * abs(vx)
                + .6 * max(0., .62 - abs(row['landing'][1])) + .025 * row['cost'])
        scored[index] = (float(loss), [float(x), float(y), float(z)], len(window))
    return scored


def _plan(opponent):
    arena = opponent.arena
    goals, options = _targets(opponent)
    if not len(goals):
        return None, dict(candidates=0, predicted_legal=0, teaching_candidates=0)
    tape, paths = opponent.predict(goals)
    rows = trial_quality(paths, opponent.position, opponent.velocity, arena.table_half_length,
                         arena.table_half_width, arena._received_bounce)
    scored = _teaching_quality(opponent, paths, rows)
    if scored:
        picked = min(scored, key=scored.get)
        _, corridor, window = scored[picked]
    else:
        valid = [i for i, row in enumerate(rows) if row is not None]
        if not valid:
            return None, dict(candidates=len(rows), predicted_legal=0, teaching_candidates=0)
        picked = min(valid, key=lambda i: rows[i]['cost'])
        corridor, window = None, 0
    return tape[picked], dict(candidates=len(rows), predicted_legal=sum(row is not None for row in rows),
                              teaching_candidates=len(scored), selected=options[picked],
                              predicted_corridor=corridor, window_microsteps=window, **rows[picked])


def install_coach(arena):
    """Change only rival receiving commands. The ball and strict referee are unmodified."""
    rival = install_opponent(arena)
    rival.plan = partial(_plan, rival)
    return rival
