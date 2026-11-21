"""Measure physical controls and table rebound in the MuJoCo receiving arena."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_RADIUS, MODEL_PATH, TABLE_HEIGHT, TT4Arena, mujoco


def measure_table_drop(arena, height):
    """Drop a nonspinning ball over the surface and measure raw engine rebound."""
    arena.reset(seed=0, render=False)
    arena.ball[:] = (.3, -.8, TABLE_HEIGHT + BALL_RADIUS + height)
    arena.velocity[:] = 0.
    dof = arena.model.jnt_dofadr[arena.model.joint('ball_free').id]
    arena.data.qvel[dof + 3:dof + 6] = 0.
    pair = {arena.model.geom('table').id, arena.model.geom('ball').id}
    impact = release = first = last = None
    rebound = max_penetration = 0.
    dt = arena.model.opt.timestep
    for _ in range(int(2 / dt)):
        incoming_z = arena.velocity[2]
        arena._aerodynamics()
        mujoco.mj_step(arena.model, arena.data)
        touching = any(set(contact.geom) == pair and contact.efc_address >= 0 for contact in arena.data.contact)
        if touching:
            if impact is None:
                impact = -float(incoming_z)
                first = arena.data.time
            last = arena.data.time
            max_penetration = max(max_penetration, TABLE_HEIGHT + BALL_RADIUS - arena.ball[2])
        elif impact is not None:
            if release is None:
                release = float(arena.velocity[2])
            rebound = max(rebound, arena.ball[2] - TABLE_HEIGHT - BALL_RADIUS)
            if arena.velocity[2] < 0:
                break
    if release is None:
        raise RuntimeError('ball did not leave the tabletop')
    return dict(drop_m=height, rebound_m=float(rebound), incident_mps=impact, rebound_mps=release,
                restitution=release / impact, contact_s=last - first + dt,
                max_penetration_m=float(max_penetration))


def measure_net(arena):
    """Strike the center of the engine's net collider below the tape."""
    arena.reset(seed=0, render=False)
    arena.ball[:] = (.2, .15, TABLE_HEIGHT + .08)
    arena.velocity[:] = (0., -4., 0.)
    entry = None
    while not (arena.terminated or arena.truncated):
        _, _, _, _, info = arena.step(np.zeros(4), render=False)
        for event in info['events']:
            if event['kind'] == 'net':
                entry = event
            if event['kind'] == 'separation' and event['collider'] == 'net':
                return dict(entry, exit_ball=event['ball'], exit_velocity=event['velocity'])
    raise RuntimeError('ball did not touch and then separate from the net')


def measure_controls(arena, seeds, *, random_seed):
    """Run full points with fixed, random, and evaluator-only scripted actions."""
    results = {}
    for name in ('stationary', 'random', 'oracle'):
        rng = np.random.default_rng(random_seed)
        counts, reasons = Counter(), Counter()
        locations, separated = [], []
        for seed in seeds:
            arena.reset(seed=seed, render=False)
            while not (arena.terminated or arena.truncated):
                if name == 'oracle':
                    action = arena.oracle_action()
                elif name == 'random':
                    action = rng.uniform(-1., 1., 4)
                else:
                    action = np.zeros(4)
                arena.step(action, render=False)
            point = arena.point_result()
            counts.update({key: point[key] for key in ('contacts', 'legal_returns', 'rival_contacts',
                                                       'rival_returns', 'points_for', 'points_against')})
            counts['opening_contact_points'] += int(point['contacts'] > 0)
            counts['opening_return_points'] += int(point['legal_returns'] > 0)
            counts['repeated_return_points'] += int(point['legal_returns'] > 1)
            counts['max_legal_player_returns'] = max(counts['max_legal_player_returns'], point['legal_returns'])
            first_bounces = [event['side'] for event in arena.events if event['kind'] == 'bounce'][:2]
            counts['legal_two_bounce_openings'] += int(first_bounces == [-1, 1])
            counts['actual_net_contacts'] += sum(event['kind'] == 'net' for event in arena.events)
            locations.extend(event['location'] for event in arena.events
                             if event['kind'] == 'contact' and event['side'] == 1)
            separated.extend(event['velocity'][1] for event in arena.events
                             if event['kind'] == 'separation' and event['collider'] == 'learner_racket')
            reasons[point['reason']] += 1
        coordinates = np.asarray(locations)
        results[name] = dict(counts, point_reasons=dict(reasons),
                             player_contact_xyz_range_m=np.vstack((coordinates.min(0), coordinates.max(0))).tolist()
                             if coordinates.size else None,
                             median_player_racket_exit_vy_mps=float(np.median(separated)) if separated else None)
    return results


def audit(start_seed=0, points=40, random_seed=85416):
    """Return a JSON-ready validation of the engine and arena's endpoints."""
    if points <= 0:
        raise ValueError('expected a positive point count')
    arena = TT4Arena()
    seeds = tuple(range(start_seed, start_seed + points))
    return dict(mujoco_version=mujoco.__version__, model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                arena_source_sha256=hashlib.sha256(MODEL_PATH.with_suffix('.py').read_bytes()).hexdigest(),
                physics_hz=1 / arena.model.opt.timestep, command_hz=1 / arena.control_dt,
                table_drops=[measure_table_drop(arena, height) for height in (.075, .15, .3, .45, .7)],
                net_contact=measure_net(arena), seeds=seeds, random_seed=random_seed,
                controls=measure_controls(arena, seeds, random_seed=random_seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--points', type=int, default=40)
    parser.add_argument('--random-seed', type=int, default=85416)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.start_seed, args.points, args.random_seed), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')
