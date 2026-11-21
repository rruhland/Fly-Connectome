"""Evaluate a full two-bounce serve with matched visual learners and physical controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import gzip
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from tt3_arena import (BALL_RADIUS, BASE, FOREARM, GRAVITY, HZ, INITIAL_JOINTS,
                       JOINT_LIMITS, JOINT_RATES, NET_HEIGHT, PADDLE_RESTITUTION,
                       TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_RESTITUTION,
                       UPPER_ARM, arm_pose, heldout_cases)
from scripts.experimental.tt3_learning_trial import summarize
from tt3_motor_beginner import PreparedTT3Arena
from tt3_numeric_transfer import run_match, summary


RATES = np.array(JOINT_RATES)
RECEIVE_DEPTHS = np.linspace(-1.07, -1.51, 13)


def _inverse_pose(x, y, z):
    depth = y - BASE[1]
    horizontal = math.hypot(x - BASE[0], depth)
    height = z - BASE[2]
    cosine = ((horizontal**2 + height**2 - UPPER_ARM**2 - FOREARM**2)
              / (2 * UPPER_ARM * FOREARM))
    if not -1. < cosine < 1.:
        return None
    elbow = math.acos(cosine)
    shoulder = (math.atan2(height, horizontal)
                - math.atan2(FOREARM * math.sin(elbow), UPPER_ARM + FOREARM * cosine))
    joints = np.array((math.atan2(x - BASE[0], depth), shoulder, elbow))
    if any(not low <= value <= high for value, (low, high) in zip(joints, JOINT_LIMITS)):
        return None
    return joints


def _to_surface(z, vz):
    return (vz + math.sqrt(max(0., vz**2 + 2 * GRAVITY * (z - BALL_RADIUS)))) / GRAVITY


def _receive_at(arena, depth):
    x, y, z = arena.ball
    vx, vy, vz = arena.velocity
    horizon = (depth - y) / vy
    if horizon <= 0.:
        return None
    time_left = horizon
    bounced = arena._bounced
    serve_bounce = arena._server_bounce_due
    for _ in range(2):
        if bounced:
            break
        until_bounce = _to_surface(z, vz)
        bounce_x, bounce_y = x + vx * until_bounce, y + vy * until_bounce
        if (until_bounce >= time_left or abs(bounce_x) > TABLE_HALF_WIDTH
                or abs(bounce_y) > TABLE_HALF_LENGTH):
            return None
        if bounce_y > 0. and serve_bounce:
            serve_bounce = False
        elif bounce_y < 0.:
            bounced = True
        else:
            return None
        x, y, z = bounce_x, bounce_y, BALL_RADIUS
        vz = TABLE_RESTITUTION * (GRAVITY * until_bounce - vz)
        time_left -= until_bounce
    if not bounced:
        return None
    return horizon, (x + vx * time_left, depth, z + vz * time_left - GRAVITY * time_left**2 / 2), (
        vx, vy, vz - GRAVITY * time_left)


def _shot_cost(spot, incoming, joints):
    normal = np.asarray(arm_pose(joints)[3])
    velocity = np.asarray(incoming)
    outgoing = velocity - (1. + PADDLE_RESTITUTION) * (velocity @ normal) * normal
    if outgoing[1] <= .1:
        return 2.
    to_net = -spot[1] / outgoing[1]
    net_height = spot[2] + outgoing[2] * to_net - GRAVITY * to_net**2 / 2
    flight = _to_surface(spot[2], outgoing[2])
    bx, by = np.array(spot[:2]) + flight * outgoing[:2]
    return (1.8 * max(0., NET_HEIGHT + BALL_RADIUS + .04 - net_height)
            + 1.8 * max(0., abs(bx) - (TABLE_HALF_WIDTH - .09))
            + 1.4 * max(0., .13 - by, by - (TABLE_HALF_LENGTH - .13)))


def _geometric_command(arena):
    """Use true state solely as a reachability evaluator for the physical course."""
    if arena.frames < arena.preparation_frames:
        return (0., 0., 0.)
    goal = np.array(INITIAL_JOINTS)
    if arena.velocity[1] < -.3:
        goal = np.array(arena.joints)
        best = math.inf
        for depth in RECEIVE_DEPTHS:
            intercept = _receive_at(arena, depth)
            if intercept is None:
                continue
            horizon, spot, incoming = intercept
            joints = _inverse_pose(spot[0], spot[1] - BALL_RADIUS, spot[2] - .008)
            if joints is None:
                continue
            travel = np.max(abs(joints - arena.joints) / RATES)
            cost = _shot_cost(spot, incoming, joints) + 3. * max(0., travel - horizon) + .35 * abs(depth + 1.29)
            if cost < best:
                goal, best = joints, cost
    return np.clip((goal - arena.joints) * HZ / RATES, -1., 1.)


def physical_controls(seed, points, preparation_frames):
    cases = heldout_cases(f'{seed}/full-serve-controls', points)
    result = {}
    for name in ('parked', 'geometry'):
        started = perf_counter()
        rows = []
        replay = None
        for case in cases:
            arena = PreparedTT3Arena(case.seed, view_seed=case.view_seed, preparation_frames=preparation_frames)
            arena.reset(render=False)
            commands = []
            while not arena.done:
                action = _geometric_command(arena) if name == 'geometry' else (0., 0., 0.)
                commands.append(list(action))
                arena.step(action, render=False)
            row = arena.point_result()
            row.update(case=asdict(case), events=arena.events)
            rows.append(row)
            if replay is None and row['points_for'] and row['legal_returns'] >= 2 and row['rival_returns']:
                replay = {**row, 'actions': commands}
        result[name] = dict(summary=summarize([{'points': rows}]), rows=rows,
                            first_sustained_win=replay, wall_seconds=perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--preparation-frames', type=int, default=24)
    parser.add_argument('--physical-controls', type=int, default=0)
    parser.add_argument('--matches', type=int, default=0)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    payload = dict(seed=args.seed, preparation_frames=args.preparation_frames)
    if args.physical_controls:
        payload['controls'] = physical_controls(args.seed, args.physical_controls, args.preparation_frames)
        print({name: entry['summary'] for name, entry in payload['controls'].items()}, flush=True)
    if args.matches:
        roles = json.loads(args.roles.read_text())
        roles = roles.get('role_model', roles)
        source = json.loads(args.source.read_text())
        conditions = ('trained', 'scratch', 'fresh_goal', 'fixed_extrapolation')
        jobs = [(index, 'standard', roles, source, conditions, args.points, args.seed, True, True)
                for index in range(args.matches)]
        arena = partial(PreparedTT3Arena, preparation_frames=args.preparation_frames)
        runner = partial(run_match, arena_factory=arena)
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            matches = list(pool.map(runner, jobs))
        stats = summary(matches)
        payload['visual'] = dict(stable=True, forecast=True, stats=stats, matches=matches,
                                 source=source, roles=roles, wall_seconds=perf_counter() - started)
        print({condition: phases['overall'] for condition, phases in stats.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))


if __name__ == '__main__':
    main()
