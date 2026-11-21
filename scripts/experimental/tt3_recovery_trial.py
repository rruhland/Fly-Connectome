"""Audit first and later reachable chances after each camera-only action."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from tt3_arena import (BASE, FOREARM, IMAGE_HEIGHT, IMAGE_WIDTH, JOINT_LIMITS, PADDLE_RADIUS,
                       UPPER_ARM, TT3Arena, _project, heldout_cases)
from tt3_numeric_transfer import PortableMemory, VisualPlayer
from tt3_visual import RoleModel


def screen(world):
    horizontal, top = _project(world, 0)
    _, side = _project(world, 1)
    return (np.asarray((horizontal, top, side)) + .5 -
            (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, IMAGE_HEIGHT / 2)) / IMAGE_WIDTH


def pose_exists(point):
    """The point center can lie in the physical end-effector workspace."""
    x, y, z = (a - b for a, b in zip(point, BASE, strict=True))
    radial = math.hypot(x, y)
    cosine = (radial * radial + z * z - UPPER_ARM**2 - FOREARM**2) / (2 * UPPER_ARM * FOREARM)
    if not -1. <= cosine <= 1.:
        return False
    elbow = math.acos(cosine)
    shoulder = math.atan2(z, radial) - math.atan2(FOREARM * math.sin(elbow),
                                                 UPPER_ARM + FOREARM * math.cos(elbow))
    return all(low <= angle <= high for angle, (low, high) in zip(
        (math.atan2(x, y), shoulder, elbow), JOINT_LIMITS, strict=True))


def measure(arena, player, rows):
    if arena._hitter != -1 or not arena._bounced or arena.velocity[1] >= 0. or arena.ball[1] > -.55:
        return
    phase = arena.rival_returns
    actual = screen(arena.ball)
    paddle = screen(arena.paddle)
    positions = player.history[-1].positions
    control = positions.get(player.identity)
    relevant = positions.get(player.flight.previous_track)
    ready = player.ready
    normal = player.normal
    goal = player.goal
    projection = None if ready is None or normal is None else float((actual - ready) @ normal)
    goal_error = None if goal is None or normal is None else float(np.linalg.norm(
        (actual - goal) - ((actual - goal) @ normal) * normal))
    rows.setdefault(phase, []).append(dict(frame=arena.frames, ball=arena.ball, velocity=arena.velocity,
        paddle=arena.paddle, control_ok=control is not None and bool(np.linalg.norm(control - paddle) < .019),
        relevant_ok=relevant is not None and bool(np.linalg.norm(relevant - actual) < .019),
        depth=projection, goal_error=goal_error, center_reachable=pose_exists(arena.ball),
        lateral_error=math.hypot(arena.ball[0] - arena.paddle[0], arena.ball[2] - arena.paddle[2])))


def play(arena, player):
    player.reset()
    frames = arena.reset()
    phases = {}
    initial_trace = sha256()
    while True:
        action = player.act(frames)
        if not arena.contacts:
            initial_trace.update(np.asarray(action, dtype='<f8').tobytes())
        measure(arena, player, phases)
        frames, reward, done, _ = arena.step(action)
        player.observe(reward, terminal_frames=frames if done else None)
        if done:
            break
    result = arena.point_result()
    detail = []
    legal_returns = sum(event['kind'] == 'legal_return' and event.get('side') == -1 for event in arena.events)
    if any(event['kind'] == 'bounce' and event.get('side') == 1 for event in arena.events):
        for phase in range(legal_returns + 1):
            rows = phases.get(phase, [])
            start = next((event['frame'] for idx, event in enumerate(arena.events)
                          if event['kind'] == 'legal_return' and event.get('side') == -1
                          and sum(old['kind'] == 'legal_return' and old.get('side') == -1
                                  for old in arena.events[:idx]) == phase - 1), 0) if phase else 0
            finish = next((event['frame'] for event in arena.events if event['frame'] > start
                           and event['kind'] == 'contact' and event.get('side') == 1), None)
            if rows and finish is not None:
                rows = [row for row in rows if row['frame'] < finish]
            fixed_plane = [row for row in rows if row['depth'] is not None]
            closest = min(fixed_plane, key=lambda row: abs(row['depth'])) if fixed_plane else None
            reached = any(abs(row['depth']) < .009 for row in fixed_plane)
            detail.append(dict(phase=phase, start=start, contacted=finish is not None,
                               reach_frames=sum(row['center_reachable'] for row in rows),
                               reached_nominal_plane=reached,
                               correct_control=sum(row['control_ok'] for row in rows),
                               correct_relevance=sum(row['relevant_ok'] for row in rows),
                               frames=len(rows), closest=closest, frames_data=rows))
    result.update(audit=detail, first_action_sha256=initial_trace.hexdigest(), events=arena.events,
                  **player.finish())
    return result


def aggregate(rows):
    phases = Counter()
    misses = Counter()
    for point in rows:
        for chance in point['audit']:
            stage = 'later' if chance['phase'] else 'first'
            phases[stage, 'chances'] += 1
            phases[stage, 'contacted'] += chance['contacted']
            phases[stage, 'reachable'] += chance['reach_frames'] > 0
            phases[stage, 'reached_fixed_plane'] += chance['reached_nominal_plane']
            if stage == 'later' and not chance['contacted']:
                misses['short_reachable'] += bool(chance['reach_frames'] and not chance['reached_nominal_plane'])
                misses['reachable_at_fixed_plane'] += bool(chance['reach_frames'] and chance['reached_nominal_plane'])
                misses['center_never_reachable'] += not chance['reach_frames']
                misses['bad_control_majority'] += chance['correct_control'] < chance['frames'] / 2
                misses['bad_target_majority'] += chance['correct_relevance'] < chance['frames'] / 2
    return dict(points=len(rows), landed=sum(p['legal_returns'] > 0 for p in rows),
                landed_total=sum(p['legal_returns'] for p in rows),
                rival_returns=sum(p['rival_returns'] for p in rows),
                repeated_points=sum(p['legal_returns'] >= 2 for p in rows),
                later_landings=sum(max(0, p['legal_returns'] - 1) for p in rows),
                recovery_frames=sum(p.get('recovery_frames', 0) for p in rows),
                wins=sum(p['points_for'] for p in rows),
                chances={stage: {key: value for (phase, key), value in phases.items() if phase == stage}
                         for stage in ('first', 'later')}, later_misses=dict(misses))


def _same_openings(args, roles, numeric):
    from tt3_recovery import RecoveryPlayer

    ordinary, adapted = [], []
    for match in range(args.matches):
        seed = args.seed + match
        schedule = heldout_cases(seed, args.points, variant=args.variant)
        base = VisualPlayer(RoleModel(**roles), PortableMemory.restore(numeric), seed=seed,
                            recall=True, forecast=True, learn_roles=True)
        for case in schedule:
            other = RecoveryPlayer(base.binder.model, PortableMemory.restore(base.memory.state()),
                                   seed=seed, recall=True, forecast=True, learn_roles=True, defer=3.8)
            other.binder, other.motor, other.evidence, other.rng = deepcopy(
                (base.binder, base.motor, base.evidence, base.rng))
            setup = dict(view_seed=schedule[0].view_seed, variant=case.variant)
            point = play(TT3Arena(case.seed, **setup), base)
            if not point['rival_returns']:
                continue
            alternate = play(TT3Arena(case.seed, **setup), other)
            for row in (point, alternate):
                row.update(case=asdict(case), match=match)
            ordinary.append(point)
            adapted.append(alternate)
    matched = sum(a['first_action_sha256'] == b['first_action_sha256']
                  for a, b in zip(ordinary, adapted, strict=True))
    return dict(first_action_matched=matched, baseline=aggregate(ordinary),
                deeper=aggregate(adapted), pairs=list(zip(ordinary, adapted, strict=True)))


def run(args):
    roles = json.loads(args.roles.read_text())['role_model']
    numeric = json.loads(args.numeric.read_text())
    headers = dict(seed=args.seed, variant=args.variant, matches=args.matches, points=args.points,
                   roles=roles, numeric=numeric)
    if args.same_openings:
        result = _same_openings(args, roles, numeric)
        with gzip.open(args.output, 'wt') as output:
            json.dump(dict(**headers, same_openings=result), output, separators=(',', ':'))
        print({key: value for key, value in result.items() if key != 'pairs'}, flush=True)
        return
    condition_results = {}
    for name in args.conditions:
        rows = []
        for match in range(args.matches):
            schedule = heldout_cases(args.seed + match, args.points, variant=args.variant)
            cls = VisualPlayer
            if name != 'base':
                from tt3_recovery import RecoveryPlayer
                cls = RecoveryPlayer
            policy = dict(defer=3.8 if name == 'later' else 0.) if cls is not VisualPlayer else {}
            player = cls(RoleModel(**roles), PortableMemory.restore(numeric), seed=args.seed + match,
                         recall=True, forecast=True, learn_roles=True, **policy)
            for case in schedule:
                point = play(TT3Arena(case.seed, view_seed=schedule[0].view_seed, variant=case.variant), player)
                point.update(case=asdict(case), match=match)
                rows.append(point)
        condition_results[name] = dict(summary=aggregate(rows), points=rows)
        print(name, condition_results[name]['summary'], flush=True)
    if 'base' in condition_results:
        base = condition_results['base']['points']
        for name, current in condition_results.items():
            if name == 'base':
                continue
            pairs = list(zip(base, current['points'], strict=True))
            same = [(x, y) for x, y in pairs if x['first_action_sha256'] == y['first_action_sha256']]
            current['first_action_matched'] = dict(points=len(same), base=aggregate([x for x, _ in same]),
                                                   variant=aggregate([y for _, y in same]))
    with gzip.open(args.output, 'wt') as output:
        json.dump(dict(**headers, results=condition_results), output, separators=(',', ':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--roles', type=Path, default=Path('artifacts/tt3/pong-source.json'))
    parser.add_argument('--numeric', type=Path, default=Path('/workspace/m1b-experiments/tt3-numeric-final/numeric-state.json'))
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--seed', type=int, default=786225)
    parser.add_argument('--variant', default='standard')
    parser.add_argument('--conditions', nargs='+', choices=('base', 'recovery', 'later'),
                        default=('base', 'recovery', 'later'))
    parser.add_argument('--same-openings', action='store_true',
                        help='replay baseline replies with identical opening learner state')
    run(parser.parse_args())
