"""Align archived camera plans with privileged physics after replay."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_arena import GRAVITY, TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt5_player import TABLE_END
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt17_rotary_player import replay_first_pace


ARCHIVES = ('tt15/rotary-blind-held', 'tt14/rotary-recenter-held-replication',
            'tt15/rotary-blind-game-1', 'tt15/rotary-blind-game-2', 'tt15/rotary-blind-game-3')
PLANE = -TABLE_END - .022
HEIGHT = np.array([0., 0., TABLE_HEIGHT])


def second_events(row):
    events = row['events']
    bounces = [index for index, event in enumerate(events)
               if event['kind'] == 'legal_return' and event.get('side') == -1]
    which = int(row['server'] == 1)
    if len(bounces) <= which:
        return None
    index = bounces[which]
    earlier = next(event for event in reversed(events[:index])
                   if event['kind'] == 'contact' and event.get('side') == -1)
    until = next((i for i, event in enumerate(events[index + 1:], index + 1)
                  if event['kind'] == 'contact' and event.get('side') == -1), len(events))
    contact = next((event for event in events[index + 1:until]
                    if event['kind'] == 'contact' and event.get('side') == 1), None)
    return earlier, events[index], contact


def physics(replay, row, arena, wiring):
    interface = PointInterface(wiring, prepare_point(arena, row['phase']),
                               server=row['server'], seed=replay['seed'])
    interface.reset(seed=row['seed'], render=False)
    ball, blade = [arena.ball.copy() - HEIGHT], [arena.paddle.copy() - HEIGHT]
    for command in row['actions']:
        interface.step(command, render=False)
        ball.append(arena.ball.copy() - HEIGHT)
        blade.append(arena.paddle.copy() - HEIGHT)
    if interface.applied != row['applied'] or arena.point_result() != row['evaluator']:
        raise ValueError(f"point physics replay differed on {row['seed']}")
    return np.asarray(ball), np.asarray(blade)


def true_arrival(ball, bounce, contact, plane, dt):
    stop = len(ball) if contact is None else contact['step']
    for tick in range(max(1, bounce['step']), stop):
        previous, following = ball[tick - 1:tick + 1]
        if previous[1] > plane >= following[1]:
            fraction = (plane - previous[1]) / (following[1] - previous[1])
            return tick - 1 + fraction, previous + fraction * (following - previous)
    if contact is not None:
        point = np.asarray(contact['ball']) - HEIGHT
        speed = np.asarray(contact['incoming'])
        if speed[1] < -.4:
            time = (plane - point[1]) / speed[1]
            if abs(time) <= 3 * dt:
                arrival = point + time * speed - np.array([0., 0., .5 * GRAVITY * time**2])
                return (contact['time'] + time) / dt, arrival
    return None


def measure(row, events, ball, blade, dt):
    strike, bounce, contact = events
    first = true_arrival(ball, bounce, contact, PLANE, dt)
    output = dict(seed=row['seed'], phase=row['phase'], server=row['server'], rival_strike=strike['step'],
                  bounce=bounce['step'], contact=None if contact is None else contact['step'],
                  landed_second=row['player_returns'] >= 2, result=row['evaluator']['reason'])
    if first is None:
        return dict(output, classification='no_plane_crossing')
    tick, position = first
    plans = [fr for fr in row['optical_trace'] if strike['step'] <= fr['tick'] <= tick + 1
             and fr['horizon'] is not None]
    forecasts = []
    stroke = row['impact']['used_speed']
    for frame in plans:
        now, horizon, pitch = frame['tick'], frame['horizon'], frame['desired_pitch']
        anchor = np.asarray(frame['goal']) + [0., np.clip(stroke * horizon, -.107, .145), 0.]
        hit = anchor + np.array([0., .027 * np.cos(pitch), .027 * np.sin(pitch)])
        reference = true_arrival(ball, bounce, contact, hit[1], dt)
        if reference is None:
            continue
        when, where = reference
        pose_error = blade[now] - frame['goal']
        observed_pose = frame['blade']
        optical_error = None if observed_pose is None else (np.asarray(observed_pose) - frame['goal']).tolist()
        forecasts.append(dict(tick=now, until=when-now, error_frames=now+horizon-when,
                              error_xz=(hit-where)[[0, 2]].tolist(), before_bounce=now < bounce['step'],
                              blade_goal_error_m=pose_error.tolist(), optical_goal_error_m=optical_error))
    milestones = {}
    for lookback in (12, 8, 4, 1):
        compare = [forecast for forecast in forecasts if forecast['until'] >= lookback]
        milestones[str(lookback)] = compare[-1] if compare else None
    tick_index = int(np.clip(round(tick if contact is None else min(tick, contact['time']/dt)), 0, len(blade) - 1))
    if contact is None:
        blade_at_arrival = np.array([np.interp(tick, np.arange(len(blade)), axis) for axis in blade.T])
        actual_gap = position - blade_at_arrival
    else:
        actual_gap = np.asarray(contact['ball']) - np.asarray(contact['racket'])
    observed = row['optical_trace'][tick_index]['blade']
    camera_error = None if observed is None else (np.asarray(observed) - blade[tick_index]).tolist()
    servo_error = np.asarray(row['optical_trace'][tick_index]['actor']) - blade[tick_index]
    before, after = max(0, tick_index - 1), min(len(blade) - 1, tick_index + 1)
    blade_forward = (blade[after, 1] - blade[before, 1]) / ((after-before) * dt)
    first_pace = replay_first_pace(row['optical_trace'], row['timing_route']['first_receiver_frame'])
    separation = None if contact is None else next(
        (event for event in row['events'] if event['kind'] == 'separation' and event.get('collider') == 'learner_racket'
         and contact['time'] < event['time'] <= contact['time'] + 2 * dt), None)
    at8 = milestones['8']
    at4 = milestones['4']
    late = at8 is None or abs(at8['error_frames']) > 2. or np.linalg.norm(at8['error_xz']) > .080
    persistent = at4 is None or abs(at4['error_frames']) > 2. or np.linalg.norm(at4['error_xz']) > .080
    category = ('landed' if output['landed_second'] else 'post_contact' if contact is not None else
                'forecast_persistent' if late and persistent else 'forecast_recovered' if late else 'miss_with_early_plan')
    return dict(output, plane_time=tick, plane_ball=position.tolist(),
                plans=len(plans), first_plan=None if not plans else plans[0]['tick'], milestones=milestones,
                actual_gap=actual_gap.tolist(), camera_blade_error=camera_error, estimated_blade_error=servo_error.tolist(),
                blade_forward_mps=float(blade_forward), first_optical_pace_mps=None if first_pace is None else first_pace/dt,
                departure_mps=None if separation is None else separation['velocity'], classification=category)


def audit(output):
    records = []
    for label in ARCHIVES:
        source = ROOT / f'artifacts/{label}.json.gz'
        tape = _read(source)
        arena = TT8ServiceArena(**tape['arena'])
        wiring = TT6Wiring(arena, tape['installation_seed'])
        try:
            for row in tape['points']:
                events = second_events(row)
                if events:
                    ball, blade = physics(tape, row, arena, wiring)
                    details = measure(row, events, ball, blade, arena.control_dt)
                    records.append(dict(details, source=label, source_sha256=_hash(source)))
        finally:
            arena.close()
    counts = Counter(record['classification'] for record in records)
    _write(output, dict(records=records, counts=dict(counts)))
    print(dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output)
