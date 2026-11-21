"""Render the paired held SO-101 shoulder-preposition win with native RGB proof."""

import argparse
from functools import partial
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_camera import PerspectiveCamera
from .tt4_transfer_trial import _hash, _write
from .tt19_view_trial import CAMERAS
from .tt20_fixture import TT20Arena
from .tt20_opponent import install_opponent
from .tt21_ready_position import PRACTICE, PositionStation
from .tt21_recovery_movie import (AMBER, BG, CHIP, GREEN, INK, LINE, OUTPUT, PANEL_WIDTH,
                                  SIZE, SPECTATOR_HEIGHT, STREAMS, _line, encode, native_events,
                                  paint_views, replay)


SOURCE = OUTPUT / 'ready-position-incoming-held.json.gz'
PRIOR = OUTPUT / 'ready-position-prior-held.json.gz'
POINT = 225376170
MODES = ('center', 'shift')
MOTOR = 'learner_shoulder_pan'


def _event_state(row, tick):
    legal = [t for t in native_events(row, 'legal_return', 1) if t <= tick]
    actions = len(row['actions'])
    rival = [t for t in native_events(row, 'contact', -1) if t <= tick]
    replies = [t for t in native_events(row, 'legal_return', -1) if t <= tick]
    contact = [t for t in native_events(row, 'contact', 1) if t <= tick]
    shifted = bool(row['decision']['goal'])
    if tick >= actions:
        status = 'POINT WON · rival reply goes long' if shifted else 'POINT LOST · reply goes past paddle'
    elif len(rival) > 1:
        status = f'Rival answers again · frame {rival[-1]}'
    elif len(legal) > 1:
        status = f'Second legal return lands · frame {legal[-1]}'
    elif len(contact) > 1:
        status = f'Second paddle contact · frame {contact[-1]}'
    elif shifted and len(row['plans']) > 1 and tick >= row['plans'][1]['tick']:
        status = f'Second stroke launches · frame {row["plans"][1]["tick"]}'
    elif tick >= row['decision']['tick'] and shifted:
        status = f'Shoulder prepositions · frame {row["decision"]["tick"]}'
    elif replies:
        status = f'Rival reply lands · frame {replies[-1]}'
    elif rival:
        status = f'Rival answers first return · frame {rival[-1]}'
    elif legal:
        status = f'First legal return lands · frame {legal[-1]}'
    else:
        status = 'Opponent serves'
    return status, len(legal)


def _arrow(draw, start, end):
    start, end = np.asarray(start), np.asarray(end)
    distance = np.linalg.norm(end - start)
    if distance < 7:
        return
    axis = (end-start)/distance
    tangent = np.array((-axis[1], axis[0]))
    left, right = end - 8*axis + 4*tangent, end - 8*axis - 4*tangent
    draw.line((tuple(start), tuple(end)), fill=GREEN, width=3)
    draw.polygon((tuple(end), tuple(left), tuple(right)), fill=GREEN)


def compose(rows, frames, tick, *, training_count, rate, body, aggregate):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _line(draw, (20, 12), 'SO-101 moves into the path of the return', size=31, color=INK, bold=True)
    _line(draw, (22, 55), f'Held point {POINT}  ·  Unmarked stereo RGB  ·  Half speed  ·  No-practice prior took identical actions here', size=16)
    draw.line((20, 82, 1036, 82), fill=LINE)
    for mode, x, accent in (('center', 20, AMBER), ('shift', 536, GREEN)):
        row = rows[mode]
        index = min(tick, len(row['actions']))
        title = 'CENTERED BASELINE' if mode == 'center' else 'LEARNED PREPOSITION'
        _line(draw, (x, 94), title, size=23, color=accent, bold=True)
        cue = row['decision']['tick']
        detail = 'Waits at center' if mode == 'center' else f'Learns the incoming lane from {training_count} prior rallies'
        if tick >= cue and mode == 'shift':
            lane = row['decision']['lane']*100
            detail = f'Predicted lane {lane:+.1f} cm · learner’s right'
        _line(draw, (x, 125), detail, size=16)
        status, count = _event_state(row, tick)
        draw.rounded_rectangle((x+344, 94, x+499, 129), radius=7, fill=CHIP)
        _line(draw, (x+356, 102), f'{count} legal return{"s" if count != 1 else ""}', size=16, color=accent, bold=True)
        paint_views(canvas, draw, x, frames[mode][index])
        if mode == 'shift' and cue <= tick <= row['plans'][1]['tick']:
            center = body['center'][index]
            shifted = body['shift'][index]
            angle = shifted['shoulder_deg'] - center['shoulder_deg']
            draw.rounded_rectangle((x+265, 440, x+491, 474), radius=5, fill=BG)
            _line(draw, (x+275, 449), f'Shoulder {angle:+.1f}° vs centered', size=13, color=GREEN, bold=True)
            offset = np.array((x, 151))
            _arrow(draw, np.array(center['blade_spectator_px']) + offset,
                   np.array(shifted['blade_spectator_px']) + offset)
        draw.rounded_rectangle((x, 704, x+499, 737), radius=5, fill=CHIP)
        _line(draw, (x+10, 711), status, size=16, color=accent, bold=True)
    _line(draw, (20, 749), f'{aggregate["starts"]} held starts · +{aggregate["continuations"]} legal continuations '
          f'({aggregate["second_returns"]} second return, {aggregate["after_own_serve"]} after own serve) '
          f'· +{aggregate["wins"]} wins vs centered', size=15)
    _line(draw, (883, 749), f'Game time  {tick/rate:0.2f} s', size=15, color=INK)
    return canvas


def check_policy(data, rows, prior, prior_row):
    station = PositionStation()
    learned = np.asarray(data['coefficients'])
    treatments = (('center', rows['center'], np.zeros_like(learned)), ('shift', rows['shift'], learned),
                  ('prior', prior_row, np.asarray(prior['coefficients'])))
    try:
        for mode, row, coef in treatments:
            new = station.point(POINT, row['role'], coef, cue_mode='incoming')
            for field in ('actions', 'optical', 'decision', 'plans', 'rgb', 'outcome', 'events', 'reward'):
                if json.dumps(new[field]) != json.dumps(row[field]):
                    raise ValueError(f'{mode} live policy does not reproduce {field}')
    finally:
        station.close()


def _following(row):
    awaiting = False
    count = 0
    for event in row['events']:
        if event['kind'] == 'legal_return':
            if event['side'] == -1:
                awaiting = True
            elif awaiting:
                count += 1
                awaiting = False
    return count


def _aggregate(data):
    center, shift = (data['conditions'][mode] for mode in MODES)
    copies = [(a, b) for a, b in zip(center['points'], shift['points'], strict=True)]
    if any((a['seed'], a['role']) != (b['seed'], b['role']) for a, b in copies):
        raise ValueError('aggregate held starts are unmatched')
    summary = {key: shift['summary'][key] - center['summary'][key]
               for key in ('multi_return_points', 'subsequent_returns', 'wins')}
    after_serve = sum(_following(b) - _following(a) for a, b in copies if a['role'] == 1)
    return dict(starts=len(copies), continuations=summary['subsequent_returns'],
                second_returns=summary['multi_return_points'], after_own_serve=after_serve, wins=summary['wins'])


def render(source, prior_source, destination, receipt):
    snapshot = source.read_bytes()
    data = json.loads(gzip.decompress(snapshot))
    prior_snapshot = prior_source.read_bytes()
    prior = json.loads(gzip.decompress(prior_snapshot))
    if data['cue_mode'] != 'incoming' or _hash(PRACTICE) != data['source_sha256']:
        raise ValueError('the incoming stereo fit or its original training source differs from archive')
    rows = {mode: next(row for row in data['conditions'][mode]['points'] if row['seed'] == POINT) for mode in MODES}
    a, b = (rows[mode] for mode in MODES)
    first = next(i for i, (x, y) in enumerate(zip(a['actions'], b['actions'])) if x != y)
    cue, lane = b['decision']['tick'], b['decision']['lane']
    if (data['arena']['marked'] or not data['arena']['strict_endline'] or a['role'] != b['role'] or first != cue
            or a['actions'][first][1:] != b['actions'][first][1:] or a['reward'] >= 0 or b['reward'] <= 0
            or [a['outcome']['legal_returns'], b['outcome']['legal_returns']] != [1, 2]
            or not np.isclose(np.dot(b['decision']['cue'], data['coefficients']), lane)):
        raise ValueError('the selected point is not the paired shoulder-preposition win')
    aggregate = _aggregate(data)
    if tuple(aggregate.values()) != (32, 3, 1, 2, 3):
        raise ValueError('aggregate held counts differ from the caption')
    prior_row = next(row for row in prior['points'] if row['seed'] == POINT)
    if (prior['source_sha256'] != sha256(snapshot).hexdigest() or prior['target_examples'] != 0
            or any(prior_row[key] != b[key] for key in ('actions', 'events', 'outcome', 'rgb'))):
        raise ValueError('the no-practice comparison does not match this paired point')
    check_policy(data, rows, prior, prior_row)
    print('all three live policies reproduced exact RGB, motor commands, decisions and referee events', flush=True)
    _, eye, target, fov = CAMERAS['broadcast']
    projector = PerspectiveCamera(eye, target, PANEL_WIDTH, SPECTATOR_HEIGHT, fov)
    body = {mode: [] for mode in MODES}

    def measure(mode, tick, arena):
        angle = np.degrees(float(arena.data.joint(MOTOR).qpos[0]))
        blade = arena.paddle - [0., 0., TABLE_HEIGHT]
        view = projector.project(blade)[:2]
        body[mode].append(dict(tick=tick, shoulder_deg=float(angle), blade_x_m=float(blade[0]),
                               blade_spectator_px=list(map(float, view))))

    arena = TT20Arena(**data['arena'])
    install_opponent(arena)
    view = mujoco.Renderer(arena.model, width=PANEL_WIDTH, height=SPECTATOR_HEIGHT)
    rate = round(1/arena.control_dt)
    try:
        frames = {mode: replay(arena, view, rows[mode], measure=partial(measure, mode)) for mode in MODES}
    finally:
        view.close()
        arena.close()
    destination.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as folder:
        path = Path(folder) / destination.name
        movie = encode(path, rows, frames, training_count=len(data['observations']), rate=rate,
                       compose=partial(compose, body=body, aggregate=aggregate))
        path.replace(destination)
    verified = {mode: dict(rgb_sha256=row['rgb'], decision=row['decision'], commands=len(row['actions']),
                           learner_contact=native_events(row, 'contact', 1), rival_contact=native_events(row, 'contact', -1),
                           legal_learner_returns=native_events(row, 'legal_return', 1),
                           legal_rival_returns=native_events(row, 'legal_return', -1), outcome=row['outcome'],
                           point_sha256=sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                           shoulder_and_blade_evaluator_only=body[mode], live_policy_verified=True, rgb_verified=True,
                           physics_replay_verified=True) for mode, row in rows.items()}
    _write(receipt, dict(schema='tt21-ready-position-video-1', seed=POINT, source=source.name,
                         snapshot_sha256=sha256(snapshot).hexdigest(), arena=data['arena'], opponent='search',
                         optical_training_points=len(data['observations']), practice_sha256=data['source_sha256'],
                         policy_streams=STREAMS, first_action_difference=first, first_difference_axis=MOTOR,
                         caption_counts=aggregate, native_control_hz=rate, movie=movie, branches=verified,
                         no_practice_prior=dict(source=prior_source.name, sha256=sha256(prior_snapshot).hexdigest(),
                                                target_examples=0, same_commands_rgb_and_outcome_as_shift=True,
                                                live_policy_verified=True, points=len(prior['points']),
                                                continuations=prior['summary']['subsequent_returns'],
                                                wins=prior['summary']['wins'], decision=prior_row['decision']),
                         renderer_sha256=_hash(Path(__file__)), shared_replay_sha256=_hash(
                             Path(__file__).with_name('tt21_recovery_movie.py'))))
    print(destination, movie, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--prior', type=Path, default=PRIOR)
    p.add_argument('--destination', type=Path, default=OUTPUT / 'ready-position-rally.mp4')
    p.add_argument('--receipt', type=Path, default=OUTPUT / 'ready-position-rally.json')
    args = p.parse_args()
    render(args.source, args.prior, args.destination, args.receipt)
