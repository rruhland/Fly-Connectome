"""Render all points of a native TT19 match and verify the original RGB cadence."""
import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_view import STREAMS, ViewTransport
from .tt19_view_replay import _font
from .tt19_view_trial import SO101ViewArena


ROOT = Path('artifacts/tt19')
GAME = 424520913
ARENA = dict(venue='hall', strict_endline=True, width=240, height=180,
             view_mode='wrist+fixed', max_steps=190)
VIDEO_SIZE = (1280, 778)
MAIN_SIZE = (864, 648)
GEOMETRY = ((8, 68), (880, 100), (880, 433), (1078, 433))
ROBOT = '#f0a6fa'
RIVAL = '#9bd3fb'
INK = '#eef4f8'
MUTED = '#a2b2c0'
BACKGROUND = '#11181f'
RESULTS = {
    'own_half': 'Ball landed on the hitter’s half',
    'long_past_endline': 'Shot went long',
    'floor': 'Ball hit the floor',
    'out': 'Ball went out',
    'second_bounce': 'Second bounce',
    'serve_let': 'Let · replay serve',
    'serve_missed_toss': 'Missed serve',
    'serve_missed_own_half': 'Serve missed the server’s half',
    'serve_illegal_toss_strike': 'Illegal toss strike',
    'serve_illegal_strike_position': 'Illegal service position',
    'volley': 'Volley before the bounce',
}
FONT = lru_cache(None)(_font)


def digest_json(value):
    canonical = json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
    return sha256(canonical).hexdigest()


def select_game(report, seed):
    if not (report['native'] and report['strict_endline'] and report['course'] == 'full'):
        raise ValueError('expected native RGB play on the full strict court')
    game = next((game for game in report['games'] if game['seed'] == seed), None)
    if game is None or not game['finished']:
        raise ValueError(f'game {seed} is missing or unfinished')
    return game


def camera_ticks(row):
    ticks = [tick for tick, _ in row['sources']]
    count = len(row['actions'])
    if len(row['optical']) != count or not ticks or ticks != sorted(set(ticks)) or ticks[-1] >= count:
        raise ValueError('the optical tape does not align to the requested actions')
    first_odd = next((tick for tick in ticks if tick % 2), count)
    expected = [tick for tick in range(count) if tick % 2 == 0 or tick >= first_odd]
    if ticks != expected or (row['server'] == -1 and ticks != list(range(count))):
        raise ValueError('the optical tape does not follow the original capture cadence')
    return ticks


def spectator_renderer(arena):
    visual = arena.model.vis
    initial = visual.global_.offwidth, visual.global_.offheight, visual.quality.offsamples
    try:
        visual.global_.offwidth, visual.global_.offheight = MAIN_SIZE
        visual.quality.offsamples = 0
        return mujoco.Renderer(arena.model, width=MAIN_SIZE[0], height=MAIN_SIZE[1])
    finally:
        visual.global_.offwidth, visual.global_.offheight, visual.quality.offsamples = initial


def frame_label(draw, box, title, subtitle=None, *, active=False):
    x, y = box
    draw.text((x, y), title, font=FONT(15), fill=INK)
    if subtitle is not None:
        draw.text((x, y + 20), subtitle, font=FONT(13), fill='#8cdbb4' if active else MUTED)


def seconds(ticks, fps):
    return f'{int(ticks / fps) // 60:02d}:{(ticks / fps) % 60:05.2f}'


def score_text(score):
    return f'{score[0]}–{score[1]}'


def shot_marker(events, tick):
    for event in reversed(events):
        age = tick - event['step']
        if age < 0:
            continue
        if age > 14:
            return ''
        if event['kind'] == 'legal_serve':
            return ('SO-101' if event['side'] == 1 else 'Opponent') + ' legal serve'
        if event['kind'] == 'legal_return':
            return ('SO-101' if event['side'] == 1 else 'Opponent') + ' legal return'
    return ''


def _show(draw, score, server, ranked, total, elapsed, fps, terminal, last, result, serve_mark,
          catch, sampled, video_index, segments):
    draw.rectangle((8, 8, 1271, 58), fill='#1a252e')
    draw.text((22, 15), 'SO-101', font=FONT(26), fill=ROBOT)
    draw.text((131, 11), score_text(score), font=FONT(32), fill=INK)
    draw.text((223, 15), 'Opponent', font=FONT(26), fill=RIVAL)
    draw.text((493, 19), f'Point {ranked:02d} / {total}', font=FONT(19), fill=INK)
    server_name = 'SO-101 serve' if server == 1 else 'Opponent serve'
    draw.text((657, 19), '●  ' + server_name, font=FONT(19), fill=ROBOT if server == 1 else RIVAL)
    draw.text((1010, 19), f'Match clock   {seconds(elapsed, fps)}', font=FONT(19), fill=INK)
    frame_label(draw, (880, 72), 'POLICY WRIST RGB')
    frame_label(draw, (880, 411), 'POLICY FIXED LEFT')
    frame_label(draw, (1078, 411), 'POLICY FIXED RIGHT')
    capture_label = f'{"Captured" if sampled else "Last capture"}   {seconds(catch, fps)}'
    frame_label(draw, (880, 593), 'CAMERA FEED', capture_label, active=sampled)
    if terminal:
        winner, _, reason = result.partition(' · ')
        draw.text((880, 651), 'Match complete' if last else winner, font=FONT(20), fill=INK)
        detail = result if last and FONT(14).getlength(result) <= 390 else reason
        if detail:
            draw.text((880, 683), detail, font=FONT(14), fill=MUTED)
    else:
        draw.text((880, 652), serve_mark, font=FONT(20), fill=INK)
    length = segments[-1][1]
    for start, end, winner in segments:
        left, right = 14 + round(1250 * start / length), 14 + round(1250 * end / length)
        draw.rectangle((left, 733, max(left, right - 2), 740), fill='#344350')
        if video_index >= end - 1 or (terminal and start <= video_index < end):
            draw.rectangle((left, 733, max(left, right - 2), 740), fill=winner)
    cursor = 14 + round(1250 * video_index / max(1, length - 1))
    draw.ellipse((cursor - 5, 731, cursor + 5, 742), fill=INK)
    draw.text((14, 747), 'Opening', font=FONT(14), fill=MUTED)
    draw.text((1219, 747), 'Final', font=FONT(14), fill=MUTED)


def _composition(broadcast, images, **state):
    canvas = Image.new('RGB', VIDEO_SIZE, BACKGROUND)
    canvas.paste(Image.fromarray(broadcast), GEOMETRY[0])
    for key, shape, anchor in zip(('wrist', 'left', 'right'), ((392, 294), (192, 144), (192, 144)),
                                  GEOMETRY[1:], strict=True):
        canvas.paste(Image.fromarray(images[key]).resize(shape, Image.Resampling.NEAREST), anchor)
    _show(ImageDraw.Draw(canvas), **state)
    return canvas.tobytes()


def presentation(points, fps):
    segment = []
    cursor = 0
    for index, row in enumerate(points, 1):
        hold = round(1.2 * fps) if index == len(points) else round(.35 * fps)
        winner = ROBOT if row['result']['points_for'] else RIVAL if row['result']['points_against'] else MUTED
        end = cursor + len(row['actions']) + hold
        segment.append((cursor, end, winner))
        cursor = end
    return segment


def _encoder(path, fps):
    return subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def _observations(tracker, views, row, tick, sources):
    tracker.update(views)
    seen = tracker.ball
    position = None if seen is None or not seen.observed else list(seen.position)
    if position != row['optical'][tick] or list(tracker.sources.get('ball', ())) != sources[tick]:
        raise ValueError(f"optical observation differs on point {row['seed']} at tick {tick}")


def restore_camera_history(arena, transport, report, game):
    """Recover the camera registry carried forward from the earlier scored games."""
    prior_points = 0
    for previous in report['games']:
        if previous['seed'] == game['seed']:
            break
        for row in previous['points']:
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            tracker = transport.tracker()
            sources = dict(row['sources'])
            for tick, action in enumerate(row['actions']):
                if tick in sources:
                    _observations(tracker, arena.render_cameras(), row, tick, sources)
                arena.step(action, render=False)
            if tracker.digest.hexdigest() != row['rgb'] or arena.point_result() != row['result']:
                raise ValueError('camera history failed to reproduce a prior scored point')
            prior_points += 1
    return prior_points


def render(source, output, seed=GAME):
    report = _read(source)
    game = select_game(report, seed)
    initial_game = digest_json(game)
    points = game['points']
    output.parent.mkdir(parents=True, exist_ok=True)
    arena = SO101ViewArena(**ARENA)
    fps = round(1. / arena.control_dt)
    segments = presentation(points, fps)
    referee = ServiceScore(first_server=-1)
    schedule = np.random.default_rng(seed)
    transport = ViewTransport(ARENA['view_mode'])
    results = []
    elapsed = 0
    reader = None
    try:
        arena.reset(seed=20017)
        prior_points = restore_camera_history(arena, transport, report, game)
        spectator = mujoco.MjData(arena.model)
        reader = spectator_renderer(arena)
        with TemporaryDirectory(dir=output.parent) as temporary:
            pending = Path(temporary) / output.name
            encoder = _encoder(pending, fps)
            try:
                for index, (row, segment) in enumerate(zip(points, segments, strict=True), 1):
                    if int(schedule.integers(2**31)) != row['seed'] or referee.finished or referee.server != row['server']:
                        raise ValueError('archived point seed or service order differs')
                    ticks, sources = camera_ticks(row), dict(row['sources'])
                    tracker = transport.tracker()
                    arena.reset(seed=row['seed'], server=referee.server, render=False)
                    video_start, video_end, _ = segment
                    score_before = list(referee.score)
                    common = dict(server=row['server'], ranked=index, total=len(points), fps=fps, segments=segments)
                    views = None
                    capture = 0
                    for tick, action in enumerate(row['actions']):
                        sampled = tick in sources
                        if sampled:
                            views = arena.render_cameras()
                            if tuple(views) != STREAMS[ARENA['view_mode']]:
                                raise ValueError('native policy camera stream order changed')
                            _observations(tracker, views, row, tick, sources)
                            capture = elapsed + tick
                        elif row['optical'][tick] is not None:
                            raise ValueError('a recorded optical observation has no capture')
                        mujoco.mj_copyData(spectator, arena.model, arena.data)
                        mujoco.mj_kinematics(arena.model, spectator)
                        mujoco.mj_camlight(arena.model, spectator)
                        broadcast = arena._render_view(reader, spectator, 'tt19_broadcast')
                        encoder.stdin.write(_composition(
                            broadcast, views, score=referee.score, elapsed=elapsed + tick,
                            terminal=False, last=False, result='', serve_mark=shot_marker(row['events'], tick),
                            catch=capture, sampled=sampled, video_index=video_start + tick, **common))
                        _, _, done, timeout, _ = arena.step(action, render=False)
                        if (done or timeout) != (tick == len(row['actions']) - 1):
                            raise ValueError('action tape ended at a different control frame')
                    if tracker.digest.hexdigest() != row['rgb']:
                        raise ValueError(f"native RGB capture hash differs for point {row['seed']}")
                    result = arena.point_result()
                    if digest_json(arena.events) != digest_json(row['events']) or result != row['result']:
                        raise ValueError(f"physical outcome or events differ for point {row['seed']}")
                    referee.record(result)
                    elapsed += len(row['actions'])
                    mujoco.mj_copyData(spectator, arena.model, arena.data)
                    mujoco.mj_kinematics(arena.model, spectator)
                    mujoco.mj_camlight(arena.model, spectator)
                    broadcast = arena._render_view(reader, spectator, 'tt19_broadcast')
                    ending = RESULTS.get(result['reason'], result['reason'].replace('_', ' ').capitalize())
                    if result['points_for'] or result['points_against']:
                        winner = 'SO-101 point' if result['points_for'] else 'Opponent point'
                        ending = f'{winner} · {ending}'
                    for frame in range(video_start + len(row['actions']), video_end):
                        encoder.stdin.write(_composition(
                            broadcast, views, score=referee.score, elapsed=elapsed, terminal=True,
                            last=referee.finished, result=ending, serve_mark='', catch=capture, sampled=False,
                            video_index=frame, **common))
                    results.append(dict(attempt=index, seed=row['seed'], server=row['server'], score_before=score_before,
                                        score_after=list(referee.score), video_frame_start=video_start,
                                        video_frame_result=video_start + len(row['actions']), video_frame_end=video_end,
                                        control_frames=len(row['actions']), observation_ticks=ticks,
                                        action_sha256=sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest(),
                                        event_sha256=digest_json(row['events']), result=result, rgb_sha256=row['rgb'],
                                        verified_rgb=tracker.digest.hexdigest(), matched_optical_captures=len(ticks)))
                    print(f'Point {index:02d} {score_text(referee.score)} · {len(ticks)} RGB captures and exact physics',
                          flush=True)
            finally:
                encoder.stdin.close()
                status = encoder.wait()
            if status != 0 or not referee.finished or list(referee.score) != game['score']:
                raise ValueError('the video did not reach the archived completed score')
            current = select_game(_read(source), seed)
            if digest_json(current) != initial_game:
                raise ValueError('the archived game changed during verification')
            metadata = inspect_video(pending)
            if metadata['video_frames'] != segments[-1][1] or metadata['fps'] != fps:
                raise ValueError('the video container lost frames or changed playback speed')
            pending.replace(output)
    finally:
        if reader is not None:
            reader.close()
        arena.close()
    receipt = dict(schema='tt19-native-full-game-replay-1', source=source.name, source_sha256=_hash(source),
                   game_sha256=initial_game, seed=seed, arena=ARENA, score=game['score'], movie=output.name,
                   previous_registration_points=prior_points,
                   video_sha256=_hash(output), **metadata, controls=elapsed, original_captures=sum(
                       row['matched_optical_captures'] for row in results), verified_points=len(results),
                   legal_returns=game['returns'], legal_own_serves=game['serves'], points=results)
    path = output.with_name(output.stem + '-replay.json.gz')
    _write(path, receipt)
    return receipt


def inspect_video(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(path)]
    stream = json.loads(subprocess.check_output(command))['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    return dict(codec=stream['codec_name'], width=stream['width'], height=stream['height'],
                fps=numerator / denominator, video_frames=int(stream['nb_frames']), duration=float(stream['duration']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path, nargs='?', default=ROOT / 'match-model-rollout-native.json.gz')
    p.add_argument('--output', type=Path, default=ROOT / 'so101-full-game.mp4')
    p.add_argument('--seed', type=int, default=GAME)
    args = p.parse_args()
    receipt = render(args.source, args.output, args.seed)
    print({key: receipt[key] for key in ('score', 'duration', 'verified_points', 'original_captures')})


if __name__ == '__main__':
    main()
