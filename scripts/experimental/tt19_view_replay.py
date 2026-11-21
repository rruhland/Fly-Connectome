"""Audit RGB observations and produce a native video of verified SO-101 game tapes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt18_view import CameraFit
from .tt18_view_trial import configure_view_colors
from .tt19_so101 import TIP, WRIST_CAMERA
from .tt19_view import MODES, STREAMS, WRIST, ViewTransport
from .tt19_view_trial import CAMERAS, SO101ViewArena


ORIGIN = np.array((0., 0., TABLE_HEIGHT))
ALL_STREAMS = (WRIST, 'left', 'right', 'side')
VIDEO_SIZE = (1120, 624)
TIP_VISIBILITY_PX = 10.
BLADE_VISIBILITY_PX = 27.


def _make(report, settings=None):
    if report['schema'] == 'tt19-so101-online-v1':
        from .tt19_online_trial import SO101Wiring

        spec = report['fixture']
        args = dict(spec['arena'], **(settings or {}))
        arena = SO101ViewArena(**args, plant=spec.get('plant'))
        return arena, SO101Wiring(arena, spec['layout_seed'])
    if report['schema'] == 'tt19-direct-stroke-v1':
        from .tt19_stroke_trial import IdentityWiring, _arena

        arena = _arena(dict(report['arena'], **(settings or {})), report['cameras'], report['custom_table'])
        return arena, IdentityWiring(arena)
    raise ValueError('expected a TT19 scored online or direct-stroke game receipt')


def _game(report, game_index):
    if 'games' in report:
        game = report['games'][game_index]
        if not game['finished']:
            raise ValueError('the source receipt does not contain a completed game')
        return game
    if not report.get('game_completed') or game_index != 0:
        raise ValueError('the source receipt does not contain that completed game')
    return dict(seed=report['test'][0]['game_seed'], points=report['test'], score=report['game_score'])


def _points(game, clip_count):
    rows = game['points']
    strongest = max(range(len(rows)), key=lambda i: (rows[i]['evaluator']['legal_returns'],
                                                     rows[i]['evaluator']['contacts'], len(rows[i]['actions'])))
    serve = max((i for i, row in enumerate(rows) if row['server'] == 1),
                key=lambda i: (rows[i]['evaluator']['legal_serve'], rows[i]['evaluator']['service_contact'],
                               rows[i]['evaluator']['legal_returns']), default=strongest)
    receive = max((i for i, row in enumerate(rows) if row['server'] == -1),
                  key=lambda i: (rows[i]['evaluator']['legal_returns'], rows[i]['evaluator']['contacts']),
                  default=strongest)
    score = ServiceScore(first_server=-1)
    selected = list(dict.fromkeys((strongest, serve, receive, *range(len(rows)))))[:clip_count]
    clips = []
    for index, row in enumerate(rows):
        if index in selected:
            clips.append((index, score.score, row))
        score.record(row['evaluator'])
    if list(score.score) != game['score'] or not score.finished:
        raise ValueError('the archived score is inconsistent')
    return clips


def _interface(wiring, game_seed, row):
    disturbance = (prepare_point(wiring.arena, row['phase']),) if 'phase' in row else ()
    return PointInterface(wiring, *disturbance, server=row['server'], seed=game_seed)


def _renderer_camera(arena):
    model = arena.model.camera(WRIST_CAMERA)
    camera = arena.data.camera(WRIST_CAMERA)
    orientation = np.diag((1., -1., -1.)) @ camera.xmat.reshape(3, 3).T
    focal = arena.height / (2 * np.tan(np.deg2rad(model.fovy[0]) / 2))
    return CameraFit(camera.xpos.copy() - ORIGIN, orientation, focal, arena.width, arena.height)


def _capture(arena):
    configure_view_colors(arena)
    renderer = arena._view_renderer()
    mujoco.mj_kinematics(arena.model, arena.data)
    mujoco.mj_camlight(arena.model, arena.data)
    images, elapsed = {}, {}
    for stream in ALL_STREAMS:
        camera = WRIST_CAMERA if stream == WRIST else CAMERAS[stream][0]
        began = perf_counter()
        images[stream] = arena._render_view(renderer, arena.data, camera)
        elapsed[stream] = 1000. * (perf_counter() - began)
    return images, elapsed


def _projected_error(tracker, role, camera, true_point):
    blob = tracker.detections[role].get(WRIST)
    if blob is None:
        return None
    projected = camera.project(true_point)
    if projected[2] <= 0.:
        return None
    return float(np.linalg.norm(blob.center - projected[:2]))


def _tracked_error(track, true_point):
    return None if track is None else float(np.linalg.norm(track.position - true_point))


def _audit(tracker, images, elapsed, arena, mode, evaluator):
    requested = STREAMS[mode]
    streams = {stream: images[stream] for stream in requested}
    began = perf_counter()
    tracker.update(streams)
    wall_ms = 1000. * (perf_counter() - began)
    ball = arena.ball.copy() - ORIGIN
    blade = arena.paddle.copy() - ORIGIN
    tip = arena.data.site(TIP).xpos.copy() - ORIGIN
    true_normal = arena.data.geom('learner_racket').xmat.reshape(3, 3)[:, 2]
    normal = tracker.actor_normal
    normal_error = (None if normal is None else
                    float(np.rad2deg(np.arccos(np.clip(abs(true_normal @ normal), 0., 1.)))))
    camera = tracker.fits.get(WRIST)
    return dict(sightings=tracker.sightings[-1], streams=tracker.sources,
                blade_m=_tracked_error(tracker.actor, blade), ball_m=_tracked_error(tracker.ball, ball),
                blade_normal_deg=normal_error,
                wrist_eye_m=None if camera is None else float(np.linalg.norm(camera.eye - evaluator.eye)),
                wrist_tip_px=_projected_error(tracker, 'tip', evaluator, tip),
                wrist_blade_px=_projected_error(tracker, 'blade', evaluator, blade),
                wrist_ball_px=_projected_error(tracker, 'ball', evaluator, ball),
                render_ms=sum(elapsed[key] for key in requested), track_ms=wall_ms)


def _font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default(size=size)


def _video_frame(broadcast, wrist, fixed, score, index, row, tick, wrist_live):
    canvas = Image.new('RGB', VIDEO_SIZE, '#11161c')
    canvas.paste(Image.fromarray(broadcast).resize((768, 576)), (0, 48))
    canvas.paste(Image.fromarray(wrist).resize((336, 252)), (776, 48))
    canvas.paste(Image.fromarray(fixed).resize((336, 252)), (776, 348))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 13), f'SO-101 table tennis   {score[0]}–{score[1]}', font=_font(24), fill='#eef3f6')
    service = 'Robot serve' if row['server'] == 1 else 'Opponent serve'
    draw.text((486, 17), f'Point {index + 1} · {service} · {tick / 30:.1f}s',
              font=_font(18), fill='#c2cfd8')
    draw.text((778, 312), 'Wrist RGB' if wrist_live else 'Wrist RGB · evaluator view',
              font=_font(18), fill='#eef3f6')
    draw.text((778, 602), 'Fixed camera', font=_font(16), fill='#eef3f6')
    return canvas.tobytes()


def _writer(path, fps):
    return subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}', '-framerate', f'{fps:.7g}', '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def _physical_match(interface, row):
    arena = interface.arena
    if (interface.applied != row['applied'] or arena.point_result() != row['evaluator']
            or json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True)):
        raise ValueError(f"the native replay diverged on point {row['seed']}")
    if 'action_sha256' in row:
        digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
        if digest != row['action_sha256']:
            raise ValueError(f"the action tape digest changed on point {row['seed']}")


def _sample_ranges(values):
    if not values:
        return dict(count=0, median=None, p90=None)
    median, p90 = np.quantile(values, (.5, .9))
    return dict(count=len(values), median=float(median), p90=float(p90))


def _visible(row):
    tip, blade = row['wrist_tip_px'], row['wrist_blade_px']
    return tip is not None and tip < TIP_VISIBILITY_PX and blade is not None and blade < BLADE_VISIBILITY_PX


def summarize(clips):
    summary = {}
    for mode in MODES:
        data = [row[mode] for clip in clips for row in clip['audit']]
        result = dict(captures=len(data), wrist_registered=sum(row['sightings'][6] for row in data),
                      wrist_ball_used=sum(WRIST in row['streams'].get('ball', ()) for row in data),
                      wrist_tip_and_blade=sum(map(_visible, data)),
                      blade_sources=dict(Counter(row['streams']['blade_pose'] for row in data
                                                 if 'blade_pose' in row['streams'])))
        for key in ('blade_m', 'ball_m', 'blade_normal_deg', 'wrist_eye_m', 'wrist_tip_px',
                    'wrist_blade_px', 'wrist_ball_px', 'render_ms', 'track_ms'):
            result[key] = _sample_ranges([row[key] for row in data if row[key] is not None])
        result['sensor_plus_tracker_ms'] = _sample_ranges([row['render_ms'] + row['track_ms'] for row in data])
        summary[mode] = result
    return summary


def run(source, folder, game_index, clip_count):
    report = _read(source)
    game = _game(report, game_index)
    source_view = report.get('view', report.get('cameras'))
    folder.mkdir(parents=True, exist_ok=True)
    movie = folder / 'view-game.mp4'
    pending = movie.with_name('view-game-pending.mp4')
    arena, wires = _make(report)
    display, display_wires = _make(report, {'detail': 'cad', 'width': 768, 'height': 576})
    writer = _writer(pending, 1 / arena.control_dt)
    transports = {mode: ViewTransport(mode) for mode in MODES}
    results, seen, still = [], 0, None
    try:
        for index, score, row in _points(game, clip_count):
            interface = _interface(wires, game['seed'], row)
            interface.reset(seed=row['seed'], render=False)
            spectator = _interface(display_wires, game['seed'], row)
            spectator.reset(seed=row['seed'], render=False)
            trackers = {mode: transport.tracker() for mode, transport in transports.items()}
            frames, camera_hash = [], sha256()
            for tick, action in enumerate([None, *row['actions']]):
                if action is not None:
                    interface.step(action, render=False)
                    spectator.step(action, render=False)
                images, elapsed = _capture(arena)
                evaluator = _renderer_camera(arena)
                frames.append({mode: _audit(trackers[mode], images, elapsed, arena, mode, evaluator)
                               for mode in MODES})
                if tick % report.get('capture_stride', 1) == 0:
                    for stream in STREAMS[source_view]:
                        camera_hash.update(stream.encode())
                        camera_hash.update(images[stream])
                measured = frames[-1]['wrist+fixed']
                if still is None and _visible(measured):
                    Image.fromarray(images[WRIST]).save(folder / 'view-wrist.png')
                    still = dict(attempt=index + 1, tick=tick, sha256=_hash(folder / 'view-wrist.png'))
                mujoco.mj_kinematics(display.model, display.data)
                mujoco.mj_camlight(display.model, display.data)
                configure_view_colors(display)
                broadcast = display.render_broadcast()
                preview = images['left' if source_view != 'wrist+side' else 'side']
                writer.stdin.write(_video_frame(broadcast, images[WRIST], preview, score, index, row, tick,
                                                WRIST in STREAMS[source_view]))
                seen += 1
            _physical_match(interface, row)
            _physical_match(spectator, row)
            reference = row.get('vision_digest', row.get('observation_sha256'))
            digest = camera_hash.hexdigest()
            results.append(dict(attempt=index + 1, seed=row['seed'], score_before=list(score), rgb_sha256=digest,
                                reference_rgb_sha256=reference, original_rgb_reproduced=digest == reference,
                                legal_returns=row['evaluator']['legal_returns'],
                                legal_serve=row['evaluator']['legal_serve'], audit=frames))
            print('point', index + 1, 'frames', len(frames), 'physical replay exact',
                  'original RGB', digest == reference, flush=True)
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        status = writer.wait()
        arena.close()
        display.close()
    if status != 0 or not seen:
        raise RuntimeError('the native video did not encode')
    pending.replace(movie)
    result = dict(schema='tt19-so101-view-audit-v1', source=source.name, source_sha256=_hash(source),
                  game=game_index + 1, score=game['score'], source_view=source_view, video=movie.name,
                  video_sha256=_hash(movie), video_frames=seen, fps=1 / arena.control_dt,
                  wrist_still=still, visibility_tolerance_px=dict(tip=TIP_VISIBILITY_PX, blade=BLADE_VISIBILITY_PX),
                  exact_physics_replays=len(results), views=summarize(results), clips=results)
    _write(folder / 'view-audit.json.gz', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt19'))
    parser.add_argument('--game', type=int, default=1)
    parser.add_argument('--clips', type=int, default=3)
    args = parser.parse_args()
    if args.game < 1 or args.clips < 1:
        parser.error('game and clip count must be positive')
    result = run(args.receipt, args.output, args.game - 1, args.clips)
    print(json.dumps(result['views'], indent=2))


if __name__ == '__main__':
    main()
