"""Render the first improved wrist-only return from the paired delay games."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _write
from .tt8_service_arena import ServiceScore
from .tt20_curriculum import CHECKPOINT, OUTPUT as TT20, configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe
from .tt20_natural import NaturalViewTransport
from .tt21_game_delay import GameDelayPlayer, NominalMotion, PassiveDelay
from .tt21_game_delay_trial import play_point
from .tt21_recovery_movie import AMBER, BG, CHIP, GREEN, INK, LINE, _line, native_events


OUTPUT = ROOT / 'artifacts/tt21'
SEED = 1920024739
MODES = ('frozen', 'adaptive')
SIZE = (1056, 776)
SPECTATOR = (500, 298)
WRIST = (352, 264)
POINT_FIELDS = ('seed', 'role', 'lead', 'actions', 'applied', 'optical', 'rgb_sha256',
                'outcome', 'events', 'plans', 'reward')


def first_improved(data, game_seed):
    game = next(row for row in data['results']['wrist'] if row['seed'] == game_seed)['games']
    frozen, adaptive = (game[mode]['points'] for mode in MODES)
    for i, (base, learned) in enumerate(zip(frozen, adaptive)):
        if (i >= data['switch_after'] and base['seed'] == learned['seed'] and base['role'] == learned['role'] == -1
                and base['evaluator_delay'] == learned['evaluator_delay'] == learned['lead'] != base['lead']
                and not base['outcome']['legal_returns'] and learned['outcome']['legal_returns']
                and base['reward'] < learned['reward']):
            if any(any(a[key] != b[key] for key in POINT_FIELDS) for a, b in zip(frozen[:i], adaptive[:i], strict=True)):
                raise ValueError('the selected wrist point follows different game histories')
            return i, dict(zip(MODES, (base, learned), strict=True)), adaptive[:i]
    raise ValueError('no identical opponent-service start with a genuine improved wrist return')


def check_record(actual, recorded, fields):
    for name in fields:
        if json.dumps(actual[name]) != json.dumps(recorded[name]):
            raise ValueError(f'point {recorded["seed"]} did not reproduce {name}')


def recover_delay(data, arena, player, lenses, prefix):
    tracker = NaturalViewTransport('wrist', lens_fov=lenses, law=player.player.law)
    nominal = NominalMotion('wrist', data['arena'])
    learner = PassiveDelay(nominal)
    score = ServiceScore(first_server=-1)
    accepted = []
    try:
        for index, row in enumerate(prefix, 1):
            if row['role'] != score.server:
                raise ValueError('service order differs from archive')
            point = play_point(arena, player, tracker, 'wrist', row['seed'], row['role'],
                               row['candidate'], CommandPipe(row['evaluator_delay']), learner)
            check_record(point, row, (*POINT_FIELDS, 'measured'))
            inference = {k: v for k, v in point['learning'].items() if k != 'wall_seconds'}
            check_record({'inference': inference}, {'seed': row['seed'], 'inference': {
                k: v for k, v in row['learning'].items() if k != 'wall_seconds'}}, ('inference',))
            if inference['delay'] != inference['previous']:
                accepted.append(dict(point=index, seed=row['seed'], delay=inference['delay'],
                                     control_frames=len(row['actions']), rgb_sha256=row['rgb_sha256']))
            score.record(point['outcome'])
            if list(score.score) != row['score']:
                raise ValueError('prefix score differs from archive')
    finally:
        nominal.close()
    return tracker.state(), list(score.score), learner.delay, accepted


def capture(arena, spectator, player, state, lenses, row):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    player.serves.force = row['candidate']
    player.reset(row['role'])
    tracker = NaturalViewTransport('wrist', state=state, lens_fov=lenses, law=player.player.law).tracker()
    queue = CommandPipe(row['evaluator_delay'])
    frames, hashes = [], []

    def sample(camera):
        view = Image.fromarray(arena.render_broadcast(spectator))
        wrist = Image.fromarray(camera).resize(WRIST, Image.Resampling.LANCZOS)
        return view, wrist

    for tick, (expected, applied, optical) in enumerate(zip(row['actions'], row['applied'], row['optical'], strict=True)):
        images = arena.render_cameras()
        if tuple(images) != ('wrist',):
            raise ValueError('unexpected policy camera')
        camera = images['wrist']
        hashes.append(sha256(camera).hexdigest())
        frames.append(sample(camera))
        tracker.update(images, tick=tick)
        ball = tracker.ball
        xyz = None if ball is None or not ball.observed else list(map(float, ball.position))
        request = player.command(tick, xyz, estimated_velocity=tracker.ball_velocity)
        action = queue.apply(request)
        if xyz != optical or request.tolist() != expected or action.tolist() != applied:
            raise ValueError(f'optics or motor command changed at frame {tick}')
        _, reward, ended, capped, _ = arena.step(action, render=False)
        if bool(ended or capped) != (tick == len(row['actions']) - 1):
            raise ValueError(f'physical point ended at the wrong frame {tick}')
    check_record(dict(rgb_sha256=tracker.digest.hexdigest(), outcome=arena.point_result(),
                      events=arena.events, plans=player.player.plans, reward=reward),
                 row, ('rgb_sha256', 'outcome', 'events', 'plans', 'reward'))
    frames.append(sample(arena.render_cameras()['wrist']))
    return frames, hashes


def difference(first, second):
    return next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)


def decorate(frames, rows, score_before, tick, chronology):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _line(draw, (20, 12), 'Learning the four-frame motor delay', size=30, color=INK, bold=True)
    _line(draw, (22, 55), 'Same archived opponent serve · Same physical delay · Half-speed replay', size=16)
    draw.line((20, 82, 1036, 82), fill=LINE)
    for mode, x, accent, heading in (('frozen', 20, AMBER, 'Frozen · 0 control frames'),
                                     ('adaptive', 536, GREEN, 'Learned · 4 control frames')):
        row = rows[mode]
        index = min(tick, len(row['actions']))
        ended = index == len(row['actions'])
        view, wrist = frames[mode][index]
        _line(draw, (x, 95), heading, size=23, color=accent, bold=True)
        _line(draw, (x, 127), 'Spectator · replay only', size=15)
        canvas.paste(view, (x, 149))
        _line(draw, (x, 453), 'Wrist RGB · policy input', size=15, color=INK)
        canvas.paste(wrist, (x, 474))
        draw.rounded_rectangle((x + 362, 474, x + 500, 737), radius=7, fill=CHIP)
        returns = [n for n in native_events(row, 'legal_return', 1) if n <= index]
        contacts = [n for n in native_events(row, 'contact', 1) if n <= index]
        status = ('Own-half bounce', 'Point lost') if ended and row['reward'] < 0 else (
            ('Legal return', 'Point won') if ended else
            ('Legal return', f'Frame {returns[0]}') if returns else
            ('Paddle contact', f'Frame {contacts[0]}') if contacts else
            ('Opponent serves', ''))
        bx = x + 373
        _line(draw, (bx, 487), f'Frame {index}', size=15)
        _line(draw, (bx, 524), 'Game score', size=15)
        score = row['score'] if ended else score_before
        _line(draw, (bx, 545), f'{score[0]}–{score[1]}', size=27, color=INK, bold=True)
        _line(draw, (bx, 593), f'{len(returns)} legal return{"s" if len(returns) != 1 else ""}',
              size=15, color=accent, bold=True)
        _line(draw, (bx, 642), status[0], size=14, color=INK)
        _line(draw, (bx, 665), status[1], size=15, color=accent, bold=True)
    _line(draw, (20, 751), 'Point 5: delay accepted from wrist RGB', size=14)
    _line(draw, (340, 751), 'Point 6: requests differ at {requested} · motors at {applied} · wrist RGB at {wrist_rgb}'
          .format(**chronology), size=14)
    return canvas


def encode(path, frames, rows, score_before, chronology, fps):
    count = max(len(sequence) for sequence in frames.values())
    opening, closing = 22, 45
    with subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                           '-preset', 'fast', '-threads', '1', '-crf', '20', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE) as encoder:
        for tick in range(count):
            pixels = decorate(frames, rows, score_before, tick, chronology).tobytes()
            repeats = opening if tick == 0 else closing if tick == count - 1 else 2
            for _ in range(repeats):
                encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('video encoding failed')
    video = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                     'stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt',
                                     '-of', 'json', str(path)])
    stream, = json.loads(video)['streams']
    expected = opening + 2 * (count - 2) + closing
    if (stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p' or
            (stream['width'], stream['height']) != SIZE or int(stream['nb_frames']) != expected):
        raise ValueError('unexpected encoded video')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
                    '-f', 'null', '-'], check=True)
    return dict(stream, playable_seconds=expected / fps, sha256=_hash(path), bytes=path.stat().st_size)


def render(source, destination, receipt, game_seed=SEED):
    snapshot = source.read_bytes()
    data = json.loads(gzip.decompress(snapshot))
    index, rows, prefix = first_improved(data, game_seed)
    if (not data['arena']['strict_endline'] or data['arena']['marked']
            or _hash(CHECKPOINT) != data['source_sha256']
            or _hash(TT20 / 'curriculum-dry-catalog.npz') != data['atlas_sha256']
            or _hash(TT20 / 'curriculum-factory.json.gz') != data['factory_sha256']):
        raise ValueError('source factory checkpoint or apparatus differs')
    programs, catalogue, law, response, _ = configuration(TT20)
    player = GameDelayPlayer(programs, catalogue, law, response, model_arena())
    arena = TT20Arena(**data['arena'], view_mode='wrist')
    spectator = mujoco.Renderer(arena.model, width=SPECTATOR[0], height=SPECTATOR[1])
    fps = round(1 / arena.control_dt)
    frames, hashes = {}, {}
    try:
        lenses = {'wrist': arena.lens_intrinsics()['wrist']}
        state, score_before, learned, accepted = recover_delay(data, arena, player, lenses, prefix)
        if learned != rows['adaptive']['lead'] or not accepted:
            raise ValueError('the lead was not independently relearned in the point history')
        if any(tuple(np.add(score_before, (row['outcome']['points_for'], row['outcome']['points_against'])))
               != tuple(row['score']) for row in rows.values()):
            raise ValueError('selected point score does not match the verified game prefix')
        for mode, row in rows.items():
            player.lead = learned if mode == 'adaptive' else 0
            if player.lead != row['lead']:
                raise ValueError('replay lead differs from archive')
            frames[mode], hashes[mode] = capture(arena, spectator, player, state, lenses, row)
    finally:
        spectator.close()
        arena.close()
        player.player.close()
    frozen, adaptive = (rows[mode] for mode in MODES)
    chronology = {name: difference(frozen[key], adaptive[key]) for name, key in
                  (('requested', 'actions'), ('applied', 'applied'), ('ball_position', 'optical'))}
    chronology['wrist_rgb'] = difference(*(hashes[mode] for mode in MODES))
    if (chronology['applied'] != chronology['requested'] + data['evaluator_delay']
            or not chronology['applied'] < chronology['wrist_rgb'] <= chronology['ball_position']):
        raise ValueError('order of motor and optical divergence changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        movie = encode(temporary, frames, rows, score_before, chronology, fps)
        temporary.replace(destination)
    branches = {mode: dict(lead=row['lead'], score=row['score'],
                           requested_sha256=sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest(),
                           applied_sha256=sha256(np.asarray(row['applied'], dtype='<f8').tobytes()).hexdigest(),
                           source_row_sha256=sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                           rgb_sha256=row['rgb_sha256'], verified_frames=len(hashes[mode]),
                           learner_contacts=native_events(row, 'contact', 1),
                           legal_returns=native_events(row, 'legal_return', 1), outcome=row['outcome'], reward=row['reward'])
                for mode, row in rows.items()}
    _write(receipt, dict(schema='tt21-game-delay-video-1', source=source.name, source_snapshot_sha256=sha256(snapshot).hexdigest(),
                         selection='earliest actual matched improved wrist return', game_seed=game_seed,
                         point=index + 1, seed=frozen['seed'], server=frozen['role'], score_before=score_before,
                         policy_streams=['wrist'], spectator_excluded_from_policy=True, replayed_game_prefix=len(prefix),
                         verified_delay_updates=accepted, evaluator_delay=data['evaluator_delay'],
                         first_different_frame=chronology, branches=branches, movie=movie,
                         renderer_sha256=_hash(Path(__file__))))
    print(destination, movie, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT / 'game-delay.json.gz')
    parser.add_argument('--destination', type=Path, default=OUTPUT / 'game-delay-wrist.mp4')
    parser.add_argument('--receipt', type=Path, default=OUTPUT / 'game-delay-wrist.json')
    args = parser.parse_args()
    render(args.source, args.destination, args.receipt)
