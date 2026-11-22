"""Render a matched old-rate and learned-rate replay of the remapped SO-101."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt21_recovery_movie import AMBER, BG, CHIP, GREEN, INK, LINE, MUTED, _line
from .tt22_learned_body import CONFIG, JitteredOpening, OUTPUT, Wiring, digest_json, replay_point


SIZE = (936, 408)
COURT = (444, 254)
PANELS = (('unadapted', 16, AMBER, 'Unadapted port commands'),
          ('learned', 476, GREEN, 'Wrist-learned port commands'))


def picture(rows, frames, tick, seed, practice, resets):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _line(draw, (16, 11), 'SO-101 rallies after learning its motor wiring', size=27, color=INK, bold=True)
    _line(draw, (17, 47), f'Held launch {seed}  ·  Same remapped arm, stereo controller and opponent', size=15)
    draw.line((16, 72, 920, 72), fill=LINE)
    for branch, x, accent, label in PANELS:
        row = rows[branch]
        index = min(tick, len(row['actions']))
        _line(draw, (x, 80), label, size=19, color=accent, bold=True)
        canvas.paste(frames[branch][index], (x, 109))
        near, far = (sum(event['kind'] == 'legal_return' and event['side'] == side and event['step'] <= index
                         for event in row['events']) for side in (1, -1))
        draw.rounded_rectangle((x + 7, 116, x + 242, 144), radius=5, fill=CHIP)
        _line(draw, (x + 16, 122), f'{near + far} legal returns  ·  Arm {near}  Rival {far}', size=13, color=INK)
        if index == len(row['actions']):
            result = {1: 'Point won by SO-101', -1: 'Point won by rival', 0: 'Point timed out'}[row['reward']]
            draw.rounded_rectangle((x + 241, 328, x + 437, 355), radius=5, fill=BG)
            _line(draw, (x + 252, 332), result, size=14, color=accent, bold=True)
    _line(draw, (16, 379), f'Calibration  ·  {practice} target control frames across {resets} probe resets  ·  Wrist RGB',
          size=15, color=MUTED)
    _line(draw, (728, 379), 'Spectator replay · real time', size=14, color=MUTED)
    return canvas


def encode(path, rows, frames, seed, practice, resets, rate):
    count = max(map(len, frames.values()))
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', 'x'.join(map(str, SIZE)), '-r', str(rate), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-threads', '1', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for tick in range(count):
            pixels = picture(rows, frames, tick, seed, practice, resets).tobytes()
            repeats = rate if tick in (0, count - 1) else 1
            for _ in range(repeats):
                encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('movie encoding failed')
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'stream=codec_name,width,height,nb_frames', '-of', 'json', str(path)])
    stream, = json.loads(raw)['streams']
    expected = count - 2 + 2 * rate
    if (stream['codec_name'] != 'h264' or (stream['width'], stream['height']) != SIZE
            or int(stream['nb_frames']) != expected):
        raise ValueError('the movie format or frame count changed')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
                    '-f', 'null', '-'], check=True)
    return dict(stream, seconds=expected/rate, sha256=_hash(path), bytes=path.stat().st_size)


def render(source, probes, output, receipt):
    data, acquisition = _read(source), _read(probes)
    if data['probe_sha256'] != _hash(probes):
        raise ValueError('the saved probe tape does not match the scored installation')
    seed = data['rgb_witness']['seed']
    rows = {name: next(row for row in data['branches'][name] if row['seed'] == seed) for name, _, _, _ in PANELS}
    wiring = Wiring.restore(data['evaluator_installation'])
    correction = Wiring.restore(data['installed_correction'])
    arena = JitteredOpening(**CONFIG)
    spectator = mujoco.Renderer(arena.model, width=COURT[0], height=COURT[1])
    frames = {name: [] for name in rows}
    rate = round(1 / arena.control_dt)
    try:
        for name, row in rows.items():
            def capture(scene):
                frames[name].append(Image.fromarray(scene.render_broadcast(spectator)))

            replay_point(arena, row, wiring, correction if name == 'learned' else None, rgb=True, observe=capture)
    finally:
        spectator.close()
        arena.close()
    practice, resets = acquisition['target_frames'], acquisition['target_resets']
    with TemporaryDirectory(dir=output.parent) as folder:
        temporary = Path(folder)/output.name
        movie = encode(temporary, rows, frames, seed, practice, resets, rate)
        temporary.replace(output)
    _write(receipt, dict(schema='tt22-learned-body-paired-movie-1', seed=seed, score_sha256=_hash(source),
                         practice_frames=practice, probe_resets=resets, rgb_and_physics_verified=True, movie=movie,
                         branches={name: dict(row_sha256=digest_json(row), longest=row['longest'],
                                              own_returns=row['outcome']['legal_returns'], reward=row['reward'],
                                              issued_public_sha256=row['public_sha256'], rgb_sha256=row['rgb'])
                                   for name, row in rows.items()}))
    print(output, movie, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    args = parser.parse_args()
    name = args.folder/'tt22_learned_body'
    render(name.with_name(name.name + '_held.json.gz'), name.with_name(name.name + '_probes.json.gz'),
           name.with_suffix('.mp4'), name.with_name(name.name + '_movie.json'))
