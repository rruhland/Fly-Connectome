"""Reissue the exported wrist pupil's commands and show its paired held drop."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED


ROOT = Path('artifacts/tt24')
EVIDENCE = ROOT / 'bounce-ident.json.gz'
CHECKPOINT = ROOT / 'bounce-ident-pupil.npz'
MOVIE = ROOT / 'wrist-bounce-video.mp4'
RECEIPT = ROOT / 'wrist-bounce-video.json'
SEED = 247202
TITLES = {'wrist_fitted_motion': 'Live wrist pupil', 'stationary': 'Stationary paddle'}
PANEL = (384, 288)
CANVAS = (812, 722)
SPECTATOR_CROP = (36, 26, 164, 122)
STRIDE = 2
HOLD = 22


@dataclass(frozen=True)
class _Frame:
    wrist: np.ndarray
    spectator: np.ndarray
    consecutive: int
    terminal: str | None


def _reissue(arena, reference, pupil):
    obs = arena.reset(seed=reference['seed'], split=reference['split'], regime=reference['regime'])
    recorded = np.asarray(reference['commands'], dtype='<f8')
    actions = sha256()
    wrist_rgb = sha256()
    frames = {}
    done = False
    reward = 0.
    for tick, expected in enumerate(recorded):
        wrist_rgb.update(obs['rgb'])
        action = pupil.command({'rgb': obs['rgb'], 'joints': obs['joints']}) if pupil else np.zeros(arena.action_size)
        action_bytes = action.astype('<f8').tobytes()
        if action_bytes != expected.tobytes():
            raise ValueError(f'exported rates differ from the held tape at step {tick} on seed {reference["seed"]}')
        actions.update(action_bytes)
        if tick % STRIDE == 0:
            metrics = arena.judge_metrics()
            frames[tick] = _Frame(obs['rgb'], arena.render_external(), metrics['longest_consecutive_rebounds'], None)
        obs, score, done = arena.step(action)
        reward += score
        if bool(done) != (tick + 1 == len(recorded)):
            raise ValueError(f'replay changed the terminal step on seed {reference["seed"]}')
    metrics = arena.judge_metrics()
    expected = {key: reference[key] for key in metrics}
    if not done or reward != reference['reward'] or digest_json(metrics) != digest_json(expected):
        raise ValueError(f'replay changed the physical or referee results on seed {reference["seed"]}')
    frames[len(recorded)] = _Frame(obs['rgb'], arena.render_external(), metrics['longest_consecutive_rebounds'],
                                   metrics['termination'])
    proof = dict(command_sha256=actions.hexdigest(), native_wrist_rgb_sha256=wrist_rgb.hexdigest(),
                 commands_match_exactly=True, physical_and_referee_fields_match_exactly=True, metrics=metrics)
    return frames, proof


def _make_canvas(tick, branches, total_seconds, typefaces):
    heading, title, small, score_font = typefaces
    canvas = Image.new('RGB', CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 9), 'Wrist-camera paddle bouncing', font=heading, fill=INK)
    draw.text((17, 39), f'Held drop {SEED}  ·  Four supervised training demonstrations', font=small, fill=MUTED)
    draw.text((CANVAS[0] - 16, 15), f'{total_seconds:.2f} s', anchor='ra', font=small, fill=MUTED)
    finished = {'timeout': 'Five-second limit reached', 'non_paddle_contact': 'Ball touched the table'}
    for col, (mode, frames) in enumerate(branches.items()):
        frame = frames[min(tick, max(frames))]
        x = 16 + col * (PANEL[0] + 12)
        draw.text((x, 65), TITLES[mode], font=title, fill=INK)
        draw.text((x + PANEL[0], 65), f'{frame.consecutive} consecutive', anchor='ra', font=score_font,
                  fill=ACCENT if mode == 'wrist_fitted_motion' else MUTED)
        spectator = Image.fromarray(frame.spectator).crop(SPECTATOR_CROP).resize(PANEL, Image.Resampling.BILINEAR)
        wrist = Image.fromarray(frame.wrist).resize(PANEL, Image.Resampling.BILINEAR)
        canvas.paste(spectator, (x, 87))
        draw.text((x, 378), 'Spectator', font=small, fill=MUTED)
        draw.text((x, 700), 'Onboard wrist', font=small, fill=MUTED)
        if frame.terminal:
            draw.text((x + PANEL[0], 378), finished[frame.terminal], anchor='ra', font=small, fill=MUTED)
        canvas.paste(wrist, (x, 408))
    return canvas.tobytes()


def _encode(branches, fps, dt):
    last = max(max(frames) for frames in branches.values())
    ticks = list(range(0, last + 1, STRIDE))
    ticks.extend([last] * HOLD)
    typefaces = tuple(ImageFont.truetype(path, size) for path, size in
                      ((BOLD, 24), (BOLD, 16), (FONT, 13), (BOLD, 14)))
    with TemporaryDirectory(dir=MOVIE.parent, prefix='wrist-bounce-video-') as directory:
        pending = Path(directory) / MOVIE.name
        arguments = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                     '-s', f'{CANVAS[0]}x{CANVAS[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                     '-threads', '1', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                     '-movflags', '+faststart', str(pending)]
        with subprocess.Popen(arguments, stdin=subprocess.PIPE) as encoder:
            for tick in ticks:
                encoder.stdin.write(_make_canvas(tick, branches, tick * dt, typefaces))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the wrist bounce replay')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(pending),
                        '-f', 'null', '-'], check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_frames', '-of', 'json', str(pending)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != len(ticks):
            raise ValueError('the encoded replay is missing frames')
        pending.replace(MOVIE)
    return len(ticks)


def main():
    evidence = _read(EVIDENCE)
    examples = evidence['demo_seeds']
    if len(examples) != 4 or SEED in examples or SEED not in evidence['held_seeds']:
        raise ValueError('expected a fresh held witness after four demonstrations')
    reference = {mode: next(row for row in evidence['cohorts'][mode]['episodes'] if row['seed'] == SEED)
                 for mode in TITLES}
    if (reference['wrist_fitted_motion']['longest_consecutive_rebounds'],
            reference['stationary']['longest_consecutive_rebounds']) != (14, 3):
        raise ValueError('held witness and stationary scores changed')
    conditions = [(row['seed'], row['split'], row['regime'], row['arm'], row['drop']) for row in reference.values()]
    if conditions[0] != conditions[1] or conditions[0][:4] != (SEED, 'held', 'standard', 'stock'):
        raise ValueError('expected two stock-body replays of the same held drop')
    arena = BounceArena(view='wrist', width=evidence['width'], height=evidence['height'])
    frozen = WristPupil.load(CHECKPOINT)
    branches, proofs = {}, {}
    try:
        for mode, row in reference.items():
            pupil = frozen if mode == 'wrist_fitted_motion' else None
            branches[mode], proofs[mode] = _reissue(arena, row, pupil)
        fps = round(1. / (arena.dt * STRIDE))
        frames = _encode(branches, fps, arena.dt)
    finally:
        arena.close()
    _write(RECEIPT, dict(schema='tt24-verified-wrist-bounce-movie-1', seed=SEED, split='held', regime='standard',
                         arm='stock', source=EVIDENCE.name, source_sha256=_hash(EVIDENCE),
                         checkpoint=CHECKPOINT.name, checkpoint_sha256=_hash(CHECKPOINT),
                         movie=MOVIE.name, video_sha256=_hash(MOVIE), fps=fps, frames=frames,
                         terminal_hold_frames=HOLD, input_wrist_dimensions=[arena.width, arena.height],
                         display_spectator_crop=SPECTATOR_CROP,
                         motor_training=evidence['teacher'], supervised_demo_count=len(examples), demo_seeds=examples,
                         perception_training=evidence['perception_training'], training_performed_for_video=False,
                         pupil_runtime=evidence['sighted_runtime'], spectator_used_by_pupil=False,
                         branches=proofs, video_decode_verified=True))
    print(f'Verified held drop {SEED}: 14 live versus 3 stationary, {frames} video frames', flush=True)


if __name__ == '__main__':
    main()
