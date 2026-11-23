"""Show the same changed-wiring release with and without the bounce decoder."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import BG, GREEN, INK, MUTED, font
from . import tt23_sequence_coach_movie as movie
from .tt23_sequence import CONFIG, SequenceArena
from .tt24_rally_bank import CACHE, DATA_RECEIPT
from .tt24_rally_policy import EYES
from .tt24_rally_remap import driver
from .tt24_rally_trial import alternating


ROOT = Path('artifacts/tt24')
SOURCE = ROOT / 'bounce-to-match.json.gz'
COMPARISON = ROOT / 'match-bridge-changed.mp4'
NATIVE = ROOT / 'match-bridge-native.mp4'
CHANGED_SEED = 1821600722
NATIVE_SEED = 1248661335
PANEL = (476, 272)
CANVAS = (984, 410)
HOLD = 18
LABELS = {'bounce': 'Calibrated on five bounce drops', 'frozen': 'Frozen nominal motor map'}


def _inputs():
    course = _read(SOURCE)
    training = _read(DATA_RECEIPT)
    calibration_path = ROOT / course['source']
    calibration = _read(calibration_path)
    if (len(course['cases']) != 24 or course['verified'] != 96 or training['demos'] != 48 or
            training['samples_sha256'] != _hash(CACHE) or course['target_experience'] != calibration['target_cost'] or
            calibration['target_cost']['drops'] != 5 or course['table_tennis_exposure_after_adaptation'] != 0):
        raise ValueError('match bridge source, exposures or verified cohort changed')
    exposure = dict(table_tennis_demonstration_points=training['demos'],
                    table_tennis_demo_frames_used=training['fit_samples'],
                    table_tennis_practice_on_changed_wiring=course['table_tennis_exposure_after_adaptation'],
                    changed_wiring_bounce_calibration=calibration['target_cost'],
                    earlier_nominal_bounce_calibration=calibration['source_cost'],
                    calibration_file=calibration_path.name, calibration_sha256=_hash(calibration_path),
                    training_manifest_sha256=_hash(DATA_RECEIPT), training_samples_sha256=training['samples_sha256'])
    return course, calibration, training['dimensions'], exposure


def _match_setup(row, matrix, decoder):
    recorded = np.asarray(row['actions'], dtype='<f8')
    transformed = np.asarray([driver(matrix, decoder, np.asarray(rate)) for rate in row['requested_actions']],
                             dtype='<f8')
    if transformed.tobytes() != recorded.tobytes():
        raise ValueError('the issued tape disagrees with this wiring and decoder')
    return sha256(recorded.tobytes()).hexdigest()


def _capture(row, dimensions):
    arena = SequenceArena(**dict(CONFIG, width=dimensions[0], height=dimensions[1]))
    stereo, prefix = sha256(), sha256()
    frames = []
    length = len(row['actions'])
    try:
        arena.lane = row['lane']
        arena.reset(seed=row['seed'], server=-1, render=False)
        with mujoco.Renderer(arena.model, width=movie.VIEW[0], height=movie.VIEW[1]) as spectator:
            def panel():
                image = Image.fromarray(arena.render_broadcast(spectator)).crop((74, 44, 550, 316))
                return image, alternating(arena.events), arena.legal_returns, arena.rival_returns

            for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                views = arena.render_cameras()
                for eye in EYES:
                    stereo.update(eye.encode())
                    stereo.update(views[eye])
                    if tick == 0:
                        prefix.update(eye.encode())
                        prefix.update(views[eye])
                frames.append(panel())
                rates = np.asarray(far)
                arena._aim = arena._command_to = lambda *args, rates=rates: rates
                _, reward, done, cap, _ = arena.step(near, render=False)
                if bool(done or cap) != (tick + 1 == length):
                    raise ValueError(f'replay changed the terminal frame on seed {row["seed"]}')
            frames.append(panel())
        referee = digest_json(arena.events)
        outcome = arena.point_result()
        if (stereo.hexdigest() != row['rgb'] or prefix.hexdigest() != row['prefix_rgb'] or
                referee != digest_json(row['events']) or reward != row['reward'] or outcome != row['outcome'] or
                alternating(arena.events) != row['longest']):
            raise ValueError(f'replay changed the exact RGB, referee trace or score on seed {row["seed"]}')
    finally:
        arena.close()
    check = dict(verified=True, stereo_rgb_sha256=stereo.hexdigest(), first_rgb_sha256=prefix.hexdigest(),
                 referee_sha256=referee, outcome=outcome, alternating_returns=row['longest'], terminal_frame=length)
    return frames, check


def _frame(tick, captures):
    canvas = Image.new('RGB', CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), 'Table tennis after a motor wiring change', font=font(22, True), fill=INK)
    draw.text((11, 33), '48 earlier table-tennis demonstrations · No table-tennis practice on the changed wiring',
              font=font(13), fill=MUTED)
    for mode, x in [('bounce', 10), ('frozen', 498)]:
        frames = captures[mode]
        pixels, run, own, rival = frames[min(tick, len(frames) - 1)]
        draw.text((x, 57), LABELS[mode], font=font(15, True), fill=INK)
        draw.text((x + PANEL[0], 57), f'{run} alternating returns', anchor='ra', font=font(14, True),
                  fill=GREEN if mode == 'bounce' else MUTED)
        canvas.paste(pixels, (x, 80))
        draw.text((x, 358), f'Learner {own} · Opponent {rival}', font=font(12), fill=MUTED)
        if tick >= len(frames) - 1:
            draw.text((x + PANEL[0], 358), 'Point lost', anchor='ra', font=font(12), fill=MUTED)
    draw.text((10, 386), f'Same held release {CHANGED_SEED}', font=font(12), fill=MUTED)
    draw.text((974, 386), f'{tick/movie.FPS:.2f} s', anchor='ra', font=font(12), fill=MUTED)
    return canvas.tobytes()


def _encode(captures):
    final = max(map(len, captures.values())) - 1
    count = final + 1 + HOLD
    with TemporaryDirectory(dir=ROOT, prefix='match-bridge-') as directory:
        pending = Path(directory) / COMPARISON.name
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{CANVAS[0]}x{CANVAS[1]}', '-r', str(movie.FPS), '-i', '-', '-an', '-c:v', 'libx264',
                   '-threads', '1', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(pending)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for tick in range(count):
                encoder.stdin.write(_frame(min(tick, final), captures))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the changed-wiring comparison')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(pending),
                        '-f', 'null', '-'], check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_frames', '-of', 'json', str(pending)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != count or pending.stat().st_size >= 2_000_000:
            raise ValueError('the comparison needs every frame and a file size below 2 MB')
        pending.replace(COMPARISON)
    return count


def comparison(course, calibration, dimensions, exposure):
    case = next(row for row in course['cases'] if row['seed'] == CHANGED_SEED)
    pair = {mode: case['results'][mode] for mode in LABELS}
    if ([(row['longest'], row['reward']) for row in pair.values()] != [(10, -1), (0, -1)] or
            any((row['seed'], row['lane']) != (case['seed'], case['lane']) for row in pair.values()) or
            len({row['prefix_rgb'] for row in pair.values()}) != 1):
        raise ValueError('expected the exact changed-wiring 10-to-0 matched release')
    matrix = np.asarray(calibration['wire'])
    decoders = {'bounce': np.asarray(calibration['decoder']), 'frozen': np.eye(5)}
    captures, proofs = {}, {}
    for mode, row in pair.items():
        action_hash = _match_setup(row, matrix, decoders[mode])
        captures[mode], proofs[mode] = _capture(row, dimensions)
        proofs[mode]['verified_transformed_action_sha256'] = action_hash
    if captures['bounce'][0][0].tobytes() != captures['frozen'][0][0].tobytes():
        raise ValueError('the actual first images in the matched comparison differ')
    frames = _encode(captures)
    _write(COMPARISON.with_suffix('.json'), dict(schema='tt24-match-bridge-movie-1', source=SOURCE.name,
           source_sha256=_hash(SOURCE), video_sha256=_hash(COMPARISON), verified_source_tapes=course['verified'],
           seed=CHANGED_SEED, lane=case['lane'], exposure=exposure, training_run_for_video=False,
           wiring_sha256=digest_json(calibration['wire']), same_raw_initial_broadcast_frame=True,
           frames=frames, fps=movie.FPS, terminal_hold_frames=HOLD,
           view='native MuJoCo broadcast RGB, labels outside the images', branches=proofs))
    print(f'Verified matched release {CHANGED_SEED}: 10 versus 0 alternating returns', flush=True)


def native(course, dimensions, exposure):
    case = next(row for row in course['cases'] if row['seed'] == NATIVE_SEED)
    row = case['results']['native']
    if (row['seed'], row['longest'], row['reward'], row['outcome']['legal_returns'],
            row['outcome']['rival_returns']) != (NATIVE_SEED, 25, 1, 13, 12):
        raise ValueError('expected the stock-wiring 25-return point win')
    actions = _match_setup(row, np.eye(5), np.eye(5))
    movie.CONFIG = dict(movie.CONFIG, width=dimensions[0], height=dimensions[1])
    movie.render_tape(row, SOURCE, NATIVE, branch='native',
                      subtitle='Stock wiring · 48 earlier table-tennis demonstrations',
                      terminal_caption='Point won · Learner 13 legal returns · Opponent 12 replies')
    receipt = NATIVE.with_suffix('.json')
    stock_exposure = {key: value for key, value in exposure.items() if key.startswith('table_tennis_demo')}
    _write(receipt, _read(receipt) | dict(exposure=stock_exposure, training_run_for_video=False,
                                        verified_transformed_action_sha256=actions))
    print(f'Verified stock release {NATIVE_SEED}: won with 25 alternating returns', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='render the separate stock-wiring point win')
    args = parser.parse_args()
    course, calibration, dimensions, exposure = _inputs()
    if args.native:
        native(course, dimensions, exposure)
    else:
        comparison(course, calibration, dimensions, exposure)


if __name__ == '__main__':
    main()
