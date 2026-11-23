"""Show the same changed-arm release with the newly learned and stale motor maps."""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

from .tt4_transfer_trial import _read
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_current_pause import INVISIBLE, scene
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED
from .tt24_single_arm_course import CAP


def replay_frames(original, case, row, server):
    arm = InstalledArm(case['plan']['body'])
    arm.arena.max_steps = CAP
    frames = []
    try:
        fixture = scene({case['plan']['body']: arm}, original, case, server)
        fixture.start(False)
        for tick, (action, phase) in enumerate(zip(row['actions'], row['phases'], strict=True)):
            if tick == row['release_tick']:
                fixture.release(True)
                if fixture.release_state != row['release_state']:
                    raise ValueError('wrist image or release state changed')
            _, done = fixture.advance(action, phase, False)
            if done != (tick + 1 == len(row['actions'])):
                raise ValueError('terminal time changed')
            if tick % 2 == 0 or done:
                frames.append((arm.arena.render_external(), arm.arena.render_wrist(),
                               len(arm.arena._rebounds), phase, done))
        for key in ('metrics', 'actions', 'events'):
            if digest_json(fixture.score()[key]) != digest_json(row[key]):
                raise ValueError(f'physical replay changed {key}')
        return frames
    finally:
        arm.arena.close()


def main():
    archive = _read(ROOT / 'current-pause-stream-2.json.gz')
    case = archive['rows'][INVISIBLE]
    original = _read(ROOT / archive['original'])
    current, stale = case['current'], case['yoked_stale']
    held = current['release_tick']
    if current['actions'][:held] != stale['actions'][:held] or current['release_state'] != stale['release_state']:
        raise ValueError('the paired release does not have identical physical preparation')
    server = BounceArena(view='none')
    try:
        sides = [replay_frames(original, case, run, server) for run in (current, stale)]
    finally:
        server.close()
    titles = ('Use the newly learned motor map', 'Keep the previous motor map')
    phases = {'setup': 'Preparing', 'surprise_inspection': 'Measuring joint response', 'playing': 'Ball released'}
    title = ImageFont.truetype(BOLD, 25)
    label = ImageFont.truetype(BOLD, 17)
    note = ImageFont.truetype(FONT, 16)
    target = ROOT / 'current-pause-paired.mp4'
    with TemporaryDirectory(dir=ROOT) as folder:
        pending = Path(folder) / target.name
        command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '936x490',
                   '-r', '30', '-i', '-', '-vf', 'tpad=stop_mode=clone:stop_duration=1', '-c:v', 'libx264',
                   '-threads', '1', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', str(pending)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for index in range(max(map(len, sides))):
                frame = Image.new('RGB', (936, 490), BACKGROUND)
                pen = ImageDraw.Draw(frame)
                pen.text((14, 11), 'The first ball after a hidden motor change', font=title, fill=INK)
                pen.text((14, 50), 'Same start, setup movements, release time and wrist image', font=note, fill=MUTED)
                for side, images in enumerate(sides):
                    outside, wrist, score, phase, done = images[min(index, len(images)-1)]
                    x = 14 + 466 * side
                    pen.text((x, 84), titles[side], font=label, fill=INK)
                    frame.paste(Image.fromarray(outside).resize((444, 333), Image.Resampling.BILINEAR), (x, 112))
                    camera = Image.fromarray(wrist).resize((144, 108), Image.Resampling.BILINEAR)
                    frame.paste(camera, (x + 295, 332))
                    pen.rectangle((x + 295, 314, x + 439, 332), fill=BACKGROUND)
                    pen.text((x + 299, 314), 'Wrist camera', font=note, fill=MUTED)
                    state = 'Drop ended' if done else phases[phase]
                    pen.text((x, 457), state, font=note, fill=MUTED)
                    pen.text((x + 444, 456), f'{score} rebounds', anchor='ra', font=label, fill=ACCENT)
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('video encoding failed')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(pending), '-f', 'null', '-'], check=True)
        pending.replace(target)


if __name__ == '__main__':
    main()
