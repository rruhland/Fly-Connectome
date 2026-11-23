"""Render the settled-map witness from two independently checked physical tapes."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import BG, GREEN, INK, MUTED, font
from .tt24_donor_trial import ROOT
from .tt24_match_bridge_movie import CANVAS, PANEL, _capture, _match_setup
from .tt24_rally_bank import DIMS
from .tt24_rally_trial import returns
from .tt24_settled_match import exchange


SOURCE = ROOT/'settled-match-map-0.json.gz'
OUTPUT = ROOT/'settled-match-witness.mp4'
SEED = 1656923015
FPS = 30
HOLD = 24
MODE = ('calibrated','frozen')


def frame(tick, capture, pair):
    canvas = Image.new('RGB', CANVAS, BG)
    pen = ImageDraw.Draw(canvas)
    pen.text((10, 5), 'Table tennis after a motor wiring change', font=font(22, True), fill=INK)
    pen.text((11, 33), 'Same release and learned match head · Zero match practice on the changed wiring',
             font=font(13), fill=MUTED)
    for name, x, label in zip(MODE, (10,498), ('Single 1.45 s motor calibration','Uncompensated motor wiring'), strict=True):
        recorded = capture[name]
        pixels, run, near, far = recorded[min(tick, len(recorded)-1)]
        canvas.paste(pixels, (x,80))
        pen.text((x, 57), label, font=font(15, True), fill=INK)
        pen.text((x+PANEL[0], 57), f'{run} alternating returns', anchor='ra', font=font(14, True),
                 fill=GREEN if name=='calibrated' else MUTED)
        pen.text((x, 358), f'Learner {near} · Opponent {far}', font=font(12), fill=MUTED)
        if tick >= len(recorded)-1:
            verdict = {1:'Point won', -1:'Point lost', 0:'Point ended'}[pair[name]['reward']]
            pen.text((x+PANEL[0], 358), verdict, anchor='ra', font=font(12), fill=INK)
    pen.text((10, 386), f'Native camera replay · Held seed {SEED}', font=font(12), fill=MUTED)
    pen.text((974, 386), f'{tick/FPS:.2f} s', anchor='ra', font=font(12), fill=MUTED)
    return canvas.tobytes()


def export(capture, pair):
    final = max(map(len,capture.values())) - 1
    count = final+1+HOLD
    with TemporaryDirectory(dir=ROOT,prefix='settled-match-movie-') as directory:
        pending = Path(directory)/OUTPUT.name
        command = ['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
                   '-s',f'{CANVAS[0]}x{CANVAS[1]}','-r',str(FPS),'-i','-','-an','-c:v','libx264',
                   '-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                   '-movflags','+faststart',str(pending)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for index in range(count):
                encoder.stdin.write(frame(min(index,final),capture,pair))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg did not complete')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(pending),
                        '-f','null','-'], check=True)
        probe = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                                         'stream=nb_frames','-of','json',str(pending)])
        if int(json.loads(probe)['streams'][0]['nb_frames'])!=count:
            raise ValueError('the display must preserve every simulated decision frame')
        pending.replace(OUTPUT)
    return count


def main():
    archive = _read(SOURCE)
    case = next(x for x in archive['cases'] if x['seed']==SEED)
    pair = {name:case['results'][name] for name in MODE}
    if [(len(returns(row,1)),exchange(row),row['reward']) for row in pair.values()]!=[(8,7,1),(0,0,-1)]:
        raise ValueError('the registered comparison is not present')
    models = archive['settings']
    truth = np.asarray(models['truth'])
    inverses = dict(calibrated=np.linalg.inv(models['fit']),frozen=np.eye(5))
    capture, checks = {}, {}
    for name,row in pair.items():
        action = _match_setup(row, truth, inverses[name])
        capture[name], checks[name] = _capture(row, DIMS)
        checks[name].update(action_sha256=action, exchange_pairs=exchange(row),
                            near_returns=len(returns(row,1)), far_returns=len(returns(row,-1)))
    if capture['calibrated'][0][0].tobytes()!=capture['frozen'][0][0].tobytes():
        raise ValueError('the broadcast start differs across the two tapes')
    count = export(capture,pair)
    _write(OUTPUT.with_suffix('.json'),dict(schema='tt24-settled-match-movie-1',source=SOURCE.name,
           source_sha256=_hash(SOURCE), video_sha256=_hash(OUTPUT), seed=SEED, frames=count, fps=FPS,
           seconds=count/FPS, initial_broadcast_identical=True, source_pair_digest=digest_json(pair),
           archived_motor_calibration_seconds=models['observation_frames']/models['probe_speed_hz'],
           target_practice_used=0, training_for_video=0, branches=checks))
    print('verified', OUTPUT, count, 'frames', OUTPUT.stat().st_size, 'bytes', flush=True)


if __name__=='__main__':
    main()
