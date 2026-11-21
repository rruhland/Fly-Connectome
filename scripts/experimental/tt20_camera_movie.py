"""Replay a single-lens game, checking native camera pixels before adding a spectator view."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _read
from .tt8_service_arena import ServiceScore
from .tt19_view_full_game import FONT, RESULTS, score_text, shot_marker
from .tt19_view_trial import BROADCAST_CAMERA
from .tt20_fixture import TT20Arena


def render(receipt, destination):
    data = _read(receipt)
    eye = data['eye']
    name = {'wrist': 'Wrist camera', 'left': 'Outside camera'}[eye]
    arena = TT20Arena(**data['arena'], view_mode=eye)
    if data['strong']:
        from .tt20_opponent import install_opponent
        install_opponent(arena)
    scores = ServiceScore(first_server=-1)
    spectator = mujoco.Renderer(arena.model, width=480, height=360)
    fps = round(1 / arena.control_dt)
    canvas = Image.new('RGB', (984, 458), '#13191f')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), 'SO-101   /   Table tennis', font=FONT(24), fill='#f1f3f2')
    draw.text((12, 50), 'Spectator', font=FONT(17), fill='#b9c8d0')
    draw.text((498, 50), name + '   ·   learner input', font=FONT(17), fill='#94e3be')
    with TemporaryDirectory(dir=destination.parent) as folder:
        path = Path(folder) / destination.name
        encoder = subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                                    '-s','984x458','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
                                    '-threads','1','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
                                   stdin=subprocess.PIPE)
        try:
            for index, row in enumerate(data['points'], 1):
                if row['role'] != scores.server:
                    raise ValueError('service order changed')
                arena.reset(seed=row['seed'], server=scores.server, render=False)
                digest = sha256()
                score = score_text(scores.score)
                for tick, action in enumerate(row['actions']):
                    pixels = arena.render_cameras()[eye]
                    digest.update(pixels)
                    spectator.update_scene(arena.data, camera=BROADCAST_CAMERA)
                    canvas.paste(Image.fromarray(spectator.render()), (8,76))
                    canvas.paste(Image.fromarray(pixels), (496,76))
                    draw.rectangle((494,0,983,48), fill='#13191f')
                    draw.text((498,9), f'SO-101  {score}  Opponent', font=FONT(24), fill='#f1f3f2')
                    draw.rectangle((0,436,983,457), fill='#13191f')
                    label = 'SO-101 serves' if row['role'] == 1 else 'Opponent serves'
                    draw.text((12,436), f'Point {index}   ·   {label}', font=FONT(14), fill='#d9e4eb')
                    draw.text((498,436), shot_marker(row['events'],tick), font=FONT(14), fill='#d9e4eb')
                    encoder.stdin.write(canvas.tobytes())
                    arena.step(action, render=False)
                if arena.point_result() != row['outcome'] or digest.hexdigest() != row['rgb_sha256']:
                    raise ValueError(f"RGB or physics replay changed on point {index}")
                scores.record(row['outcome'])
                draw.rectangle((494,0,983,48), fill='#13191f')
                draw.text((498,9), f'SO-101  {score_text(scores.score)}  Opponent', font=FONT(24), fill='#f1f3f2')
                draw.rectangle((494,436,983,457), fill='#13191f')
                reason = row['outcome']['reason']
                draw.text((498,436), RESULTS.get(reason,reason.replace('_',' ')), font=FONT(14), fill='#f1f3f2')
                for _ in range(30 if scores.finished else 8):
                    encoder.stdin.write(canvas.tobytes())
            if not scores.finished or list(scores.score) != data['score']:
                raise ValueError('final score did not reproduce')
        finally:
            encoder.stdin.close()
            status = encoder.wait()
            spectator.close()
            arena.close()
        if status:
            raise RuntimeError(f'ffmpeg exited {status}')
        path.replace(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    render(a.receipt,a.destination)
