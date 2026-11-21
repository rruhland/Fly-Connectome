"""Show the first shared held serve on which both transferred delay estimates helped."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import json

from PIL import Image, ImageDraw

from .tt4_transfer_trial import _read, _write, _hash
from .tt19_view_full_game import FONT
from .tt20_camera_carry import OUTPUT
from .tt20_fixture import TT20Arena

CELL = (368,276)
COLS = (8,384)
ROWS = (67,392)
SIZE = (760,720)


def replay(arena, eye, point):
    arena.reset(seed=point['seed'], server=-1, render=False)
    frames = []
    digest = sha256()
    for a in point['applied']:
        rgb = arena.render_cameras()[eye]
        digest.update(eye.encode())
        digest.update(rgb)
        frames.append(Image.fromarray(rgb).resize(CELL,Image.Resampling.LANCZOS))
        arena.step(a,render=False)
    if (digest.hexdigest()!=point['rgb_sha256'] or json.dumps(arena.events,sort_keys=True)!=json.dumps(point['events'],sort_keys=True)
            or arena.point_result()!=point['result']):
        raise ValueError(f"{eye} {point['learned_delay']} replay: rgb={digest.hexdigest()==point['rgb_sha256']} outcome={arena.point_result()==point['result']} events={json.dumps(arena.events,sort_keys=True)==json.dumps(point['events'],sort_keys=True)}")
    return frames


def render():
    receipt = OUTPUT/'direct-receive-latency-replication.json.gz'
    d = _read(receipt)
    eyes = ('left','wrist')
    pairs = {e:{r['seed']:r for r in d['pairs'][e]} for e in eyes}
    seed = next(r['seed'] for r in d['pairs']['left']
                if all(pairs[e][r['seed']]['transferred']['result']['legal_returns'] >
                       pairs[e][r['seed']]['unadapted']['result']['legal_returns'] for e in eyes))
    tiles = []
    for eye in eyes:
        arena = TT20Arena(**d['arena'],view_mode=eye)
        try:
            for condition in ('unadapted','transferred'):
                point = pairs[eye][seed][condition]
                tiles.append((replay(arena,eye,point),point))
        finally:
            arena.close()
    duration = max(len(images) for images,_ in tiles)
    canvas = Image.new('RGB',SIZE,'#121b20')
    draw = ImageDraw.Draw(canvas)
    draw.text((9,5),'One incoming serve  /  four independent plays',font=FONT(23),fill='#f0f3f2')
    draw.text((9,36),'No delay correction',font=FONT(17),fill='#d5dcdf')
    draw.text((384,36),'Delay learned through the other camera',font=FONT(17),fill='#94e3be')
    output = OUTPUT/'cross-camera-transfer.mp4'
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp)/output.name
        writer = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                                   '-s','760x720','-r','15','-i','-','-an','-c:v','libx264','-threads','1',
                                   '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)],
                                  stdin=subprocess.PIPE)
        try:
            for tick in range(duration+19):
                for index,(images,point) in enumerate(tiles):
                    left,top = COLS[index%2],ROWS[index//2]
                    label = 'Outside camera' if index<2 else 'Wrist camera'
                    canvas.paste(images[min(tick,len(images)-1)],(left,top))
                    draw.rectangle((left,top+276,left+367,top+313),fill='#1d282f')
                    state = ''
                    if any(e['kind']=='legal_return' and e.get('side')==1 and e['step']<=tick for e in point['events']):
                        state = '   ·   Legal return'
                    elif tick>=len(images)-1:
                        state = '   ·   No legal return'
                    draw.text((left+6,top+282),label+state,font=FONT(15),fill='#f0f3f2')
                draw.text((10,699),'Half speed    ·    Identical serve, four-frame motor delay',font=FONT(13),fill='#b9c8d0')
                writer.stdin.write(canvas.tobytes())
        finally:
            writer.stdin.close()
            status=writer.wait()
        if status:
            raise RuntimeError('ffmpeg failed')
        pending.replace(output)
    _write(OUTPUT/'cross-camera-transfer-replay.json',dict(source=receipt.name,source_sha256=_hash(receipt),
           seed=seed,video_sha256=_hash(output),fps=15,frames=duration+19,qualified=[dict(eye=p['eye'],
           delay=p['learned_delay'],frames=len(frames),rgb_sha256=p['rgb_sha256'],outcome=p['result']) for frames,p in tiles]))
    print(seed,duration+19)


if __name__ == '__main__':
    render()
