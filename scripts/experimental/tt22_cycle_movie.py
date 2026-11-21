"""Render the canonical paired held optical rally with the fixed far policy."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _write
from .tt21_recovery_movie import (AMBER, BG, CHIP, GREEN, INK, LINE, PANEL_WIDTH, SIZE,
                                  SPECTATOR_HEIGHT, _frame, _line, encode, native_events, paint_views)
from .tt22_cycle import World
from .tt22_cycle_trial import OUTPUT


POINT = 1134144980
MODES = ('teacher-frozen', 'teacher-plus-own-optics')
SOURCE = OUTPUT/'cycle-optical-held.json.gz'


def _replay(world, spectator, row):
    world.reset(POINT)
    digest = sha256()
    frames = []
    for t, (near, far) in enumerate(row['actions']):
        frames.append(_frame(world.arena, spectator, digest))
        _, reward, ended, capped, _ = world.step(near, far)
        if bool(ended or capped) != (t == len(row['actions'])-1):
            raise ValueError('point ended on the wrong frame')
    if (digest.hexdigest()!=row['rgb_sha256'] or world.arena.point_result()!=row['result']
            or reward!=row['reward'] or json.dumps(world.arena.events)!=json.dumps(row['events'])):
        raise ValueError('paired optics and event trace did not replay')
    frames.append(_frame(world.arena, spectator))
    return frames


def _decorate(rows, frames, tick, *, training_count, rate):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _line(draw, (20, 12), 'SO-101 extends one held rally using its own RGB', size=29, color=INK, bold=True)
    _line(draw, (22, 55), f'Held {POINT} · Same ordinary serve · Same physical far policy · Half speed', size=16)
    draw.line((20, 82, 1036, 82), fill=LINE)
    for mode, x, accent, title in ((MODES[0], 20, AMBER, 'FROZEN TEACHER STYLES'),
                                   (MODES[1], 536, GREEN, 'AFTER OWN-RGB PRACTICE')):
        row = rows[mode]
        index = min(tick, len(row['actions']))
        scored = [event['side'] for event in row['events'] if event['kind']=='legal_return' and event['step']<=index]
        shown = sorted((step, label) for label, side in (('SO-101 legal return', 1), ('Rival legal reply', -1))
                       for step in native_events(row, 'legal_return', side) if step<=index)
        detail = ('Stereo forecast with the carried style library' if mode==MODES[0] else
                  f'Same library · RGB forecast fit across {training_count} practice points')
        status = ('POINT LOST · ball lands on own half' if index==len(row['actions']) else
                  f'{shown[-1][1]} · frame {shown[-1][0]}' if shown else 'Opponent serves')
        _line(draw, (x, 95), title, size=21, color=accent, bold=True)
        _line(draw, (x, 125), detail, size=15)
        draw.rounded_rectangle((x+353, 93, x+499, 128), radius=7, fill=CHIP)
        _line(draw, (x+365, 103), f'{len(scored)} total returns', size=14, color=accent, bold=True)
        paint_views(canvas, draw, x, frames[mode][index])
        draw.rounded_rectangle((x, 704, x+499, 737), radius=5, fill=CHIP)
        _line(draw, (x+10, 712), status, size=15, color=accent, bold=True)
    _line(draw, (20, 750), 'Own-RGB fit added two returns here and lost two on the other held start.', size=15)
    _line(draw, (876, 750), f'Game time  {tick/rate:0.2f} s', size=15, color=INK)
    return canvas


def render(source, output, receipt):
    original = source.read_bytes()
    data = json.loads(gzip.decompress(original))
    rows = {mode: next(row for row in data['rows'][mode] if row['seed']==POINT) for mode in MODES}
    if [r['legal_alternating_returns'] for r in rows.values()] != [4,6]:
        raise ValueError('unexpected representative paired point')
    if any(sum(r['legal_alternating_returns'] for r in data['rows'][m]) != 6 for m in MODES):
        raise ValueError('aggregate held result differs from the caption')
    world = World()
    watch = mujoco.Renderer(world.arena.model, width=PANEL_WIDTH, height=SPECTATOR_HEIGHT)
    rate = round(1/world.arena.control_dt)
    try:
        frames = {mode: _replay(world, watch, row) for mode,row in rows.items()}
    finally:
        watch.close()
        world.close()
    with TemporaryDirectory(dir=output.parent) as folder:
        video = Path(folder)/output.name
        movie = encode(video,rows,frames,training_count=len(data['rows']['own-practice']),rate=rate,compose=_decorate)
        video.replace(output)
    _write(receipt,dict(schema='tt22-optical-cycle-paired-replay-1',seed=POINT,source=source.name,
                        source_snapshot_sha256=sha256(original).hexdigest(), movie=movie,
                        branches={mode:dict(rgb_sha256=r['rgb_sha256'], legal_returns=native_events(r,'legal_return',1),
                                            legal_rival_returns=native_events(r,'legal_return',-1), result=r['result'],
                                            point_sha256=sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                                            rgb_and_physics_verified=True) for mode,r in rows.items()},
                        total_returns_two_held=dict.fromkeys(MODES,6), renderer_sha256=_hash(Path(__file__))))
    print(output, movie, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT/'cycle-optical-paired.mp4')
    p.add_argument('--receipt', type=Path, default=OUTPUT/'cycle-optical-paired.json')
    a=p.parse_args(); render(a.source,a.output,a.receipt)
