"""Replay an RGB motor-memory point, audit every physics state, and film the verified tape."""

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_partner import PartnerArena
from .tt29_conditions import CONDITIONS, apply, cooperative_targets, table_settings
from .tt29_rally_physics import measured_launches


OUT = Path('artifacts/tt29')
SOURCE = OUT / 'camera-action-memory.json.gz'
WIDTH, HEIGHT = 1248, 758
BROADCAST = (868, 651)
EYE = (328, 246)
FPS = 30


def label(canvas, position, text, size, *, bold=False, color='#e9efee'):
    face = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    draw = ImageDraw.Draw(canvas)
    draw.text(position, text, font=ImageFont.truetype(face, size), fill=color)


def frame(arena, renderer, tick, count, streams, shot_id, capture=None):
    scene = Image.new('RGB', (WIDTH, HEIGHT), '#1b252d')
    label(scene, (18, 13), 'Camera-controlled rally', 27, bold=True)
    label(scene, (18, 49), f'Simulated SO-101 near end · Measured opening {shot_id}', 16, color='#a9bbc4')
    native = (capture or arena.render_cameras)()
    witness = Image.fromarray(arena.render_broadcast(renderer)).resize(BROADCAST, Image.Resampling.LANCZOS)
    scene.paste(witness, (18, 83))
    label(scene, (902, 23), f'{count:02d} legal returns', 23, bold=True, color='#a7e5c4')
    label(scene, (902, 52), f'{tick * arena.control_dt:05.2f} s · Real-speed playback', 15, color='#a9bbc4')
    if len(streams) == 2:
        for name, top in zip(streams, (100, 400), strict=True):
            label(scene, (902, top - 22), f'{name.title()} RGB input', 16, bold=True)
            image = Image.fromarray(native[name]).resize(EYE, Image.Resampling.LANCZOS)
            scene.paste(image, (902, top))
    else:
        name = streams[0]
        label(scene, (902, 78), f'{name.title()} RGB input', 16, bold=True)
        scene.paste(Image.fromarray(native[name]).resize(EYE, Image.Resampling.LANCZOS), (902, 100))
    label(scene, (902, 665), 'Far end · Powered practice carriage', 15, color='#a9bbc4')
    label(scene, (902, 697), 'Stereo estimates → near joint-rate commands' if len(streams) == 2
          else 'RGB estimates → near joint-rate commands', 13, color='#a9bbc4')
    return scene


def start_encoder(output, fps=FPS):
    return subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '1', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output),
    ], stdin=subprocess.PIPE)


def start_renderer(arena):
    width, height = BROADCAST
    offscreen = arena.model.vis.global_
    offscreen.offwidth = max(offscreen.offwidth, width)
    offscreen.offheight = max(offscreen.offheight, height)
    return mujoco.Renderer(arena.model, height=height, width=width)


def clearance_final(arena, guard, data):
    clearance = arena.clearance_result()
    data.qpos[:] = arena.data.qpos
    sample = min(guard.distances(data, horizon=.025), default=None, key=lambda row: row[2])
    clearance['samples'] += 1
    if sample is not None:
        if sample[2] < 0.:
            clearance['penetration_samples'] += 1
        nearest = clearance['nearest']
        if nearest is None or sample[2] < nearest['signed_m']:
            clearance['nearest'] = dict(first=sample[0], second=sample[1], signed_m=sample[2], time=arena.data.time)
    return clearance


def replay(feeds, source, shot_id, receipt, movie=None, *, fast_partner=False, last_frame=None, fps=FPS):
    archived_bytes = source.read_bytes()
    source_digest = sha256(archived_bytes).hexdigest()
    archive = json.loads(gzip.decompress(archived_bytes) if source.suffix == '.gz' else archived_bytes)
    row = next(r for r in archive['rows'] if r['shot']['id'] == shot_id)
    if measured_launches(feeds)[shot_id] != row['shot']:
        raise ValueError('The archived opening differs from the pinned measured ball launch')
    pose = archive['pose']
    streams = tuple(archive['view'].split('+'))
    targets = cooperative_targets(pose, archive.get('spread', 'practice'))
    condition = archive.get('condition', 'stock')
    width, height = archive.get('size', (384, 288))
    settings = dict(strike_ready=pose, width=width, height=height, clearance_guard=True, fast_search=True,
                    max_steps=max(9000, 90 * archive.get('return_target', 100)), targets=targets, guard_table=True, plant=table_settings(condition),
                    rate_scale=.8 if archive.get('rated') else 1., velocity_guard=archive.get('rated', False),
                    motor_envelope=archive.get('motor_envelope', False))
    fast_partner = fast_partner or archive.get('fast_partner', False)
    if fast_partner:
        from .tt29_partner_fast import FastPartnerArena
    arena_type = FastPartnerArena if fast_partner else PartnerArena
    arena = arena_type(**settings, **({'net_rescue': True} if archive.get('net_rescue') else {}))
    cad = PartnerArena(**settings, detail='cad')
    if condition not in ('stock', 'two_surface', 'narrow_table'):
        for station in (arena, cad):
            apply(station, CONDITIONS[condition])
            station.partner = type(station.partner)(station, targets=targets, guard_table=True)
    digest = sha256()
    renderer = process = stereo = None
    capture = None
    frames = count = 0
    far = np.zeros(5)
    try:
        arena.launch(row['shot'])
        arena.partner.reset(seed=row.get('opponent_seed', shot_id))
        arena.streams = streams
        arena.sample_clearance(True)
        guard = ArmClearance(cad.model, tolerance=0.)
        shadow = mujoco.MjData(cad.model)
        arena._cad_reader = guard, shadow
        if movie is not None or last_frame is not None:
            renderer = start_renderer(arena)
            if archive.get('cached_stereo'):
                from .tt29_fast_camera import CachedStereoRenderer
                stereo = CachedStereoRenderer(arena)
                capture = stereo.capture
        if movie is not None:
            process = start_encoder(movie, fps)
        for tick, action in enumerate(row['actions']):
            if process is not None and tick % round(1.0 / (fps * arena.control_dt)) == 0:
                process.stdin.write(frame(arena, renderer, tick, count, streams, shot_id, capture).tobytes())
                frames += 1
            _, _, done, capped, info = arena.exchange(action, far)
            # The initial launch already represents a far stroke, so subtract it from the displayed total.
            if info['events']:
                count = max(0, arena.legal_returns + arena.rival_returns - 1)
            digest.update(arena.data.qpos.astype('<f8').tobytes())
            digest.update(arena.data.qvel.astype('<f8').tobytes())
            if done or capped:
                break
        if arena.steps != len(row['actions']) or digest_json(arena.events) != digest_json(row['events']):
            raise ValueError('Recorded contact or referee events changed on replay')
        outcome = audit(arena.events)
        if {k: v for k, v in outcome.items() if k != 'shots'} != row['score']:
            raise ValueError('Recorded alternating-return summary changed on replay')
        clearance = clearance_final(arena, guard, shadow)
        checked = round(arena.data.time / arena.model.opt.timestep) + 1
        if clearance['samples'] != checked:
            raise ValueError(f'CAD audit sampled {clearance["samples"]} of {checked} physics states')
        if renderer is not None:
            # The final landing can fall between two video exposures.
            final_image = frame(arena, renderer, arena.steps, count, streams, shot_id, capture)
            if last_frame is not None:
                final_image.save(last_frame)
            if process is not None:
                process.stdin.write(final_image.tobytes())
                frames += 1
        if process is not None:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('FFmpeg did not encode the replay')
        brief = dict(schema='tt29-camera-memory-replay-1', shot_id=shot_id, source_sha256=source_digest,
                     feed_sha256=_hash(feeds), view=archive['view'], pose=pose, start_angles=arena._arm_home(1)[0],
                     events_sha256=digest_json(arena.events), event_count=len(arena.events), state_sha256=digest.hexdigest(),
                     exact_events=True, exact_score=True, score=outcome, result=arena.point_result(),
                     coverage=dict(physics_seconds=arena.data.time, physics_dt=arena.model.opt.timestep,
                                   control_dt=arena.control_dt, includes_terminal_state=True,
                                   complete_physics_state_coverage=True, full_cad=True,
                                   negative_distance_threshold_m=0., evaluated_pairs=len(guard.first)),
                     clearance=clearance, motor=arena.motor_audit(),
                     learning=dict(initial_examples=row['initial_examples'], final_examples=row['final_examples'],
                                   retrieved=row['retrieved'], planned=row['planned'],
                                   original_retrieval_wall_s=row['retrieval_s'], original_search_wall_s=row['search_s']),
                     video=None if movie is None else dict(sha256=_hash(movie), frames=frames, frame_rate=fps,
                                                          native_camera_size=[arena.width, arena.height]))
        _write(receipt, brief)
        print(shot_id, outcome['returns'], clearance, 'exact events', len(arena.events), flush=True)
        return brief
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if stereo is not None:
            stereo.close()
        if renderer is not None:
            renderer.close()
        arena.close()
        cad.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--id', type=int, default=7345)
    parser.add_argument('--receipt', type=Path, default=OUT / 'tt29_vision_memory_audit.json')
    parser.add_argument('--movie', type=Path)
    parser.add_argument('--fps', type=int, choices=(10, 15, 30), default=FPS)
    parser.add_argument('--fast-partner', action='store_true')
    parser.add_argument('--last-frame', type=Path)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.movie is None:
        replay(args.feeds, args.source, args.id, args.receipt, fast_partner=args.fast_partner, last_frame=args.last_frame, fps=args.fps)
    else:
        args.movie.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='tt29-replay-', dir=args.movie.parent) as work:
            temporary = Path(work) / 'rally.mp4'
            replay(args.feeds, args.source, args.id, args.receipt, temporary, fast_partner=args.fast_partner, last_frame=args.last_frame, fps=args.fps)
            temporary.replace(args.movie)


if __name__ == '__main__':
    main()
