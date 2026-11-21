"""Export the longest verified rally from the native multi-camera held game."""

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_view_full_game import FONT, RESULTS, digest_json, inspect_video, score_text, shot_marker
from .tt19_view_trial import BROADCAST_CAMERA
from .tt20_curriculum import GAME_SEED, OUTPUT, apparatus, verify


VIDEO_SIZE = (1224, 1060)
PANEL_SIZE = (592, 444)
STREAMS = ('wrist', 'left', 'right')
BG = '#111920'
INK = '#edf3f6'
MUTED = '#aabcc8'
INPUT = '#95e0bc'


def longest(points):
    def returns(row):
        result = row['result']
        return result['legal_returns'] + result['rival_returns']

    return max(enumerate(points, 1), key=lambda entry: returns(entry[1]))


def _spectator(arena):
    visual = arena.model.vis
    before = visual.global_.offwidth, visual.global_.offheight
    try:
        visual.global_.offwidth, visual.global_.offheight = PANEL_SIZE
        return mujoco.Renderer(arena.model, width=PANEL_SIZE[0], height=PANEL_SIZE[1])
    finally:
        visual.global_.offwidth, visual.global_.offheight = before


def _draw(broadcast, frames, before, after, server, index, final, marker, tick, fps):
    canvas = Image.new('RGB', VIDEO_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), 'SO-101   /   Multi-camera rally replay', font=FONT(27), fill=INK)
    serve = 'SO-101 serves' if server == 1 else 'Opponent serves'
    draw.text((16, 50), f'Point {index}  ·  {serve}  ·  Final match {score_text(final)}', font=FONT(16), fill=MUTED)
    draw.rectangle((846, 10, 1207, 73), fill='#20303d')
    current = before if after is None else after
    draw.text((860, 22), 'SO-101', font=FONT(22), fill=INK)
    draw.text((970, 15), score_text(current), font=FONT(32), fill=INK)
    draw.text((1050, 22), 'Opponent', font=FONT(22), fill=INK)
    draw.text((862, 53), 'Score before point' if after is None else f'Score before point {score_text(before)}',
              font=FONT(14), fill=MUTED)

    panels = [('Spectator', broadcast), ('Wrist RGB  ·  policy input', frames['wrist']),
              ('Fixed left RGB  ·  policy input', frames['left']), ('Fixed right RGB  ·  policy input', frames['right'])]
    for (name, frame), (x, y) in zip(panels, ((16, 108), (616, 108), (16, 584), (616, 584)), strict=True):
        draw.text((x, y - 23), name, font=FONT(16), fill=MUTED if name == 'Spectator' else INPUT)
        panel = Image.fromarray(frame)
        canvas.paste(panel.resize(PANEL_SIZE, Image.Resampling.NEAREST), (x, y))
    draw.text((16, 1034), f'{tick / fps:05.2f} s  ·  {marker}', font=FONT(17), fill=INK)
    draw.text((930, 1034), 'Unmarked clamped mini court', font=FONT(16), fill=MUTED)
    return canvas.tobytes()


def _encoder(path, fps):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}', '-r', str(fps), '-i', '-', '-an',
                             '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '20',
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def render(source, output):
    report = _read(source)
    if not report['native'] or not report['finished'] or report['marked'] or report['opponent'] != 'standard':
        raise ValueError('expected the finished unmarked native game against the standard rival')
    verify(report)
    index, row = longest(report['points'])
    referee = ServiceScore(first_server=-1)
    for prior in report['points'][:index - 1]:
        referee.record(prior['result'])
    before = list(referee.score)
    ticks = [sample['tick'] for sample in row['captures']]
    controls = len(row['actions'])
    if not ticks or ticks[0] != 0 or ticks != sorted(set(ticks)) or ticks[-1] >= controls:
        raise ValueError('the source does not have a valid RGB capture schedule')
    expected_action = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
    if expected_action != row['action_sha256']:
        raise ValueError('the archived action hash does not match the tape')
    captures = set(ticks)
    arena = apparatus(False)
    spectator = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        arena.reset(seed=row['seed'], server=row['server'], render=False)
        fps = round(1. / arena.control_dt)
        spectator = _spectator(arena)
        view_data = mujoco.MjData(arena.model)

        def broadcast():
            mujoco.mj_copyData(view_data, arena.model, arena.data)
            mujoco.mj_kinematics(arena.model, view_data)
            mujoco.mj_camlight(arena.model, view_data)
            return arena._render_view(spectator, view_data, BROADCAST_CAMERA)

        pixel_digest = sha256()
        stream_digests = {key: sha256() for key in STREAMS}
        hold = round(1.4 * fps)
        with TemporaryDirectory(dir=output.parent) as temporary:
            pending = Path(temporary) / output.name
            encoder = _encoder(pending, fps)
            try:
                for tick, action in enumerate(row['actions']):
                    if tick in captures:
                        frames = arena.render_cameras()
                        if tuple(frames) != STREAMS:
                            raise ValueError('native policy stream order changed')
                        for key, pixels in frames.items():
                            pixel_digest.update(key.encode())
                            pixel_digest.update(pixels)
                            stream_digests[key].update(pixels)
                    marker = shot_marker(row['events'], tick) or 'Point in play'
                    encoder.stdin.write(_draw(broadcast(), frames, before, None, row['server'], index,
                                              report['score'], marker, tick, fps))
                    _, _, done, timeout, _ = arena.step(action, render=False)
                    if (done or timeout) != (tick + 1 == controls):
                        raise ValueError('replay diverged from the original point duration')
                result = arena.point_result()
                if result != row['result'] or digest_json(arena.events) != digest_json(row['events']):
                    raise ValueError('the original event sequence or scoring result changed')
                if pixel_digest.hexdigest() != row['rgb']:
                    raise ValueError('the replay did not reproduce the archived native RGB hash')
                referee.record(result)
                after = list(referee.score)
                winner = 'SO-101 point' if result['points_for'] else 'Opponent point'
                ending = RESULTS.get(result['reason'], result['reason'].replace('_', ' ').capitalize())
                frozen = _draw(broadcast(), frames, before, after, row['server'], index, report['score'],
                               f'{winner}  ·  {ending}', controls, fps)
                for _ in range(hold):
                    encoder.stdin.write(frozen)
            finally:
                encoder.stdin.close()
                status = encoder.wait()
            metadata = inspect_video(pending)
            if status or metadata['video_frames'] != controls + hold or metadata['fps'] != fps:
                raise ValueError('the encoded clip does not match its control timeline')
            pending.replace(output)
    finally:
        if spectator is not None:
            spectator.close()
        arena.close()

    receipt = dict(schema='tt20-native-rally-clip-1', source=source.name, source_sha256=_hash(source),
                   selection='most referee-credited returns by both sides, earliest point breaks a tie',
                   match_seed=report['seed'], match_score=report['score'], point=index, point_seed=row['seed'],
                   server=row['server'], score_before=before, score_after=after, result=result, events=row['events'],
                   event_sha256=digest_json(row['events']), action_sha256=expected_action,
                   controls=controls, original_captures=len(ticks), capture_ticks=ticks, closing_hold_frames=hold,
                   native_policy_streams=list(STREAMS), spectator_policy_input=False,
                   rgb_sha256=row['rgb'], verified_rgb_sha256=pixel_digest.hexdigest(),
                   stream_sha256={key: value.hexdigest() for key, value in stream_digests.items()},
                   movie=output.name, movie_sha256=_hash(output), **metadata)
    _write(output.with_name(output.stem + '-replay.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT / f'curriculum-frozen-{GAME_SEED}.json.gz')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'so101-longest-rally.mp4')
    args = parser.parse_args()
    result = render(args.source, args.output)
    print({key: result[key] for key in ('point', 'point_seed', 'score_before', 'score_after', 'original_captures',
                                       'controls', 'video_frames', 'duration')})
