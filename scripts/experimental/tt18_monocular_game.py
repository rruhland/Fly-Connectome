"""Evaluate one-camera ball depth on the RGB tape of a scored TT18 game."""

import argparse
from hashlib import sha256
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_trial import PointInterface
from .tt18_monocular import AppearanceDepth, StereoContinuation
from .tt18_monocular_trial import DROP_START, score_continuation, score_frame, summarize
from .tt18_view import ImageColors, object_blobs, triangulate_views
from .tt18_view_trial import ORIGIN
from .tt18_viewplay import ViewTransport
from .tt18_viewplay_trial import ViewPlayArena, layout, specification, wiring_for


def evaluate(receipt):
    artifact = _read(receipt)
    if artifact['mode'] != 'adaptive':
        raise ValueError('expected the adaptive camera-change game receipt')
    seed = artifact['seed']
    spec = specification(seed)
    arena = ViewPlayArena(**artifact['renderer']['arena'])
    transport = ViewTransport(adaptive=True)
    appearance = AppearanceDepth()
    points, source_frames = [], []
    try:
        cache_hall_stereo(arena)
        wiring = wiring_for(arena, spec)
        for index, played in enumerate(artifact['games'][0]['points']):
            stage, streams = layout(artifact['mode'], index)
            if stage == 'side_and_new_high':
                break
            if list(streams) != played['streams']:
                raise ValueError(f'camera schedule differs on point {index}')
            arena.capture(streams)
            interface = PointInterface(wiring, prepare_point(arena, played['phase']),
                                       server=played['server'], seed=seed)
            continuation = StereoContinuation(arena.control_dt)
            images = interface.reset(seed=played['seed'])
            digest = sha256()
            rows, masked = [], []
            fits = {}
            for tick in range(len(played['actions']) + 1):
                for stream, rgb in images.items():
                    digest.update(stream.encode())
                    digest.update(rgb)
                colors = {stream: ImageColors(rgb) for stream, rgb in images.items()}
                if tick % 8 == 0:
                    fits = transport.register(colors)
                blobs = object_blobs(colors)['ball']
                if len(streams) == 1:
                    stream = next(iter(streams))
                    truth = arena.ball - ORIGIN
                    methods = (score_frame(appearance, continuation, tick, fits[stream], blobs[stream], stream, truth)
                               if stream in fits and stream in blobs else {})
                    rows.append(dict(tick=tick, registered=list(fits), detected=list(blobs),
                                     truth_m=truth.tolist(), methods=methods))
                else:
                    if DROP_START <= tick < DROP_START + 4:
                        stream = 'B'
                        truth = arena.ball - ORIGIN
                        methods = (score_continuation(continuation, tick, fits[stream], blobs[stream], truth)
                                   if stream in fits and stream in blobs else {})
                        masked.append(dict(tick=tick, retained_stream=stream, truth_m=truth.tolist(), methods=methods))
                    position = triangulate_views(fits, {stream: blob.center for stream, blob in blobs.items()})
                    before = appearance.source_frames
                    appearance.observe(fits, blobs, position)
                    if tick < DROP_START:
                        continuation.observe(tick, position)
                    if appearance.source_frames > before:
                        source_frames.append(dict(point=index, tick=tick))
                if tick < len(played['actions']):
                    images, _, _, _, _ = interface.step(played['actions'][tick])
            if digest.hexdigest() != played['rgb_sha256'] or arena.point_result() != played['evaluator']:
                raise ValueError(f'RGB or evaluation replay diverged on point {index}')
            points.append(dict(point=index, seed=played['seed'], frames=played['steps'] + 1,
                               stage=stage, streams=list(streams), evaluator=played['evaluator'],
                               monocular=rows, masked_dropout=masked, rgb_sha256=digest.hexdigest()))
            print(f'point {index}, {stage}, {played["steps"] + 1} RGB frames, '
                  f'{appearance.source_frames} cumulative stereo training frames', flush=True)
    finally:
        arena.close()
    return dict(schema='tt18-monocular-game-replay-v1', seed=seed, source=receipt.name,
                source_sha256=_hash(receipt), replayed_points=len(points), source_frames=source_frames,
                summary=summarize([row for point in points for row in point['monocular']]),
                masked_dropout_summary=summarize([row for point in points for row in point['masked_dropout']]),
                points=points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt18/mono-game.json.gz'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, evaluate(args.receipt))
