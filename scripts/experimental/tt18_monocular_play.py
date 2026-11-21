"""Feed bounded one-camera ball tracks into the held view-change controller prefix."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt5_vision import _Path
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_window_trial import _point_row
from .tt16_transfer_game import ROW_FIELDS
from .tt18_monocular import AppearanceDepth, Ray, StereoContinuation
from .tt18_view import object_blobs
from .tt18_viewplay import RallyViewTracker, ViewTransport
from .tt18_viewplay_trial import DONOR, ViewPlayArena, layout, restore_player, specification, wiring_for


class MonocularTransport(ViewTransport):
    def __init__(self, frame_seconds):
        super().__init__(adaptive=True)
        self.appearance = AppearanceDepth()
        self.frame_seconds = frame_seconds

    def tracker(self):
        return MonocularTracker(self)


class MonocularTracker(RallyViewTracker):
    def __init__(self, transport):
        super().__init__(transport)
        self.one_view_ball = []
        self.continuation = StereoContinuation(transport.frame_seconds)

    def update(self, frames):
        tracks = super().update(frames)
        if len(frames) != 1:
            if self.ball is not None:
                position = np.asarray(self.ball.position)
                self.continuation.observe(self.tick, position)
                self.transport.appearance.observe(self.fits, object_blobs(frames)['ball'], position)
            return tracks
        if len(self.fits) != 1:
            return tracks
        stream, camera = next(iter(self.fits.items()))
        blob = object_blobs(frames)['ball'].get(stream)
        if blob is None:
            return tracks
        estimate, _ = self.continuation.predict(self.tick, camera, blob)
        source = 'stereo_ray'
        if estimate is None:
            source = 'appearance'
            appearance = self.transport.appearance
            estimate, _ = appearance.predict(stream, camera, blob)
            if not appearance.qualified(stream, estimate, camera, blob):
                return tracks
        x, y, z = estimate.position
        if abs(x) > 1.48 or abs(y) > 2.47 or not -.20 < z < 1.42:
            return tracks
        area = len(blob.pixels)
        if 1 not in self.paths or self.tick - self.paths[1].tick > 6:
            self.paths[1] = _Path(1, self.tick, estimate.position, area)
        else:
            self.paths[1].observe(self.tick, estimate.position, area)
        self.ball = self.paths[1].track(self.tick)
        self.support[1] = (area, 0)
        ray = Ray.observed(camera, blob.center)
        lateral_sigma = max(.004, .5 * ray.distance(estimate.position) / camera.focal)
        radial_variance = (estimate.range_half_width * ray.direction)**2
        self.ball_sigma = tuple(np.sqrt(radial_variance + lateral_sigma**2 * (1. - ray.direction**2)))
        self.one_view_ball.append(dict(tick=self.tick, source=source, source_frames=estimate.source_frames,
                                       estimate_m=estimate.position.tolist(),
                                       range_half_width_m=estimate.range_half_width))
        return (self.ball,)


def verify(arena, wiring, seed, points):
    score = ServiceScore(first_server=-1)
    for index, row in enumerate(points):
        if score.server != row['server']:
            raise ValueError(f'service order differs on point {index}')
        interface = PointInterface(wiring, prepare_point(arena, row['phase']), server=row['server'], seed=seed)
        interface.reset(seed=row['seed'], render=False)
        for command in row['actions']:
            interface.step(command, render=False)
        if (interface.point_result() != row['evaluator'] or interface.applied != row['applied']
                or json.loads(json.dumps(arena.events)) != row['events']
                or sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest() != row['action_sha256']):
            raise ValueError(f'action replay diverged on point {index}')
        score.record(row['evaluator'])
    return score.score


def play(receipt):
    baseline = _read(receipt)
    if baseline['mode'] != 'adaptive':
        raise ValueError('expected the adaptive camera-change game receipt')
    seed = baseline['seed']
    spec = specification(seed)
    arena = ViewPlayArena(**baseline['renderer']['arena'])
    transport = MonocularTransport(arena.control_dt)
    score = ServiceScore(first_server=-1)
    points = []
    try:
        cache_hall_stereo(arena)
        _, _, player, _ = restore_player(DONOR, transport)
        wiring = wiring_for(arena, spec)
        for index, played in enumerate(baseline['games'][0]['points']):
            stage, streams = layout(baseline['mode'], index)
            if stage == 'side_and_new_high':
                break
            if score.server != played['server'] or list(streams) != played['streams']:
                raise ValueError(f'baseline schedule differs on point {index}')
            arena.capture(streams)
            raw = _point_row(player, arena, wiring, seed, played['seed'], played['phase'], score.server)
            tracker = player.scene.tracker
            row = {key: raw[key] for key in ROW_FIELDS}
            row.update(stage=stage, streams=list(streams), rgb_sha256=tracker.digest.hexdigest(),
                       stereo_sightings=tracker.sightings, one_view_ball=tracker.one_view_ball,
                       cumulative_stereo_source_frames=transport.appearance.source_frames)
            if len(streams) > 1 and any(row[key] != played[key] for key in ('rgb_sha256', 'action_sha256', 'evaluator')):
                raise ValueError(f'the fallback changed the stereo prefix on point {index}')
            # The opening observation was matched for both dropout points in the
            # replay receipt. Subsequent images can differ as commands diverge.
            row['baseline'] = {key: played[key] for key in ('action_sha256', 'steps', 'evaluator', 'player_returns')}
            points.append(row)
            score.record(row['evaluator'])
            print(f'point {index}, {stage}, {score.score}, {len(tracker.one_view_ball)} monocular ball tracks, '
                  f'{row["evaluator"]["reason"]}', flush=True)
        points = json.loads(json.dumps(points))
        if verify(arena, wiring, seed, points) != score.score:
            raise ValueError('replayed score differs from the live points')
    finally:
        arena.close()
    return dict(schema='tt18-monocular-live-prefix-v1', seed=seed, baseline=receipt.name,
                baseline_sha256=_hash(receipt), donor_sha256=_hash(DONOR), score=score.score,
                matched_stereo_points=sum(len(row['streams']) > 1 for row in points),
                replayed_points=len(points), calibration_motor_commands=0, points=points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt18/mono-live.json.gz'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, play(args.baseline))
