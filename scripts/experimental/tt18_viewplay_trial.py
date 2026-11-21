"""Play matched full games through private camera changes and replay the action tapes."""

import argparse
from base64 import b64decode, b64encode
from colorsys import hsv_to_rgb
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_camera import PerspectiveCamera
from .tt4_hall import hall_model
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import body_xml
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt10_series_trial import PHASES, _game_summary, _restore
from .tt10_window_trial import _point_row
from .tt14_gain_trial import memory_from_checkpoint, save_player
from .tt16_transfer_game import ROW_FIELDS, game_clock
from .tt17_allocation import NativeTravelServo
from .tt17_motor_trial import verify_physics
from .tt17_motor_protocol import setup_path, wiring_for
from .tt18_viewplay import POSE_HUE, POSE_RADIUS, ViewTransport
from .tt18_view_trial import add_view_landmarks, configure_view_colors


TT17 = ROOT / 'artifacts/tt17'
DONOR = TT17 / 'motor-gantry-coupling-travel-player.json.gz'
GAME_SEED = 1253086157
MODES = ('stable', 'adaptive', 'frozen')
# Camera positions, look directions and lenses are evaluated in the renderer only.
PRIVATE_LENSES = {
    'left': ((-.57, -2.74, .86), (.02, -.04, .18), 57.),
    'right': ((.59, -2.78, .91), (-.04, .05, .21), 56.),
    'right_moved': ((1.15, -2.77, 1.18), (.02, -.04, .20), 59.),
    'side': ((-1.64, -2.19, 1.05), (.03, -.15, .20), 57.),
    'high': ((1.31, -2.25, 1.78), (.04, -.12, .20), 62.),
}
LAYOUTS = (
    (0, 'original', {'A': 'left', 'B': 'right'}),
    (2, 'swapped_and_moved', {'A': 'right_moved', 'B': 'left'}),
    (4, 'third_added', {'A': 'right_moved', 'B': 'left', 'C': 'side'}),
    (6, 'single_remaining', {'B': 'left'}),
    (8, 'side_and_new_high', {'B': 'left', 'C': 'side', 'D': 'high'}),
)


def layout(mode, point):
    if mode == 'stable':
        return LAYOUTS[0][1:]
    return next((label, streams) for start, label, streams in reversed(LAYOUTS) if point >= start)


def add_play_cameras(document):
    """Add the visible RGB landmarks, the blade marker and private fixed camera bank."""
    add_view_landmarks(document)
    world = document.find('worldbody')
    for name, (eye, target, fovy) in PRIVATE_LENSES.items():
        camera = PerspectiveCamera(eye, target, 384, 288, fovy)
        right, up, forward = camera.basis
        quat = np.zeros(4)
        axes = np.column_stack((right, up, -forward))
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(axes).reshape(-1))
        position = np.array(eye) + (0., 0., TABLE_HEIGHT)
        ET.SubElement(world, 'camera', name=f'viewplay_{name}', pos=' '.join(map(str, position)),
                      quat=' '.join(map(str, quat)), fovy=str(fovy))
    wrist = document.find(".//body[@name='learner_wrist']")
    rgb = ' '.join(map(str, hsv_to_rgb(POSE_HUE, 1., .91)))
    ET.SubElement(wrist, 'geom', name='viewplay_pose', type='sphere', size='.010', pos=f'0 0 {POSE_RADIUS}',
                  contype='0', conaffinity='0', mass='0', rgba=rgb + ' 1')


class ViewPlayCapture:
    """Render only the active RGB views from the private fixed camera bank."""

    def __init__(self, *args, **settings):
        self._streams = LAYOUTS[0][2]
        self.first_rgb = None
        super().__init__(*args, **settings)

    def capture(self, streams):
        self._streams = streams
        self.first_rgb = None

    def render_cameras(self, mode=None):
        configure_view_colors(self)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        result = {}
        for stream, lens in self._streams.items():
            self._renderer.update_scene(self.data, camera=f'viewplay_{lens}')
            result[stream] = self._renderer.render()
        if self.first_rgb is None:
            self.first_rgb = result
        return result


class ViewPlayArena(ViewPlayCapture, TT8ServiceArena):
    """Run the regulation contact game with the RGB camera bank."""

    def _load_model(self):
        document = body_xml(self.body)
        add_play_cameras(document)
        return hall_model(document)


def specification(seed):
    spec = _read(setup_path(TT17, 'gantry-coupling'))
    spec.update(game_seed=seed, starts=_seeds(seed, 6011, 0, 65)['tests'],
                phases=match_phases(PHASES, 65, [seed, 6011]))
    return spec


def restore_player(path, transport):
    state, library, player = _restore(path, tracker_factory=transport.tracker)
    memory = memory_from_checkpoint(state, library, 'hybrid')
    motor = memory.maps[memory.active]
    player.adapter = motor.base
    if motor.coupled:
        player._make_servo = lambda: NativeTravelServo(motor, player.timing_memory, learn=player.adjust)
    return state, library, player, memory


def save_checkpoint(path, state, library, player, memory, transport, first_hash):
    series = state['series']
    # The portable player has no old optics or camera poses from the evaluator.
    for key in ('optics', 'tt16_camera', 'arena', 'calibration', 'probes', 'attempts'):
        series.pop(key, None)
    save_player(path, state, library, player, memory)
    portable = _read(path)
    portable['tt18_viewplay'] = dict(adaptive=transport.adaptive, fitted_views=transport.state(),
                                    donor_sha256=first_hash)
    _write(path, portable)
    restored = load_checkpoint(path)
    if restored[-1].state() != transport.state():
        raise ValueError('RGB-derived camera calibration did not survive a checkpoint round trip')


def load_checkpoint(path):
    metadata = _read(path)['tt18_viewplay']
    transport = ViewTransport(metadata['adaptive'], state=metadata['fitted_views'])
    return (*restore_player(path, transport), transport)


def preview(rgb):
    images = tuple(rgb.values())
    strip = np.hstack(images)
    buffer = BytesIO()
    Image.fromarray(strip).save(buffer, format='PNG', optimize=True)
    return dict(streams=list(rgb), rgb_sha256={key: sha256(frame).hexdigest() for key, frame in rgb.items()},
                rgb_strip_png=b64encode(buffer.getvalue()).decode())


def play(output, mode, seed, *, limit=None):
    spec = specification(seed)
    settings = dict(spec['arena'], width=384, height=288)
    arena = ViewPlayArena(**settings)
    transport = ViewTransport(mode != 'frozen')
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / f'viewplay-{seed}-{mode}.json.gz'
    checkpoint = receipt.with_name(receipt.name.replace('.json.gz', '-player.json.gz'))
    score = ServiceScore(first_server=-1)
    game = dict(seed=seed, points=[])
    rgb_samples = {}
    try:
        cache_hall_stereo(arena)
        state, library, player, memory = restore_player(DONOR, transport)
        wiring = wiring_for(arena, spec)
        for index, (point_seed, phase) in enumerate(zip(spec['starts'], spec['phases'], strict=True)):
            if limit is not None and index >= limit:
                return game
            stage, streams = layout(mode, index)
            arena.capture(streams)
            raw = _point_row(player, arena, wiring, seed, point_seed, phase, score.server)
            tracker = player.scene.tracker
            row = {key: raw[key] for key in ROW_FIELDS}
            row.update(stage=stage, streams=list(streams), rgb_sha256=tracker.digest.hexdigest(),
                       registration=tracker.registration, sightings=tracker.sightings)
            if stage not in rgb_samples:
                rgb_samples[stage] = preview(arena.first_rgb)
            game['points'].append(row)
            score.record(row['evaluator'])
            print(mode, stage, score.score, row['player_returns'],
                  np.sum(tracker.sightings, axis=0), tracker.registration[0]['events'], flush=True)
            if score.finished:
                break
        if not score.finished:
            raise ValueError('game did not finish within the held point budget')
        save_checkpoint(checkpoint, state, library, player, memory, transport, _hash(DONOR))
    finally:
        arena.close()
    game.update(score=score.score, metrics=_game_summary(game['points']))
    artifact = dict(schema='tt18-viewplay-v1', mode=mode, seed=seed, donor_sha256=_hash(DONOR),
                    donor=str(DONOR.relative_to(ROOT)), fixture_sha256=_hash(setup_path(TT17, 'gantry-coupling')),
                    renderer=dict(arena=settings, private_lenses=PRIVATE_LENSES, layouts=LAYOUTS),
                    clock=game_clock(game['points'], 0, spec['control_dt']), registration=transport.registry.counts,
                    checkpoint=checkpoint.name, checkpoint_sha256=_hash(checkpoint), rgb_samples=rgb_samples,
                    games=[game])
    _write(receipt, artifact)
    verify_physics(_read(receipt), spec)
    artifact['replayed_points'] = len(game['points'])
    _write(receipt, artifact)
    return artifact


def verify(folder, seed):
    spec = specification(seed)
    games = {}
    for mode in MODES:
        result = _read(folder / f'viewplay-{seed}-{mode}.json.gz')
        verify_physics(result, spec)
        if result['checkpoint_sha256'] != _hash(folder / result['checkpoint']):
            raise ValueError('saved player checkpoint digest changed')
        rows = result['games'][0]['points']
        if result['games'][0]['metrics'] != _game_summary(rows):
            raise ValueError('saved performance metrics differ from the action tape')
        if result['clock'] != game_clock(rows, 0, spec['control_dt']):
            raise ValueError('saved frame accounting differs from the action tape')
        if result['replayed_points'] != len(rows):
            raise ValueError('action replay did not cover the whole game')
        for index, row in enumerate(rows):
            stage, streams = layout(mode, index)
            if (row['stage'], row['streams']) != (stage, list(streams)):
                raise ValueError('an installation diverged from the camera schedule')
            if len(row['sightings']) != row['steps'] + 1:
                raise ValueError('RGB tracking did not cover every control frame')
            if len(streams) == 1 and any(any(frame[1:]) for frame in row['sightings']):
                raise ValueError('a one-view grab claimed a metric point')
        for stage, sample in result['rgb_samples'].items():
            strip = np.asarray(Image.open(BytesIO(b64decode(sample['rgb_strip_png']))))
            images = np.hsplit(strip, len(sample['streams']))
            for stream, rgb in zip(sample['streams'], images, strict=True):
                if sha256(rgb.tobytes()).hexdigest() != sample['rgb_sha256'][stream]:
                    raise ValueError(f'{stage} RGB sample differs from the observation digest')
        games[mode] = rows
    for reference, adaptive, frozen in zip(*(games[mode][:2] for mode in MODES), strict=True):
        for field in ('rgb_sha256', 'action_sha256', 'evaluator'):
            if reference[field] != adaptive[field] or reference[field] != frozen[field]:
                raise ValueError('controllers differed before the first camera change')
    return {mode: len(rows) for mode, rows in games.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('mode', choices=(*MODES, 'verify'))
    parser.add_argument('--seed', type=int, default=GAME_SEED)
    parser.add_argument('--limit', type=int, help='Run a development prefix without writing a result')
    args = parser.parse_args()
    if args.mode == 'verify':
        print(verify(args.output, args.seed))
    else:
        play(args.output, args.mode, args.seed, limit=args.limit)
