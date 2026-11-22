"""Play paired single-wrist and stereo continuations after a shared physical serve."""

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt18_view import CameraFit
from .tt19_flight import FlightLaw
from .tt19_so101 import WRIST_CAMERA
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt20_natural import NaturalViewTransport
from .tt21_service_drill import tape
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_match import CHECKPOINT, FACTORY, HANDOVER, STANDARD_PICK, step_recorded, summarize
from .tt22_opponent import RallyArena, install_rally_opponent
from .tt22_ordinary_game import FrozenMemory
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, alternating
from .tt23_coach import PRACTICE
from .tt23_coach_trial import OUTPUT, RECEIPT as COACH_RECEIPT, ReplyRanking


SCHEDULE = 1743013585
COUNT = 6
MODES = {'wrist': ('wrist',), 'wrist_twin': ('wrist',), 'fixed': ('left', 'right'), 'left': ('left',)}
REPORT = OUTPUT/'single-eye-comparison.json'


class MonocularPlayer(InversePlayer):
    """Use velocity from the causal monocular fit when it delivers the position."""

    speed = None

    def search(self, estimate, observation, target):
        if self.speed is None:
            return super().search(estimate, observation, target)
        revised = replace(observation, velocity=30.*self.speed)
        return RallyPlayer.search(self, (estimate[0], self.speed), revised, target)


class NominalCameraTransport(NaturalViewTransport):
    """Use the declared camera mount on the independent factory model."""

    def __init__(self, twin, eye, **settings):
        super().__init__(eye, **settings)
        self.twin = twin
        self.eye = eye
        self.camera = twin.model.camera(WRIST_CAMERA if eye == 'wrist' else CAMERAS[eye][0]).id

    def observe(self, frames, tick, prior, wrist_prior=None):
        twin = self.twin
        mujoco.mj_kinematics(twin.model, twin.data)
        mujoco.mj_camlight(twin.model, twin.data)
        eye = twin.data.cam_xpos[self.camera] - (0., 0., TABLE_HEIGHT)
        rotation = twin.data.cam_xmat[self.camera].reshape(3, 3).T * np.array([1., -1., -1.])[:, None]
        pixels = frames[self.eye]
        height, width = pixels.shape[:2]
        fit = CameraFit(eye, rotation, self.focal(self.eye, pixels), width, height)
        self.registry.known = self.registry.active = {self.eye: fit}
        return self.registry.active


def arena_for(view):
    return RallyArena(**dict(CONFIG, view_mode='wrist' if view == 'wrist_twin' else view))


class Station:
    def __init__(self, view, coach, memory, serve):
        self.view = view
        self.arena = arena_for(view)
        rival = install_rally_opponent(self.arena, target_profile=coach['profile'])
        rival.confirm = ReplyRanking(rival, coach).confirm
        law = FlightLaw()
        self.player = InversePlayer(law) if view == 'fixed' else MonocularPlayer(law)
        self.player.memory = FrozenMemory(memory)
        self.serve = serve

    def play(self, seed):
        arena, player, view = self.arena, self.player, self.view
        arena.reset(seed=seed, server=1, render=False)
        player.reset()
        intrinsics = arena.lens_intrinsics()
        lenses = {name: intrinsics[name] for name in MODES[view]}
        mono = view != 'fixed'
        settings = dict(lens_fov=lenses, law=player.law, flight=mono, associate=True, continuous_wrist=True)
        transport = (NominalCameraTransport(player.sim.twin, MODES[view][0], **settings) if view in ('wrist_twin', 'left')
                     else NaturalViewTransport(view, **settings))
        tracker = transport.tracker()
        actions, far_actions, observations = [], [], []
        for tick in range(arena.max_steps):
            images = arena.render_cameras()
            if tuple(images) != MODES[view]:
                raise ValueError('unexpected native camera stream')
            tracker.update(images, tick=tick)
            position, speed = None, None
            source = tracker.sources.get('ball_range')
            sized = (mono and tracker.raw_ball is not None and max(tracker.raw_ball_sigma) <= .020)
            if sized:
                position = np.asarray(tracker.raw_ball)
                source = 'size'
            elif tracker.ball is not None and tracker.ball.observed and (mono or source == 'stereo'):
                position = np.asarray(tracker.ball.position)
                speed = tracker.ball_velocity
            if position is not None:
                observations.append(dict(tick=tick, position=position.tolist(), source=source,
                                         speed=None if speed is None else speed.tolist()))
            if tick < HANDOVER:
                command = self.serve[tick]
                player.history.append((tick, position))
                player.sim.twin.advance(command)
            else:
                if mono:
                    player.speed = speed
                command = player.command(tick, position)
            actions.append(command.tolist())
            reward, done, capped, rival = step_recorded(arena, command)
            far_actions.append(rival)
            if done or capped:
                break
        return dict(seed=seed, role=1, view=view, reward=reward, actions=actions, rival_actions=far_actions,
                    optical=observations, events=arena.events.copy(), outcome=arena.point_result(),
                    longest=alternating(arena.events), shots=player.shots, rgb=tracker.digest.hexdigest())

    def close(self):
        self.player.close()
        self.arena.close()


def evaluate(rows):
    metrics = summarize(rows)
    metrics['twenty_or_more'] = sum(row['longest'] >= 20 for row in rows)
    metrics['at_least_one_near_reply'] = sum(row['outcome']['legal_returns'] >= 1 for row in rows)
    metrics['at_least_two_near_replies'] = sum(row['outcome']['legal_returns'] >= 2 for row in rows)
    metrics['wins'] = sum(row['outcome']['points_for'] for row in rows)
    metrics['live_frames'] = sum(len(row['actions']) for row in rows)
    return metrics


def verify(row, arena, on_frame=None):
    arena.reset(seed=row['seed'], server=1, render=False)
    digest = sha256()
    original = arena._aim, arena._command_to
    try:
        for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
            images = arena.render_cameras()
            for stream, pixels in images.items():
                digest.update(stream.encode())
                digest.update(pixels)
            if on_frame is not None:
                on_frame(tick, images)
            rates = np.asarray(far)
            arena._aim = arena._command_to = lambda *args: rates
            _, reward, done, capped, _ = arena.step(near, render=False)
            if bool(done or capped) != (tick+1 == len(row['actions'])):
                raise ValueError('independent replay changed point duration')
    finally:
        arena._aim, arena._command_to = original
    if (arena.point_result() != row['outcome'] or digest_json(arena.events) != digest_json(row['events'])
            or reward != row['reward'] or digest.hexdigest() != row['rgb']):
        raise ValueError('independent replay changed the point, event trace, or native RGB')
    if on_frame is not None:
        on_frame(len(row['actions']), arena.render_cameras())


def point_path(view):
    return OUTPUT/f'single-eye-{view}.json.gz'


def run(view, output):
    coach, state = _read(PRACTICE), _read(CHECKPOINT)
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=COUNT).tolist()
    prior = {row['seed'] for row in coach['rows']} | {row['seed'] for row in state['practice']}
    prior.update(_read(COACH_RECEIPT)['metadata']['seeds'])
    if len(seeds) != len(set(seeds)) or prior.intersection(seeds):
        raise ValueError('mono comparison requires fresh starts')
    serve = tape(np.asarray(_read(FACTORY)['programs'][STANDARD_PICK], dtype=float), 0.)
    metadata = dict(schema='tt23-single-eye-1', schedule=SCHEDULE, seeds=seeds, view=view, eyes=list(MODES[view]),
                    control_fps=30, serve=serve.tolist(), handover=HANDOVER,
                    inputs={p.name: _hash(p) for p in (PRACTICE, CHECKPOINT, FACTORY)})
    report = _read(output) if output.exists() else dict(metadata=metadata, rows=[])
    if report['metadata'] != metadata or [row['seed'] for row in report['rows']] != seeds[:len(report['rows'])]:
        raise ValueError('incompatible saved point schedule or controller')
    started = perf_counter()
    if len(report['rows']) < COUNT:
        station = Station(view, coach, state['evidence'], serve)
        try:
            for seed in seeds[len(report['rows']):]:
                row = station.play(seed)
                report['rows'].append(row)
                _write(output, report)
                print(view, seed, row['longest'], row['outcome']['legal_returns'], len(row['optical']),
                      row['outcome']['reason'], round(perf_counter()-started, 1), flush=True)
        finally:
            station.close()
    if report.get('verified') != COUNT:
        arena = arena_for(view)
        try:
            for row in report['rows']:
                verify(row, arena)
        finally:
            arena.close()
        report.update(verified=COUNT, summary=evaluate(report['rows']))
        _write(output, report)
    return report


def one_serve(row):
    for i, event in enumerate(row['events']):
        if event['kind'] == 'legal_serve':
            return row['events'][:i+1]
    return row['events']


def compare(output):
    documents = {view: _read(point_path(view)) for view in MODES}
    if any(doc.get('verified') != COUNT for doc in documents.values()):
        raise ValueError('paired comparison requires all verified points')
    audit = []
    for points in zip(*(doc['rows'] for doc in documents.values()), strict=True):
        reference = points[0]
        for row in points[1:]:
            if (row['seed'] != reference['seed'] or row['actions'][:HANDOVER] != reference['actions'][:HANDOVER]
                    or one_serve(row) != one_serve(reference)):
                raise ValueError('matched pair changed the prescribed physical service')
        audit.append(dict(seed=reference['seed'], physical_serve=True,
                          longest={row['view']: row['longest'] for row in points}))
    report = dict(schema='tt23-single-eye-paired-1', cases=audit, verified=len(MODES)*COUNT,
                  summary={view: doc['summary'] for view, doc in documents.items()},
                  sources={view: _hash(point_path(view)) for view in documents})
    _write(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=MODES, default='wrist')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    result = compare(args.output or REPORT) if args.compare else run(args.view, args.output or point_path(args.view))
    print(result['summary'], flush=True)
