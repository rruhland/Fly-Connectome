"""Matched native RGB practice with an optional visible-marker curriculum."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_catalog import dry_catalog
from .tt19_contextual import optical_placement
from .tt19_factory import FactoryModel, HOME, RATE
from .tt19_factory_serve import LOOK, ServeSimulator, propose
from .tt19_factory_serve_trial import policy
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import DrySimulator
from .tt18_view import ImageColors
from .tt19_view import ViewTransport
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport


DATA = ROOT / 'artifacts/tt19'
CHECKPOINT = DATA / 'factory-serve-player.json.gz'
CATALOG = DATA / 'factory-dry-catalog.npz'
OUTPUT = ROOT / 'artifacts/tt20'
GAME_SEED = 2082473549
PRACTICE_SEED = 2073535339
PRACTICE = 16
SWITCH = 8
MODES = ('fade', 'natural', 'frozen')
FRAME = np.array((0., 0., TABLE_HEIGHT))
ARENA = dict(venue='hall', strict_endline=True, width=320, height=240, detail='preview',
             view_mode='wrist+fixed', max_steps=190, mount='clamp', wrist_optics='outboard')
CLAMP_SHIFT = np.array((0., .035, .022))


class CurriculumArena(TT20Arena):
    """The experimental treatment consists only of the surveyed hall posts."""

    def _model_document(self):
        document = super()._model_document()
        for parent in document.iter():
            for node in list(parent):
                if node.get('name', '').startswith(('tt19_rim_', 'tt19_tip_mark')):
                    parent.remove(node)
        return document


class SurveyTransport(NaturalViewTransport):
    """Register the camera from the survey; share natural ball and blade tracking."""

    def __init__(self, lenses, state=None):
        super().__init__('wrist+fixed', state=state, lens_fov=lenses, continuous_wrist=False)
        self.survey = ViewTransport('wrist+fixed', state, wrist_fovy=lenses['wrist'])

    def observe(self, frames, tick, prior, wrist_prior=None):
        surveyed = self.survey.observe({key: ImageColors(rgb) for key, rgb in frames.items()}, tick, prior)
        self.survey_streams = tuple(surveyed)
        missing = frames.keys() - surveyed.keys()
        native = super().observe({key: frames[key] for key in frames if key in missing}, tick,
                                 {key: prior[key] for key in prior if key in missing}, wrist_prior)
        self.registry.last_events.update({key: self.survey.registry.last_events[key] for key in surveyed})
        self.registry.known.update(surveyed)
        self.registry.active = fits = {key: surveyed.get(key, native.get(key)) for key in frames
                                      if key in surveyed or key in native}
        return fits


def apparatus(marked, *, opponent=False):
    arena = CurriculumArena(**ARENA, marked=marked, role_palette='natural')
    if opponent:
        from .tt20_opponent import install_opponent

        install_opponent(arena)
    return arena


def optics(arena):
    """Expose lens specifications to the RGB estimators, excluding pose."""
    specs = arena.lens_intrinsics()
    return {stream: specs[stream]['fovy'] for stream in ('wrist', 'left', 'right')}


def transport(arena, marked, state=None):
    lenses = optics(arena)
    if marked:
        return SurveyTransport(lenses, state)
    return NaturalViewTransport('wrist+fixed', state=state, lens_fov=lenses, continuous_wrist=False)


def _sighting(track):
    return None if track is None or not track.observed else list(map(float, track.position))


def world_measurements(arena, tracker, tick, chosen):
    """Evaluator measurements never enter the policy or the practice merit."""
    truth = arena.ball.copy() - FRAME
    blade = arena.paddle.copy() - FRAME
    result = dict(tick=tick, ball=chosen, blade=_sighting(getattr(tracker, 'actor', None)),
                  actual_ball=truth.tolist(), actual_blade=blade.tolist(),
                  sources=list(tracker.sources.get('ball', ())),
                  registered=sorted(getattr(tracker, 'fits', {}).keys()))
    if chosen is not None:
        result['ball_error_m'] = float(np.linalg.norm(truth - np.asarray(chosen)))
    if result['blade'] is not None:
        result['blade_error_m'] = float(np.linalg.norm(blade - np.asarray(result['blade'])))
    return result


def play_point(arena, player, scene, seed, server, *, practice=False):
    tracker = scene.tracker()
    arena.reset(seed=seed, server=server, render=False)
    player.reset(server)
    commands, observations, capture, history, truth = [], [], [], [], []
    host = perf_counter()
    for tick in range(arena.max_steps):
        xyz = None
        truth.append((arena.ball.copy() - FRAME).tolist())
        if tick % 2 == 0 or player.phase == 'receive':
            tracker.update(arena.render_cameras())
            xyz = _sighting(tracker.ball)
            capture.append(world_measurements(arena, tracker, tick, xyz))
            if xyz is not None:
                history.append((tick, xyz))
        command = player.command(tick, xyz)
        observations.append(xyz)
        commands.append(command.tolist())
        _, reward, done, capped, _ = arena.step(command, render=False)
        if done or capped:
            break
    result = dict(seed=seed, server=server, actions=commands, optical=observations, captures=capture,
                  rgb=tracker.digest.hexdigest(), action_sha256=sha256(np.asarray(commands, dtype='<f8').tobytes()).hexdigest(),
                  result=arena.point_result(), events=arena.events.copy(), reward=reward,
                  world_ball=truth, plans=player.player.plans, wall_seconds=perf_counter() - host)
    if practice:
        result['merit'] = optical_placement(history, server, reward, FlightLaw(), None)
    return result


def retarget_programs(programs, shift=CLAMP_SHIFT):
    """Retain each factory stroke endpoint in table coordinates after a base move."""
    original, clamp = FactoryModel(), FactoryModel()
    clamp.base += shift
    adjusted = []
    for row in np.asarray(programs):
        start = HOME + RATE[:5] * row[:5]
        end = start + RATE[:5] * row[5:10]
        start_point, start_normal, before, _ = original.pose(start)
        end_point, end_normal, _, _ = original.pose(end)
        new_start = clamp.solve(start_point, start_normal, start)
        new_end = clamp.solve(end_point, end_normal, end)
        _, _, after, _ = clamp.pose(new_start)
        lateral = row[12] * np.clip(before[0, 0] / max(.03, after[0, 0]), .72, 1.4)
        adjusted.append(np.r_[(new_start - HOME) / RATE[:5], (new_end - new_start) / RATE[:5],
                              row[10:12], lateral])
    return np.asarray(adjusted)


def retarget_catalogue(catalogue, shift=CLAMP_SHIFT):
    """Translate dry paddle centers while preserving factory motor tapes."""
    poses = catalogue['poses'].copy()
    poses[:, :, :3] += shift
    return dict(catalogue, poses=poses)


def retarget_rollout(rollout_player, shift=CLAMP_SHIFT):
    """Move the independent nominal rollout plant to the declared flange location."""
    dry = rollout_player.sim
    dry.model.body('learner_base').pos[:] += shift
    paddle = int(dry.model.joint('learner_paddle_free').qposadr[0])
    dry.base[1+paddle:1+paddle+3] += shift


def model_arena():
    settings = dict(ARENA, venue='studio', strict_endline=False)
    return CurriculumArena(**settings, marked=False, role_palette='natural')


def prepare_model(directory):
    path = directory / 'curriculum-factory.json.gz'
    atlas = directory / 'curriculum-dry-catalog.npz'
    if path.exists():
        saved = _read(path)
        if saved['dry_catalog_sha256'] != _hash(atlas):
            raise ValueError('the common dry pilot catalogue changed')
        return saved
    used = FactoryModel()
    used.base += CLAMP_SHIFT
    sim = ServeSimulator(FlightLaw(), arena=model_arena())
    try:
        prepared = propose(sim, geometry=used)
    finally:
        sim.close()
    dry = dry_catalog(atlas, 2048, arena=model_arena())
    prepared.update(schema='tt20-independent-curriculum-factory-1', simulated_serve_frames=prepared['simulation_tapes']*LOOK,
                    dry_catalog=dry, dry_catalog_sha256=_hash(atlas), live_target_practice=0,
                    fixture=dict(ARENA), public_clamp_shift=CLAMP_SHIFT.tolist(), factory_pick=0)
    _write(path, prepared)
    print('factory simulations', prepared['simulation_tapes'],
          'shortlist', np.asarray(prepared['outcomes'])[:, :, 1].sum(axis=0).tolist(), flush=True)
    return prepared


def configuration(directory=None):
    saved = _read(CHECKPOINT)
    held = None if directory is None else prepare_model(directory)
    source = CATALOG if held is None else directory / 'curriculum-dry-catalog.npz'
    with np.load(source) as archive:
        catalogue = dict(archive)
    programs = retarget_programs(saved['programs']) if held is None else held['programs']
    if held is None:
        catalogue = retarget_catalogue(catalogue)
    rule = FlightLaw.from_state(saved['receiver']['law'])
    impulse = np.asarray(saved['receiver']['response'])
    prior = saved['factory_pick'] if held is None else held.get('factory_pick', 0)
    return programs, catalogue, rule, impulse, prior


def receiver(nominal, directory=None):
    programs, catalogue, rule, impulse, _ = configuration(directory)
    if nominal:
        rule, impulse = FlightLaw(), np.diag((-1.80, -.23))
    matched = policy(programs, catalogue, rule, impulse)
    if directory is None:
        retarget_rollout(matched.player)
    else:
        matched.player.sim.close()
        matched.player.sim = DrySimulator(rule, arena=model_arena())
    return matched


def practice_starts():
    rng = np.random.default_rng(PRACTICE_SEED)
    seeds = rng.integers(0, 2**31, PRACTICE).tolist()
    programs = np.concatenate([rng.permutation(4) for _ in range(PRACTICE // 4)]).tolist()
    return list(zip(seeds, programs, strict=True))


def rankings(rows, prior):
    result = []
    for stop in range(4, len(rows) + 1, 4):
        prefix = rows[:stop]
        merits = [np.mean([row['merit'] for row in prefix if row['program'] == index]) for index in range(4)]
        # Keep the same weak factory tie-break in both training conditions.
        adjusted = np.asarray(merits) + .03 * (np.arange(4) == prior)
        result.append(dict(points=len(prefix), control_frames=sum(len(row['actions']) for row in prefix),
                           last_block_serves=sum(row['result']['legal_serve'] for row in prefix[-4:]),
                           last_block_wins=sum(row['reward'] > 0 for row in prefix[-4:]),
                           merits=list(map(float, merits)), selected=int(np.argmax(adjusted))))
    return result


def practise(directory, mode, stop=PRACTICE):
    if mode == 'frozen':
        raise ValueError('the frozen reference has no target practice')
    programs, catalogue, rule, impulse, prior = configuration(directory)
    player = policy(programs, catalogue, rule, impulse, idle=True)
    path = directory / f'curriculum-{mode}-practice.json.gz'
    rows, states = [], {}
    if path.exists():
        previous = _read(path)
        if (previous['mode'] != mode or previous['source_sha256'] != _hash(CHECKPOINT)
                or previous.get('model_sha256') != _hash(directory/'curriculum-factory.json.gz')):
            raise ValueError('the existing practice receipt uses another policy')
        rows, states = previous['points'], previous['registration']
    starts = practice_starts()
    for index, row in enumerate(rows):
        if (row['seed'], row['program']) != starts[index]:
            raise ValueError('the existing practice changed the prechosen starts')
    current, arena, camera = None, None, None
    try:
        for index in range(len(rows), stop):
            marked = mode == 'fade' and index < SWITCH
            if current != marked:
                if arena is not None:
                    arena.close()
                arena = apparatus(marked)
                # The ordinary curriculum carries game experience. Camera
                # geometry gets a separate paired replay, avoiding a joint
                # intervention in the primary full-game comparison.
                camera = transport(arena, marked, states.get('marked' if marked else 'natural'))
                current = marked
            seed, candidate = starts[index]
            player.serves.force = candidate
            row = play_point(arena, player, camera, seed, 1, practice=True)
            row.update(program=candidate, marked=marked)
            rows.append(row)
            states['marked' if marked else 'natural'] = camera.state()
            _write(path, dict(schema='tt20-curriculum-practice-1', mode=mode, source_sha256=_hash(CHECKPOINT),
                              model_sha256=_hash(directory/'curriculum-factory.json.gz'),
                              schedule=[list(item) for item in starts], points=rows, registration=states,
                              learning=rankings(rows, prior), complete=len(rows) == PRACTICE))
            print(mode, 'practice', index+1, 'marked' if marked else 'natural', candidate,
                  row['result']['legal_serve'], row['reward'], round(row['merit'], 3),
                  len([entry for entry in row['captures'] if entry['ball'] is not None]), flush=True)
    finally:
        if arena is not None:
            arena.close()
    return _read(path)


def _metric(values):
    if len(values) == 0:
        return dict(count=0, median_m=None, p90_m=None)
    median, p90 = np.quantile(values, (.5, .9))
    return dict(count=len(values), median_m=float(median), p90_m=float(p90))


def perception(points):
    captures, forecast = [], []
    for row in points:
        events = {event['step'] for event in row['events'] if event['kind'] in ('contact', 'net')}
        history = []
        for sample in row['captures']:
            tick, xyz = sample['tick'], sample['ball']
            true = sample['actual_ball']
            if abs(true[0]) < .82 and abs(true[1]) < 1.62 and -.03 < true[2] < .95:
                captures.append(sample)
            history.append((tick, xyz))
            target = tick + 4
            if target >= len(row['world_ball']) or any(tick < hit <= target for hit in events):
                continue
            init = optical_start(history, .0109)
            if init is None:
                continue
            positions, _ = FlightLaw().trajectory(*init, 4, bounds=(.424,.80))
            forecast.append(np.linalg.norm(positions[-1] - np.asarray(row['world_ball'][target])))
    return dict(in_court_frames=len(captures), ball=_metric([r['ball_error_m'] for r in captures if 'ball_error_m' in r]),
                blade=_metric([r['blade_error_m'] for r in captures if 'blade_error_m' in r]),
                forecast_4_frames=_metric(forecast))


def game(directory, mode, seed, *, opponent=False, nominal=False):
    if mode == 'frozen':
        training = None
        budget = 0
        chosen = prepare_model(directory).get('factory_pick', 0)
    else:
        training = directory / f'curriculum-{mode}-practice.json.gz'
        data = _read(training)
        if not data['complete']:
            raise ValueError('finish the prechosen practice budget before evaluation')
        chosen = data['learning'][-1]['selected']
        budget = len(data['points'])
    arena = apparatus(False, opponent=opponent)
    camera = transport(arena, False)
    player = receiver(nominal, directory)
    player.serves.force = chosen
    rng = np.random.default_rng(seed)
    score = ServiceScore(first_server=-1)
    rows = []
    try:
        schedule = rng.integers(0, 2**31, 65).tolist()
        if set(schedule) & {value for value, _ in practice_starts()}:
            raise ValueError('practice and held launch schedules overlap')
        for point in schedule:
            row = play_point(arena, player, camera, point, score.server)
            score.record(row['result'])
            rows.append(row)
            print(mode, 'held', len(rows), score.score, row['server'], row['result']['legal_serve'],
                  row['result']['legal_returns'], row['result']['reason'], flush=True)
            if score.finished:
                break
    finally:
        player.player.close()
        arena.close()
    if not score.finished:
        raise ValueError('the held game exhausted its predefined launches')
    output = directory/f'curriculum-{mode}-{seed}{"-strong" if opponent else ""}{"-nominal" if nominal else ""}'
    report = dict(schema='tt20-curriculum-game-1', mode=mode, seed=seed, source_sha256=_hash(CHECKPOINT),
                  schedule=schedule, chosen=chosen, marked=False, frames=sum(len(row['actions']) for row in rows),
                  public_clamp_shift=CLAMP_SHIFT.tolist(), model_sha256=_hash(directory/'curriculum-factory.json.gz'),
                  opponent='adaptive' if opponent else 'standard', nominal_world=nominal,
                  score=list(score.score), native=True, points=rows, finished=score.finished,
                  practice_points=budget, practice_frames=0 if training is None else data['learning'][-1]['control_frames'],
                  practice_sha256=None if training is None else _hash(training), returns=sum(row['result']['legal_returns'] for row in rows),
                  multiple=sum(row['result']['legal_returns']>=2 for row in rows),
                  serves=sum(row['server']==1 and row['result']['legal_serve'] for row in rows),
                  serve_attempts=sum(row['server']==1 for row in rows), perception=perception(rows),
                  registration=camera.state())
    _write(output.with_suffix('.json.gz'), report)
    verify(report)
    report['replayed_points'] = len(rows)
    _write(output.with_suffix('.json.gz'), report)
    return report


def verify(report):
    score = ServiceScore(first_server=-1)
    arena = apparatus(False, opponent=report['opponent']=='adaptive')
    try:
        for seed, row in zip(report['schedule'], report['points']):
            if score.finished or (seed, score.server) != (row['seed'], row['server']):
                raise ValueError('held game schedule or service alternation diverged')
            arena.reset(seed=seed, server=score.server, render=False)
            for command in row['actions']:
                arena.step(command, render=False)
            digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
            if (arena.point_result() != row['result'] or digest != row['action_sha256']
                    or json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True)):
                raise ValueError(f'point {seed} diverged during action and physical replay')
            score.record(row['result'])
        if list(score.score) != report['score'] or not score.finished:
            raise ValueError('verified scores did not complete the same game')
    finally:
        arena.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=MODES)
    p.add_argument('--folder', type=Path, default=OUTPUT)
    p.add_argument('--operation', choices=('model', 'practice', 'game', 'both'), default='both')
    p.add_argument('--stop', type=int, default=PRACTICE)
    p.add_argument('--seed', type=int, default=GAME_SEED)
    p.add_argument('--strong-opponent', action='store_true')
    p.add_argument('--nominal-world', action='store_true')
    a = p.parse_args()
    a.folder.mkdir(exist_ok=True, parents=True)
    if not 1 <= a.stop <= PRACTICE:
        p.error('practice stop must be within the 16 prechosen starts')
    if a.operation == 'model':
        prepare_model(a.folder)
    if a.operation in ('practice','both') and a.mode != 'frozen':
        practise(a.folder, a.mode, a.stop)
    if a.operation in ('game','both'):
        result = game(a.folder, a.mode, a.seed, opponent=a.strong_opponent, nominal=a.nominal_world)
        print(result['score'], result['serves'], result['returns'], result['perception'])


if __name__ == '__main__':
    main()
