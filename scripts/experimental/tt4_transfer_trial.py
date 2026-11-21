"""Run paired camera-only transfer trials on the four-axis MuJoCo table."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from .tt4_transfer import (ArmAdapter, CalibrationSample, TransferPlayer, TransferPrior,
                           ZERO, calibration_commands)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts' / 'tt3'
LEGACY_MINI = ARTIFACTS / 'three-joint-player.json.gz'
NATIVE_FILES = {'roles': ARTIFACTS / 'discovered-role-model.json',
                'clock': ARTIFACTS / 'discovered-reward-clock.json',
                'numeric': ARTIFACTS / 'discovered-numeric-source.json'}
BASE_CONDITIONS = ('legacy_mini', 'pong_only', 'fresh')
VIEWS = ('first_person', 'overhead_side')


def _read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def _write(path, data):
    body = json.dumps(data, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(body, mtime=0) if path.suffix == '.gz' else body)
    temporary.replace(path)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load_sources(first_person_mini=None):
    paths = {**NATIVE_FILES, 'legacy_mini': LEGACY_MINI}
    if first_person_mini is not None:
        paths['first_person_mini'] = first_person_mini
    states = {name: _read(path) for name, path in paths.items()}
    native = {name: states[name] for name in NATIVE_FILES}
    legacy = states['legacy_mini']
    if legacy['role_prior'] != native['roles'] or legacy['clock'] != native['clock']:
        raise ValueError('mini checkpoint and native Pong roles or clock have different parents')
    if legacy['memory']['profiles']['1'] != native['numeric']:
        raise ValueError('legacy mini checkpoint has a different native Pong numeric parent')
    if first_person_mini is not None:
        mini = states['first_person_mini']
        if mini['role_prior'] != native['roles'] or mini['clock'] != native['clock']:
            raise ValueError('first-person mini checkpoint and native Pong source have different parents')
        if mini['memory']['shared']['coefficients'] != native['numeric']['coefficients']:
            raise ValueError('first-person mini checkpoint uses a different source numeric search')
    return native, states, {name: _hash(path) for name, path in paths.items()}


def _seeds(seed, index, probes, points):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    ids = rng.integers(0, 2**31 - 1, size=3 + probes + points).tolist()
    return dict(index=index, arena_seed=ids[0], view_seed=ids[1], player_seed=ids[2],
                calibration=ids[3:3 + probes], tests=ids[3 + probes:])


def _tracker_factory(arena, mode):
    rig = arena.cameras if hasattr(arena, 'cameras') else arena._camera_rig()
    if mode == 'mujoco':
        from .tt4_native_rgb import native_tracker
        return lambda: native_tracker(rig, coordinates='world')
    from .tt4_camera import PerspectiveTracker
    return lambda: PerspectiveTracker(rig, coordinates='world')


def _observed(tracker, observations):
    return {track.id: np.asarray(track.position) for track in tracker.update(observations)
            if track.observed}


class Wiring:
    """Evaluator-controlled wiring that exposes neither signs nor order to the player."""

    def __init__(self, arena, seed):
        self.arena = arena
        rng = np.random.default_rng(np.random.SeedSequence([seed, 41708]))
        self.order = rng.permutation(4)
        self.signs = rng.choice((-1, 1), 4)

    def step(self, action, *, render=True):
        physical = np.empty(4)
        physical[self.order] = self.signs * action
        return self.arena.step(physical, render=render)

    def reset(self, **kwargs):
        return self.arena.reset(**kwargs)

    def point_result(self):
        return self.arena.point_result()

    @property
    def events(self):
        return self.arena.events

    def state(self):
        return dict(order=self.order.tolist(), signs=self.signs.tolist())


def _calibrate(arena, schedule, tracker_factory, horizon):
    trajectories, rows = [], []
    for index, seed in enumerate(schedule['calibration']):
        tracker = tracker_factory()
        observations = arena.reset(seed=seed)
        samples = [CalibrationSample(ZERO, _observed(tracker, observations), tracker.appearances)]
        commands = calibration_commands(schedule['player_seed'] + index, horizon)
        used = []
        for command in commands:
            observations, _, terminated, truncated, _ = arena.step(command)
            samples.append(CalibrationSample(command, _observed(tracker, observations), tracker.appearances))
            used.append(command.tolist())
            if terminated or truncated:
                break
        rows.append(dict(seed=seed, steps=len(used), frames=len(used) + 1, actions=used))
        trajectories.append(samples)
    return ArmAdapter(trajectories), rows


def _point(arena, player, seed):
    frames = arena.reset(seed=seed)
    player.reset()
    actions = []
    action = player.update(frames)
    reward = 0.
    elapsed = 0
    while True:
        frames, increment, terminated, truncated, info = arena.step(action)
        reward += increment
        actions.append(action.tolist())
        elapsed += 1
        following = player.update(frames, action)
        if terminated or truncated:
            break
        action = following
    policy = player.finish(reward)
    score = arena.point_result() if hasattr(arena, 'point_result') else info
    events = score.get('events', getattr(arena, 'events', []))
    return dict(seed=seed, steps=elapsed, frames=elapsed + 1, public_reward=float(reward),
                terminated=bool(terminated), truncated=bool(truncated),
                action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest(),
                actions=actions, events=events, evaluator=score, **policy)


def _counts(row):
    evaluated = row['evaluator']
    events = row['events']
    # These fields are evaluated only once the policy has stopped interacting.
    first = next((event.get('step', event.get('frame'))
                  for event in events
                  if event.get('kind') == 'legal_return'
                  and event.get('side') in (1, 'player', 'near')), None)
    counters = dict(
        contacts=int(evaluated['contacts']),
        player_returns=int(evaluated['legal_returns']),
        rival_returns=int(evaluated['rival_returns']),
        points_for=int(evaluated.get('points_for', row['public_reward'] > 0)),
        points_against=int(evaluated.get('points_against', row['public_reward'] < 0)),
    )
    return dict(**counters, first_legal_return_step=first)


def _course(job):
    from .tt4_arena import TT4Arena
    schedule, view, conditions, native, states, geometry = job
    began = time.perf_counter()
    arena = TT4Arena(seed=schedule['arena_seed'], view_seed=schedule['view_seed'], view=view,
                     **geometry['arena'])
    tracker_factory = _tracker_factory(arena, geometry['arena']['render_mode'])
    wired = Wiring(arena, schedule['arena_seed'])
    try:
        adapter, calibration = _calibrate(wired, schedule, tracker_factory, geometry['probe_horizon'])
        runs = {}
        for name in conditions:
            mini = states.get(name, states['legacy_mini'])
            prior = TransferPrior(name, native, schedule['player_seed'], mini=mini)
            player = TransferPlayer(adapter, prior, tracker_factory)
            before = prior.state()
            rows = []
            for seed in schedule['tests']:
                row = _point(wired, player, seed)
                rows.append(dict(**row, **_counts(row)))
            runs[name] = dict(points=rows, initial_prior=before, final_prior=prior.state())
    finally:
        if hasattr(arena, 'close'):
            arena.close()
    return dict(schedule=schedule, view=view, calibration=calibration, adapter=adapter.stats(),
                evaluator_wiring=wired.state(),
                runs=runs, wall_seconds=time.perf_counter() - began)


def _metrics(rows):
    return dict(
        points=len(rows),
        contacts=sum(row['contacts'] for row in rows),
        with_contact=sum(row['contacts'] > 0 for row in rows),
        player_returns=sum(row['player_returns'] for row in rows),
        with_return=sum(row['player_returns'] > 0 for row in rows),
        rival_returns=sum(row['rival_returns'] for row in rows),
        with_reply=sum(row['rival_returns'] > 0 for row in rows),
        reply_after_return=sum(row['rival_returns'] > 0 and row['player_returns'] > 0 for row in rows),
        sustained=sum(row['player_returns'] >= 2 for row in rows),
        points_for=sum(row['points_for'] for row in rows),
        won_after_return=sum(row['points_for'] > 0 and row['player_returns'] > 0 for row in rows),
        points_against=sum(row['points_against'] for row in rows),
        steps=sum(row['steps'] for row in rows),
        frames=sum(row['frames'] for row in rows),
        attributed_rewards=sum(row['attributed_rewards'] for row in rows),
        apparent_departures=sum(row['predicted_returns'] for row in rows),
        visually_observed_approaches=sum(bool(row['approach_frames']) for row in rows),
    )


def _timing(course, condition):
    steps = sum(row['steps'] for row in course['calibration'])
    for index, row in enumerate(course['runs'][condition]['points'], start=1):
        if row['first_legal_return_step'] is not None:
            return dict(point=index, steps=steps + row['first_legal_return_step'], observed=True)
        if row['player_returns']:
            # If an evaluator has no subpoint timestamps, end-of-point time
            # provides a conservative reproducible upper bound.
            return dict(point=index, steps=steps + row['steps'], observed=True)
        steps += row['steps']
    return dict(point=None, steps=steps, observed=False)


def _summary(courses, conditions, n, seed):
    phases = {'first': slice(1), 'first_eight': slice(8), 'all': slice(n)}
    bootstraps = np.random.default_rng(seed).integers(len(courses), size=(8192, len(courses)))
    arms = {}
    calibration_rows = [row for course in courses for row in course['calibration']]
    calibration_steps = sum(row['steps'] for row in calibration_rows)
    for condition in conditions:
        points = [course['runs'][condition]['points'] for course in courses]
        arms[condition] = {phase: _metrics([row for sequence in points for row in sequence[phase_slice]])
                           for phase, phase_slice in phases.items()}
        timing = [_timing(course, condition) for course in courses]
        arms[condition]['time_to_first_return'] = timing
        arms[condition]['target_steps_with_common_calibration'] = (calibration_steps
                                                                   + arms[condition]['all']['steps'])
        arms[condition]['target_frames_with_common_calibration'] = (
            sum(row['frames'] for row in calibration_rows) + arms[condition]['all']['frames']
        )
    comparisons = {}
    pairs = [(name, reference) for name in conditions if name.startswith('legacy_mini')
             or name == 'first_person_mini' for reference in ('pong_only', 'fresh')
             if reference in conditions]
    if 'mini_shuffled_timing' in conditions and 'legacy_mini' in conditions:
        pairs.append(('legacy_mini', 'mini_shuffled_timing'))
    paired_fields = ('with_return', 'player_returns', 'with_reply', 'reply_after_return',
                     'sustained', 'points_for', 'won_after_return', 'steps')
    for challenger, comparator in pairs:
        name = challenger + '_minus_' + comparator
        comparisons[name] = {}
        for phase, phase_slice in phases.items():
            sets = [(_metrics(course['runs'][challenger]['points'][phase_slice]),
                     _metrics(course['runs'][comparator]['points'][phase_slice]))
                    for course in courses]
            comparisons[name][phase] = {}
            for field in paired_fields:
                delta = np.array([left[field] - right[field] for left, right in sets])
                comparisons[name][phase][field] = dict(
                    by_installation=delta.tolist(), total=int(delta.sum()), mean=float(delta.mean()),
                    bootstrap95=np.quantile(delta[bootstraps].mean(axis=1), [.025, .975]).tolist(),
                )
    return dict(arms=arms, comparisons=comparisons,
                shared_calibration=dict(points=len(calibration_rows), steps=calibration_steps,
                                        frames=sum(row['frames'] for row in calibration_rows)),
                wall_seconds=sum(row['wall_seconds'] for row in courses))


def qualify(*, seed, installations, points, probes, probe_steps, views, mode, width, height,
            workers, distractor, first_person_mini=None, conditions=None, max_steps=140, frameskip=8,
            baseline=.16, venue='studio', checkpoint_progress=None):
    native, states, inputs = load_sources(first_person_mini)
    schedules = [_seeds(seed, index, probes, points) for index in range(installations)]
    extras = ('mini_shuffled_timing',) + (('first_person_mini',) if first_person_mini else ())
    geometry = dict(probe_horizon=probe_steps,
                    arena=dict(width=width, height=height, max_steps=max_steps, frameskip=frameskip,
                               render_mode=mode, distractor=distractor, baseline=baseline, venue=venue),
                    controller=dict(optics='known stereo camera calibration and visible table plane',
                                    body_fit='causal probe tracks and bounded cubic signed-action map',
                                    initial_gravity_m_per_step2=-.009, default_restitution=.70,
                                    table_contact_height_m=.025, cross_court_speed_cue=.42,
                                    actor_match_radius_m=.34, nominal_strike_behind_ready_m=.12,
                                    swing_lead_m_per_step=.050, wrist_offset=0.))
    by_view = {view: tuple(conditions) if conditions else BASE_CONDITIONS + (
        extras if view == 'first_person' else ()) for view in views}
    jobs = [(schedule, view, by_view[view], native, states, geometry)
            for schedule in schedules for view in views]
    started = time.perf_counter()
    if workers == 1:
        rows = []
        for job in jobs:
            rows.append(_course(job))
            if checkpoint_progress:
                checkpoint_progress(rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = []
            for row in pool.map(_course, jobs):
                rows.append(row)
                if checkpoint_progress:
                    checkpoint_progress(row)
    groups = {view: [course for course in rows if course['view'] == view] for view in views}
    summaries = {view: _summary(courses, by_view[view], points, seed)
                 for view, courses in groups.items()}
    implementation = {path.name: _hash(path) for path in sorted(
        (ROOT / 'scripts' / 'experimental').glob('tt4*')) if path.is_file()}
    return dict(schema='tt4-paired-rgb-transfer-v1', seed=seed, schedules=schedules, conditions=by_view,
                inputs_sha256=inputs, implementation_sha256=implementation, geometry=geometry,
                courses=groups, summaries=summaries, wall_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=1472928681)
    parser.add_argument('--installations', type=int, default=6)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--probes', type=int, default=10)
    parser.add_argument('--probe-steps', type=int, default=36)
    parser.add_argument('--views', nargs='+', choices=VIEWS, default=VIEWS)
    parser.add_argument('--mode', choices=('projected', 'mujoco'), default='projected')
    parser.add_argument('--venue', choices=('studio', 'hall'), default='studio')
    parser.add_argument('--baseline', type=float, default=.16)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=72)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--distractor', action='store_true')
    parser.add_argument('--first-person-mini', type=Path)
    parser.add_argument('--conditions', nargs='+', choices=(*BASE_CONDITIONS, 'mini_shuffled_timing',
                                                             'first_person_mini'))
    parser.add_argument('--max-steps', type=int, default=140)
    args = parser.parse_args()
    if min(args.points, args.installations, args.workers) < 1 or args.probes < 2:
        parser.error('at least one installation, one test point, one worker and two probes are required')
    if 'first_person_mini' in (args.conditions or ()) and args.first_person_mini is None:
        parser.error('first_person_mini requires --first-person-mini')

    def progress(row):
        counts = {name: _metrics(value['points']) for name, value in row['runs'].items()}
        print(json.dumps(dict(installation=row['schedule']['index'], view=row['view'],
                              calibration=row['adapter'], results=counts)), flush=True)

    options = vars(args).copy()
    options.pop('output')
    result = qualify(**options, checkpoint_progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, result)


if __name__ == '__main__':
    main()
