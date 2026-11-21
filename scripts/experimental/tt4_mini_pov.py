"""Play the original three-joint table tennis simulator through a stereo head rig."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

from .tt3_arena import HZ, TT3Arena, heldout_cases
from .tt3_cycle import ConditionedNumeric, metrics
from .tt3_learning import play_point
from .tt3_recovery_learning import DelayMemory
from .tt3_role_discovery import DiscoveryBinder
from .tt3_table_checkpoint import export_state, load_player, make_player
from .tt3_visual import RoleModel
from .tt4_camera import PerspectiveTracker, TT3PerspectiveArena


MODES = ('first_person', 'external')


class StereoBinder(DiscoveryBinder):
    """Use the existing learned role comparisons with calibrated stereo tracks."""

    def __init__(self, model, clock, rig):
        self.rig = rig
        super().__init__(model, clock, learn=True)

    def reset(self):
        super().reset()
        self._scene.tracker = PerspectiveTracker(self.rig)


def use_stereo(player, rig):
    previous = player.binder.binder
    vision = StereoBinder(previous._initial, previous.clock, rig)
    vision.model = previous.model
    vision._bank = previous._bank
    player.binder.binder = vision


def load_camera_player(state, arena):
    """Resume a saved learner with the calibrated cameras from its installation."""
    player = load_player(state)
    exposure = state['visual_exposure']
    player.require_rebound = exposure.get('require_rebound', False)
    if exposure['view'] == 'first_person':
        use_stereo(player, arena.cameras)
    return player


def make_arena(case, view_seed, view, width, height, baseline):
    settings = dict(view_seed=view_seed, variant=case.variant, distractor=case.distractor)
    if view == 'external':
        return TT3Arena(case.seed, **settings)
    return TT3PerspectiveArena(case.seed, view=view, width=width, height=height, baseline=baseline, **settings)


def play_and_record(arena, player, view):
    actions = []

    def capture(frames, action, reward):
        actions.append(action.tolist())

    row = play_point(arena, player, capture=capture)
    row.update(seed=arena.seed, view_seed=arena.view_seed, variant=arena.variant)
    tape = dict(
        seed=arena.seed,
        view_seed=arena.view_seed,
        variant=arena.variant,
        distractor=arena.distractor,
        extra_resets=1,
        max_frames=arena.max_frames,
        view=view,
        action_sha256=row['action_sha256'],
        expected=arena.point_result(),
        actions=actions,
    )
    if view == 'first_person':
        tape.update(width=arena.cameras[0].width, height=arena.cameras[0].height)
    return row, tape


def play_course(seed, points, view, width, height, baseline, roles, numeric, clock, require_rebound=False):
    start = perf_counter()
    player = make_player(seed, RoleModel(**roles), ConditionedNumeric(numeric), DelayMemory(), clock)
    player.require_rebound = require_rebound
    cases = heldout_cases(seed, points + 1)
    view_seed = cases[0].view_seed
    rows = []
    best_tape = first_failure = None
    best_score = None
    for index, case in enumerate(cases[:-1]):
        arena = make_arena(case, view_seed, view, width, height, baseline)
        if index == 0 and view == 'first_person':
            use_stereo(player, arena.cameras)
        row, tape = play_and_record(arena, player, view)
        rows.append(row)
        tape.update(point=index + 1, course_seed=seed)
        if view == 'first_person':
            tape['baseline'] = baseline
        score = (row['legal_returns'] >= 2, row['points_for'], row['legal_returns'], row['contacts'])
        if best_score is None or score > best_score:
            best_tape, best_score = tape, score
        if first_failure is None and row['points_against']:
            first_failure = tape

    state = export_state(player, seed, clock)
    state['visual_exposure'] = dict(view=view, frame_rate_hz=HZ, course_seed=seed, view_seed=view_seed,
                                    require_rebound=require_rebound,
                                    bound_frames=sum(row['bound_frames'] for row in rows),
                                    **metrics(rows))
    if view == 'first_person':
        state['visual_exposure'].update(width=width, height=height, baseline=baseline)

    # The first-person export must reproduce the same next action sequence after reload.
    next_arena = make_arena(cases[-1], view_seed, view, width, height, baseline)
    restored = load_camera_player(json.loads(json.dumps(state, allow_nan=False)), next_arena)
    live, _ = play_and_record(next_arena, player, view)
    resumed_arena = make_arena(cases[-1], view_seed, view, width, height, baseline)
    resumed, _ = play_and_record(resumed_arena, restored, view)
    if live != resumed:
        difference = [key for key in live if live[key] != resumed[key]]
        raise ValueError(f'restored {view} player diverged on {difference}')
    state['next_point_check'] = {key: live[key] for key in
                                 ('seed', 'view_seed', 'action_sha256', 'frames', 'public_reward')}
    course = dict(seed=seed, view=view, summary=metrics(rows), points=rows,
                  replay=best_tape, first_loss=first_failure, seconds=perf_counter() - start,
                  visual_exposure=state['visual_exposure'], final_roles=state['roles'],
                  next_point_check=state['next_point_check'])
    return course, state


def read_source(directory):
    values, inputs = [], []
    for name in ('role-model', 'numeric-source', 'reward-clock'):
        path = directory / f'discovered-{name}.json'
        payload = path.read_bytes()
        values.append(json.loads(payload))
        inputs.append(dict(file=path.name, sha256=sha256(payload).hexdigest()))
    return values, inputs


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.tmp')
    with gzip.open(pending, 'wt') as stream:
        json.dump(data, stream, separators=(',', ':'), allow_nan=False)
    pending.replace(path)


def run(args):
    values, source = read_source(args.source)
    all_starts = [args.seed + index for index in range(args.matches)]
    planned = [(seed, view) for seed in all_starts for view in args.views]
    report = dict(schema='mini-tt3-camera-play-v1', source=source, frame_rate_hz=HZ,
                  course_seeds=all_starts, views=args.views, points_per_course=args.points,
                  require_rebound=args.require_rebound,
                  first_person_camera=dict(width=args.width, height=args.height, baseline=args.baseline),
                  courses=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(play_course, seed, args.points, view, args.width, args.height, args.baseline, *values,
                            require_rebound=args.require_rebound):
                (seed, view) for seed, view in planned}
        for job in as_completed(jobs):
            key = jobs[job]
            course, state = job.result()
            report['courses'].append(course)
            report['courses'].sort(key=lambda item: (item['seed'], args.views.index(item['view'])))
            report['totals'] = {view: metrics([row for item in report['courses'] if item['view'] == view
                                               for row in item['points']]) for view in args.views}
            if key == (args.seed, 'first_person'):
                state['native_source'] = source
                write(args.checkpoint, state)
            write(args.output, report)
            print(json.dumps(dict(seed=key[0], view=key[1], **course['summary'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--source', type=Path, default=Path('artifacts/tt3'))
    parser.add_argument('--seed', type=int, default=93714422)
    parser.add_argument('--points', type=int, default=20)
    parser.add_argument('--matches', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--baseline', type=float, default=.16)
    parser.add_argument('--views', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--require-rebound', action='store_true',
                        help='limit follow-up meeting proposals to times after a visual rebound')
    args = parser.parse_args()
    if min(args.points, args.matches, args.workers) < 1 or min(args.width, args.height) < 16:
        parser.error('counts must be positive and camera dimensions must be at least 16 pixels')
    if not math.isfinite(args.baseline) or args.baseline <= 0.:
        parser.error('stereo baseline must be a finite positive distance')
    if len(set(args.views)) != len(args.views):
        parser.error('each view can be requested once')
    run(args)


if __name__ == '__main__':
    main()
