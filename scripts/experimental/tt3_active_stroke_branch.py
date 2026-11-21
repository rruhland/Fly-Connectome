"""Search continuation pulses after a shared visual prefix on each practice point."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tt3_active_stroke import ActiveStrokeMemory, MomentumPlayer, Pulse, run_pool, summary
from tt3_arena import TT3Arena, VARIANTS, heldout_cases
from tt3_numeric_transfer import PortableMemory
from tt3_visual import RoleModel


def finish_point(arena, learner, frames, *, snapshot=False):
    """Capture the common state after visible separation, before a new approach."""
    branch = None
    events = []
    actions = []
    done = False
    while not done:
        action = learner.act(frames)
        frames, reward, done, info = arena.step(action)
        learner.observe(reward, terminal_frames=frames if done else None)
        events.extend(info['events'])
        actions.append(action.tolist())
        recurrence = learner.memory.recurrence
        if snapshot and branch is None and not done and recurrence.far and not recurrence.returning:
            branch = (deepcopy(arena), deepcopy(learner), frames, list(events), list(actions))
    row = learner.finish()
    row.update(arena.point_result(), events=events,
               action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest())
    return row, branch, actions


def practice_course(cases, roles, source, candidates, seed):
    if candidates[0].fraction != 0.:
        raise ValueError('first candidate must preserve the common prefix')
    start = perf_counter()
    memory = ActiveStrokeMemory(PortableMemory.restore(source), candidates[0])
    receiver = MomentumPlayer(roles, memory, seed=seed, forecast=True, recall=True, learn_roles=True)
    view = cases[0].view_seed
    courses = [dict(pulse=asdict(pulse), points=[], extra_cameras=0, replay=None) for pulse in candidates]
    for number, case in enumerate(cases, 1):
        arena = TT3Arena(case.seed, view_seed=view, variant=case.variant, distractor=case.distractor)
        case_info = {**asdict(case), 'view_seed': view}
        receiver.reset()
        frames = arena.reset()
        baseline, branch, _ = finish_point(arena, receiver, frames, snapshot=True)
        baseline.update(point=number, case=case_info, branched_at=None if branch is None else branch[0].frames)
        courses[0]['points'].append(baseline)
        courses[0]['extra_cameras'] += baseline['frames']
        for choice, course in zip(candidates[1:], courses[1:], strict=True):
            if branch is None:
                course['points'].append(baseline)
                continue
            clone, learner, pixels, earlier_events, earlier_actions = branch
            clone = deepcopy(clone)
            learner = deepcopy(learner)
            learner.memory.pulse = choice
            row, _, actions = finish_point(clone, learner, pixels)
            full = np.asarray(earlier_actions + actions, dtype='<f8')
            row.update(point=number, case=case_info, branched_at=branch[0].frames,
                       events=earlier_events + row['events'], action_sha256=sha256(full.tobytes()).hexdigest())
            course['points'].append(row)
            course['extra_cameras'] += len(actions)
            if (course['replay'] is None and row['points_for'] and row['legal_returns'] >= 2
                    and row['rival_returns'] and row['pulse_frames']):
                course['replay'] = dict(**case_info, extra_resets=1, point=number,
                                        expected=clone.point_result(), actions=full.tolist(), events=row['events'],
                                        action_sha256=row['action_sha256'])
    return courses, perf_counter() - start


def practice(roles, source, candidates, *, seed, matches, points, variant):
    groups = [[] for _ in candidates]
    total_seconds = 0.
    for index in range(matches):
        cases = heldout_cases(seed + index, points, variant=variant)
        rows, seconds = practice_course(cases, roles, source, candidates, seed + index)
        total_seconds += seconds
        for group, course in zip(groups, rows, strict=True):
            group.append(course)
        print(dict(match=index, wall_seconds=seconds, candidates=[summary(group) for group in groups]), flush=True)
    results = [dict(pulse=asdict(pulse), overall=summary(group), courses=group,
                    computed_cameras=sum(course['extra_cameras'] for course in group))
               for pulse, group in zip(candidates, groups, strict=True)]
    return results, total_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles', type=Path)
    parser.add_argument('numeric', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--practice-seed', type=int, default=786225)
    parser.add_argument('--qualification-seed', type=int, default=589381042)
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--heldout-matches', type=int, default=3)
    parser.add_argument('--heldout-points', type=int, default=16)
    parser.add_argument('--variant', choices=VARIANTS, default='standard')
    parser.add_argument('--fractions', nargs='+', type=float, default=[0., .12, .23, .34, -.12])
    parser.add_argument('--window', type=float, default=12.)
    parser.add_argument('--lead', type=float, default=.9)
    parser.add_argument('--surface-gated', action='store_true')
    parser.add_argument('--practice-only', action='store_true')
    args = parser.parse_args(argv)
    raw = json.loads(args.roles.read_text())
    roles = RoleModel(**raw.get('role_model', raw))
    source = json.loads(args.numeric.read_text())
    candidates = tuple(Pulse(fraction, args.window, args.lead,
                             require_surface=args.surface_gated and bool(fraction))
                       for fraction in args.fractions)
    if not candidates or candidates[0].fraction:
        parser.error('the first candidate must be the unchanged zero pulse')
    screened, seconds = practice(roles, source, candidates, seed=args.practice_seed,
                                  matches=args.matches, points=args.points, variant=args.variant)
    scores = [row['overall']['official_score'] for row in screened]
    selected = int(np.argmax(scores))
    print(dict(scores=scores, selected=asdict(candidates[selected]), wall_seconds=seconds), flush=True)
    qualified = [] if args.practice_only or not selected else run_pool(
        (candidates[0], candidates[selected]), args.heldout_matches, args.heldout_points,
        args.qualification_seed, args.variant, roles, source, save_tape=True)
    receipt = dict(practice_seed=args.practice_seed, qualification_seed=args.qualification_seed,
                   variant=args.variant, source=source, roles=asdict(roles), selected=selected,
                   practice=screened, practice_wall_seconds=seconds, qualification=qualified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as file:
        with gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as output:
            output.write((json.dumps(receipt, separators=(',', ':')) + '\n').encode())


if __name__ == '__main__':
    main()
