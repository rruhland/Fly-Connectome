"""Compare receiving-table learning on identical, unseen point schedules."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt3_arena import TT3Arena, VARIANTS, heldout_cases
from .tt3_learning import ReturnMemory, TableLearner, play_point
from .tt3_visual import RoleModel


# Pair each projected candidate with a different relation. The source scores
# and visit counts are exactly conserved when swapping the meaning of the rows.
WRONG_MAPPING = (2, 4, 0, 5, 1, 3)
WRONG_WAY_MAPPING = (2, 5, 0, 4, 3, 1)


def play_course(cases, role_model, memory, *, seed, probe_frames=16, explore=0.,
                tape_selector=None, learn_roles=True, learner_factory=TableLearner):
    started = perf_counter()
    learner = learner_factory(role_model, memory, seed=seed, probe_frames=probe_frames,
                              explore=explore, learn_roles=learn_roles)
    rows = []
    for number, case in enumerate(cases, 1):
        arena = TT3Arena(case.seed, view_seed=case.view_seed, variant=case.variant,
                         distractor=case.distractor)
        actions = []

        def capture(_frames, action, _reward):
            actions.append(action.tolist())

        row = play_point(arena, learner, capture=capture if tape_selector else None)
        row.update(case=asdict(case), arena_reset=1, point=number)
        if tape_selector:
            tape_selector(row, actions)
        rows.append(row)
    return dict(points=rows, final_return_memory=memory.state(), motor_samples=learner.motor.samples,
                final_role_model=asdict(learner.binder.model), wall_seconds=perf_counter() - started)


def play_open_loop_course(cases, *, seed, random_actions):
    """Use a parked arm or independent legal joint-rate pulses."""
    started = perf_counter()
    rng = np.random.default_rng(seed)
    rows = []
    for number, case in enumerate(cases, 1):
        arena = TT3Arena(case.seed, view_seed=case.view_seed, variant=case.variant,
                         distractor=case.distractor)
        arena.reset(render=False)
        done = False
        frame = 0
        action = np.zeros(3)
        while not done:
            if random_actions and frame % 4 == 0:
                action = rng.uniform(-1., 1., size=3)
            _, reward, done, _ = arena.step(action, render=False)
            frame += 1
        row = arena.point_result()
        row.update(case=asdict(case), arena_reset=1, point=number, public_reward=reward,
                   events=arena.events, option='random' if random_actions else 'parked')
        rows.append(row)
    return dict(points=rows, wall_seconds=perf_counter() - started)


def summarize(courses, max_points=None, *, start=0):
    rows = [row for course in courses for row in course['points'][start:max_points]]
    return dict(points=len(rows), any_contact=sum(row['contacts'] > 0 for row in rows),
                any_legal_return=sum(row['legal_returns'] > 0 for row in rows),
                player_returns=sum(row['legal_returns'] for row in rows),
                rival_returns=sum(row['rival_returns'] for row in rows),
                sustained_rallies=sum(row['legal_returns'] >= 2 and row['rival_returns'] >= 1
                                      for row in rows),
                wins=sum(row['points_for'] for row in rows),
                losses=sum(row['points_against'] for row in rows))


def aggregate(result):
    names = ((1, 0, 'first_point'), (8, 0, 'first_eight'),
             (None, 8, 'after_first_eight'), (None, 0, 'full_course'))
    for end, start, key in names:
        result[key] = {name: summarize(courses, end, start=start)
                       for name, courses in result['results'].items()}
    return result


def qualification(role_model, source, *, suite_seed, courses, points, variant,
                  probe_frames=16, explore=0., cold_roles=None, distractor=False, progress=None,
                  learner_factory=TableLearner):
    if courses < 1 or points < 8:
        raise ValueError('qualification requires at least one course of eight or more points')
    conditions = ['trained', 'fresh', 'wrong']
    if cold_roles is not None:
        conditions.append('cold_roles')
    results = {name: [] for name in conditions}
    replay = None
    for index in range(courses):
        schedule_seed = f'{suite_seed}/course/{index}'
        cases = heldout_cases(schedule_seed, points, variant=variant, distractor=distractor)
        trial_seed = int(np.random.SeedSequence([suite_seed, index]).generate_state(1)[0])

        def save_replay(row, actions):
            nonlocal replay
            if replay is None and row['legal_returns'] >= 2 and row['rival_returns'] >= 1:
                replay = {**row, 'course': index, 'actions': actions}

        for name in conditions:
            if name == 'trained':
                memory = ReturnMemory.restore(source.state())
            elif name == 'wrong':
                memory = source.permuted(WRONG_MAPPING)
            else:
                memory = ReturnMemory()
            model = cold_roles if name == 'cold_roles' else role_model
            result = play_course(cases, model, memory, seed=trial_seed, probe_frames=probe_frames,
                                 explore=explore, tape_selector=save_replay if name == 'trained' else None,
                                 learner_factory=learner_factory)
            result.update(course=index, schedule_seed=schedule_seed, learner_seed=trial_seed)
            results[name].append(result)
            if progress is not None:
                progress(name, index, summarize([result]))
    return aggregate(dict(suite_seed=suite_seed, variant=variant, distractor=distractor,
                          probe_frames=probe_frames, explore=explore,
                          learner=learner_factory.__name__,
                          source_return_memory=source.state(), wrong_mapping=list(WRONG_MAPPING),
                          results=results, earliest_sustained_rally=replay))


def add_controls(result, roles, source, *, progress=None, learner_factory=TableLearner):
    from .tt3_arena import TT3Case

    if result['source_return_memory'] != source.state():
        raise ValueError('receiving comparison requires the same source return memory')
    if result.get('learner', 'TableLearner') != learner_factory.__name__:
        raise ValueError('the comparison requires the same receiving learner')
    names = ('wrong_way', 'parked', 'random')
    additions = {name: [] for name in names}
    for trial in result['results']['trained']:
        cases = [TT3Case(**point['case']) for point in trial['points']]
        seed = trial['learner_seed']
        memory = source.permuted(WRONG_WAY_MAPPING)
        for name in names:
            if name == 'wrong_way':
                course = play_course(cases, roles, memory, seed=seed,
                                     probe_frames=result['probe_frames'], explore=result['explore'],
                                     learner_factory=learner_factory)
            else:
                course = play_open_loop_course(cases, seed=seed, random_actions=name == 'random')
            course.update({key: trial[key] for key in ('course', 'schedule_seed', 'learner_seed')})
            additions[name].append(course)
            if progress is not None:
                progress(name, trial['course'], summarize([course]))
    result['results'].update(additions)
    result['wrong_way_mapping'] = list(WRONG_WAY_MAPPING)
    return aggregate(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--suite-seed', type=int)
    parser.add_argument('--courses', type=int, default=4)
    parser.add_argument('--points', type=int, default=16)
    parser.add_argument('--variant', choices=VARIANTS, default='standard')
    parser.add_argument('--probe-frames', type=int, default=16)
    parser.add_argument('--explore', type=float, default=0.)
    parser.add_argument('--cold-roles', action='store_true')
    parser.add_argument('--distractor', action='store_true')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--existing-trial', type=Path)
    args = parser.parse_args(argv)

    if args.existing_trial is None and args.suite_seed is None:
        parser.error('--suite-seed is required for a new trial')

    checkpoint = json.loads(args.checkpoint.read_text())
    roles = RoleModel(**checkpoint['role_model'])
    source = ReturnMemory.restore(checkpoint['return_memory'])
    cold_roles = RoleModel.untrained() if args.cold_roles else None

    def report(name, index, summary):
        print(json.dumps(dict(condition=name, course=index, **summary)), flush=True)

    if args.existing_trial:
        result = json.loads(args.existing_trial.read_text())
    else:
        result = qualification(roles, source, suite_seed=args.suite_seed, courses=args.courses,
                               points=args.points, variant=args.variant, probe_frames=args.probe_frames,
                               explore=args.explore, cold_roles=cold_roles, distractor=args.distractor,
                               progress=report)
    if args.controls:
        result = add_controls(result, roles, source, progress=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
