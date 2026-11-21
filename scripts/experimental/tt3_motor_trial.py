"""Compare the native visual servo and an integrated-command motor model."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt3_arena import TT3Arena, heldout_cases
from .tt3_learning import ReturnMemory, TableLearner, play_point
from .tt3_learning_trial import summarize
from .tt3_motor import COMMAND_SCALE, IntegratedTableLearner
from .tt3_visual import RoleModel


def visual_body_practice(learner, cases, seed, *, frames_per_scene=30):
    """Issue centered action probes in fresh camera scenes without using score."""
    rng = np.random.default_rng(seed)
    visits = 0
    for case in cases:
        arena = TT3Arena(case.seed, view_seed=case.view_seed, variant=case.variant)
        images = arena.reset()
        learner.reset()
        for tick in range(frames_per_scene):
            action = learner.act(images)
            if tick >= 8:
                if tick % 8 == 0:
                    goal = rng.uniform(.52, 1., size=3) * rng.choice((-1., 1.), size=3)
                delta = (goal - learner.motor.coordinates) * COMMAND_SCALE
                action = np.clip(delta, -1., 1.)
                learner.action = action
            images, _, done, _ = arena.step(action)
            visits += 1
            if done:
                break
    return dict(scenes=len(cases), camera_frames=visits, body_cells=len(learner.motor.body.cells))


def run(checkpoint, *, suite_seed, courses, points, practice_scenes, variant, condition,
        fixed_option=None, gain=.22, quiet=0., mix=.85):
    roles = RoleModel(**checkpoint['role_model'])
    source = ReturnMemory.restore(checkpoint['return_memory'])
    kwargs = dict(learn_roles=True)
    rows = []
    for index in range(courses):
        started = perf_counter()
        memory = ReturnMemory.restore(source.state())
        learner_seed = int(np.random.SeedSequence([suite_seed, index]).generate_state(1)[0])
        if condition == 'local':
            learner = TableLearner(roles, memory, seed=learner_seed, **kwargs)
        else:
            learner = IntegratedTableLearner(roles, memory, seed=learner_seed, gain=gain,
                                             quiet=quiet, mix=mix, **kwargs)
        practice = None
        if condition == 'practice':
            cases = heldout_cases(f'{suite_seed}/motor-practice/{index}', practice_scenes, variant=variant)
            explorer = IntegratedTableLearner(roles, ReturnMemory.restore(source.state()), seed=learner_seed)
            explorer.motor = learner.motor
            practice = visual_body_practice(explorer, cases, learner_seed)
        cases = heldout_cases(f'{suite_seed}/received/{index}', points, variant=variant)
        course = []
        for case in cases:
            arena = TT3Arena(case.seed, view_seed=case.view_seed, variant=case.variant)
            row = play_point(arena, learner, option=fixed_option)
            row['case'] = asdict(case)
            course.append(row)
        rows.append(dict(points=course, practice=practice, wall_seconds=perf_counter() - started))
        print(json.dumps(dict(condition=condition, course=index, **summarize(rows[-1:]))), flush=True)
    return dict(condition=condition, variant=variant, suite_seed=suite_seed, fixed_option=fixed_option,
                gain=gain, quiet=quiet, mix=mix,
                course_details=rows, summary=summarize(rows),
                first_eight=summarize(rows, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--condition', choices=('local', 'online', 'practice'), required=True)
    parser.add_argument('--suite-seed', type=int, required=True)
    parser.add_argument('--variant', default='standard')
    parser.add_argument('--courses', type=int, default=4)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--practice-scenes', type=int, default=6)
    parser.add_argument('--fixed-option', type=int)
    parser.add_argument('--gain', type=float, default=.22)
    parser.add_argument('--quiet', type=float, default=0.)
    parser.add_argument('--mix', type=float, default=.85)
    args = parser.parse_args()
    data = run(json.loads(args.checkpoint.read_text()), suite_seed=args.suite_seed, courses=args.courses,
               points=args.points, practice_scenes=args.practice_scenes, variant=args.variant,
               condition=args.condition, fixed_option=args.fixed_option, gain=args.gain, quiet=args.quiet,
               mix=args.mix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
