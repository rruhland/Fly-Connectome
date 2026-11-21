"""Practice visual stroke timing and force using only the public point payout."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .tt3_arena import VARIANTS, heldout_cases
from .tt3_learning import ReturnMemory, TableLearner, _crossing_time, approach_normal, relation_goal
from .tt3_learning_trial import play_course, summarize
from .tt3_visual import RoleModel


class StrokeMemory:
    """Adjust a carried relation in the observed spatial and temporal units."""

    def __init__(self, carried, offsets):
        self.carried = carried
        self.offsets = np.asarray(offsets, dtype=float)

    def choose(self, rng, explore=0.):
        return self.carried.choose(rng, explore)

    def observe(self, option, reward):
        self.carried.observe(option, reward)

    def relation_name(self, option):
        return self.carried.relation_name(option)

    def state(self):
        return dict(return_memory=self.carried.state(), stroke_offsets=self.offsets.tolist())

    @classmethod
    def restore(cls, state):
        return cls(ReturnMemory.restore(state['return_memory']), state['stroke_offsets'])

    def goal(self, option, actor_position, mover_position, mover_velocity, ready_position,
             *, acceleration=None, normal=None, motion_bounds=None, response_axis=None,
             offset=.03):
        ready = np.asarray(ready_position, dtype=float)
        if mover_position is None or mover_velocity is None:
            return ready.copy()
        position = np.asarray(mover_position)
        velocity = np.asarray(mover_velocity)
        if normal is None:
            normal = approach_normal(position, velocity, ready)
        else:
            normal = np.asarray(normal, dtype=float)
        speed = float(velocity @ normal)
        gap = float((position - ready) @ normal)
        delay, placement, cushion = self.offsets
        shifted = ready + delay * speed * normal if gap * speed < 0. else ready
        goal = relation_goal(option, actor_position, position, velocity, shifted,
                             acceleration=acceleration, normal=normal, motion_bounds=motion_bounds,
                             response_axis=response_axis, offset=offset * (1. + placement))
        if option in (0, 1, 4, 5):
            normal_acceleration = 0. if acceleration is None else float(acceleration @ normal)
            time = _crossing_time(float((position - shifted) @ normal), speed, normal_acceleration)
            if time is not None:
                # The offset changes continuously, so the end effector retains
                # some signed motion when the observed trajectories meet.
                engagement = max(0., 1. - time / 6.)
                goal += 4.5 * cushion * speed * engagement * normal
        return goal


def _population(rng, count, center, spread):
    strata = np.column_stack([(rng.permutation(count) + rng.random(count)) / count
                              for _ in range(len(center))])
    candidates = center + spread * (2. * strata - 1.)
    candidates = np.clip(candidates, (-3., -2., -.45), (5., .7, .55))
    candidates[0] = center
    if count >= 5 and not np.any(center):
        candidates[1:4] = ((0., -1., 0.), (2., -1., .25), (-1., -.5, -.20))
    return candidates


def practice_strokes(role_model, source, *, seed, generations, population, points,
                     variant, progress=None, learner_factory=TableLearner):
    if generations < 1 or population < 3 or points < 6:
        raise ValueError('practice requires at least one generation, three strokes, and six points')
    rng = np.random.default_rng(seed)
    best = np.zeros(3)
    spread = np.array((3.5, 1.35, .5))
    records = []
    fitted_roles = role_model
    fitted_return = source
    for generation in range(generations):
        cases = heldout_cases(f'{seed}/stroke-practice/{generation}', points, variant=variant)
        candidates = _population(rng, population, best, spread)
        runs = []
        for candidate in candidates:
            memory = StrokeMemory(ReturnMemory.restore(fitted_return.state()), candidate)
            course = play_course(cases, fitted_roles, memory, seed=seed + generation,
                                 learner_factory=learner_factory)
            score = float(sum(row['public_reward'] for row in course['points']))
            runs.append(dict(stroke=candidate.tolist(), score=score, course=course))
            if progress is not None:
                progress(generation, candidate, score, summarize([course]))
        index = int(np.argmax([run['score'] for run in runs]))
        records.append(dict(runs=runs, selected=index, generation=generation))
        winner = runs[index]['course']
        best = candidates[index]
        spread *= .65
        fitted_roles = RoleModel(**winner['final_role_model'])
        fitted_return = ReturnMemory.restore(winner['final_return_memory']['return_memory'])
    return best, fitted_roles, fitted_return, records


def qualify_strokes(cases, roles, source, stroke, *, seed, learner_factory=TableLearner):
    replay = None

    def capture(row, actions):
        nonlocal replay
        if replay is None and row['legal_returns'] >= 2 and row['rival_returns'] >= 1:
            replay = {**row, 'actions': actions}

    scores = {}
    for name, offsets in (('unchanged', np.zeros(3)), ('practiced', stroke)):
        memory = StrokeMemory(ReturnMemory.restore(source.state()), offsets)
        course = play_course(cases, roles, memory, seed=seed,
                             tape_selector=capture if name == 'practiced' else None,
                             learner_factory=learner_factory)
        scores[name] = dict(course=course, summary=summarize([course]))
    return dict(results=scores, earliest_sustained_rally=replay)


def stroke_learner_factory(*, forecast=False, integrated_motor=False):
    if forecast:
        from .tt3_flight import ForecastLearner
    if integrated_motor:
        from .tt3_motor import IntegratedTableLearner

    if forecast and integrated_motor:
        class IntegratedForecastLearner(IntegratedTableLearner, ForecastLearner):
            pass

        return IntegratedForecastLearner
    if forecast:
        return ForecastLearner
    if integrated_motor:
        return IntegratedTableLearner
    return TableLearner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--qualification-seed', type=int, required=True)
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--population', type=int, default=6)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--qualification-points', type=int, default=32)
    parser.add_argument('--variant', choices=VARIANTS, default='standard')
    parser.add_argument('--qualification-variant', choices=VARIANTS, default='standard')
    parser.add_argument('--qualification-distractor', action='store_true')
    parser.add_argument('--forecast', action='store_true')
    parser.add_argument('--integrated-motor', action='store_true')
    args = parser.parse_args(argv)
    checkpoint = json.loads(args.checkpoint.read_text())
    roles = RoleModel(**checkpoint['role_model'])
    source = ReturnMemory.restore(checkpoint['return_memory'])
    learner_factory = stroke_learner_factory(forecast=args.forecast, integrated_motor=args.integrated_motor)

    def report(generation, candidate, score, summary):
        print(json.dumps(dict(generation=generation, stroke=candidate.tolist(), score=score,
                              **summary)), flush=True)

    best, refined_roles, refined_return, training = practice_strokes(
        roles, source, seed=args.seed, generations=args.generations, population=args.population,
        points=args.points, variant=args.variant, progress=report, learner_factory=learner_factory)
    cases = heldout_cases(f'{args.qualification_seed}/stroke-qualification',
                          args.qualification_points, variant=args.qualification_variant,
                          distractor=args.qualification_distractor)
    result = qualify_strokes(cases, refined_roles, refined_return, best, seed=args.qualification_seed,
                             learner_factory=learner_factory)
    payload = dict(seed=args.seed, qualification_seed=args.qualification_seed, training=training,
                   forecast=args.forecast, integrated_motor=args.integrated_motor,
                   practice_variant=args.variant,
                   qualification_variant=args.qualification_variant,
                   qualification_distractor=args.qualification_distractor,
                   source_checkpoint=checkpoint,
                   stroke=best.tolist(), return_memory=refined_return.state(),
                   role_model=asdict(refined_roles), qualification=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({name: row['summary'] for name, row in result['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
