"""Compare Pong credit, cold credit and label-shuffled credit in planar gravity."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt24_bounce_trial import _write, seeds
from .tt24_planar import PlanarBounce, STAGES
from .tt24_planar_effects import OpticalEffects
from .tt24_planar_learning import LocalCredit, Retina
from .tt24_source import CHECKPOINT, MODES, VisualConsequence


OUTPUT = Path('artifacts/tt24/planar-trial.json.gz')


def play(arena, seed, learner, prior, *, practice=False, hold=False, stride=4, witness=False, planner=None,
         fixed_vertical=False, trace=False):
    retina = Retina(prior, arena.dt)
    if learner is not None:
        learner.reset()
    optics = retina.update(arena.reset(seed))
    if planner is not None:
        planner.reset()
        planner.initial(optics)
    score = 0.
    actions, credit, support, sensor_trace = [], [], [], []
    visible = 0
    done = False
    while not done:
        before = optics
        bias = None if planner is None else planner.hint(prior, arena.dt * stride)
        if hold:
            drive, activity = (0.,) * arena.action_size, None
        else:
            drive, activity = learner.choose(before, practice=practice, bias=bias)
        if fixed_vertical:
            drive[1] = 0.
        reward = 0.
        for tick in range(stride):
            rgb, amount, done = arena.step(drive)
            reward += amount
            optics = retina.update(rgb, credit=practice and (tick == stride - 1 or done))
            visible += int(optics.visible)
            if witness:
                actions.append(list(drive))
            if done:
                break
        if practice:
            learner.update(before, optics, activity, reward, done)
            credit.append(optics.potential)
            support.append(optics.supported)
        if planner is not None and tick == stride - 1:
            planner.observed(drive, optics, plastic=practice)
        if trace:
            sensor_trace.append((before.appearance, reward))
        score += reward
    result = arena.judge() | dict(score=score, visible=visible, source_support=float(np.mean(support)) if support else 0.,
                                 source_credit=float(np.mean(credit)) if credit else 0.)
    if score != result['rebounds']:
        raise RuntimeError('the scalar reward must match verified rebound events')
    if planner is not None:
        result['model_coverage'] = planner.supported / max(1, planner.predictions)
    if witness:
        result['actions'] = actions
    if trace:
        result['trace'] = sensor_trace
    return result


def summarize(rows):
    outcomes = np.array([row['score'] for row in rows])
    powered = np.mean([sum(event['paddle_rate'] > .05 for event in row['events']) for row in rows])
    return dict(mean=float(outcomes.mean()), total=int(outcomes.sum()), peak=int(outcomes.max()),
                any=float(np.mean(outcomes >= 1)), multiple=float(np.mean(outcomes >= 2)),
                powered=float(powered), frames=float(np.mean([row['frames'] for row in rows])),
                scores=outcomes.astype(int).tolist())


def train(arena, practice_seeds, held, prior, *, options, seed, checkpoints, weights=None):
    planner = OpticalEffects(arena.action_size, source_gain=options.source_gain) if options.rule == 'planner' else None
    learner = LocalCredit(arena.action_size, np.random.default_rng(seed), weights=weights,
                          actor_rate=options.actor_rate, critic_rate=options.critic_rate,
                          source_gain=options.source_gain if planner is None else 0.)
    practice, evaluations = [], {}
    record = dict(mode=prior.mode, practice=practice, held=evaluations)
    for count in range(len(practice_seeds) + 1):
        if count in checkpoints:
            evaluations[str(count)] = summarize([play(arena, seed, learner, prior, stride=options.stride, planner=planner)
                                                 for seed in held])
        if count < len(practice_seeds):
            point = play(arena, practice_seeds[count], learner, prior, practice=True, stride=options.stride,
                         planner=planner)
            used = ('seed', 'frames', 'score', 'visible', 'source_support', 'source_credit', 'model_coverage')
            practice.append({key: point[key] for key in used if key in point})

    record['actor'] = learner.actor.tolist()
    record['critic'] = learner.value.tolist()
    if arena.action_size == 2:
        record['fixed_vertical'] = summarize([play(arena, launch, learner, prior, stride=options.stride, planner=planner,
                                                  fixed_vertical=True) for launch in held])
    if planner is not None:
        record['optical_effects'] = dict(actor=planner.motor.weights.tolist(), mover=planner.mover.weights.tolist(),
                                         actor_samples=planner.motor.samples, mover_samples=planner.mover.samples)
    print(arena.stage, seed, prior.mode, {i: round(row['mean'], 3) for i, row in evaluations.items()}, flush=True)
    return record


def run(options):
    begin = perf_counter()
    checkpoints = sorted({0, options.practice, *(int(part) for part in options.checkpoints.split(',')
                                                 if part and int(part) <= options.practice)})
    stages = ('lateral', 'energy') if options.stage == 'both' else (options.stage,)
    record = dict(schema='tt24-planar-visual-credit-1', settings=vars(options) | dict(output=str(options.output)),
                  source_sha256=sha256(CHECKPOINT.read_bytes()).hexdigest(), cohorts=[])
    priors = {name: VisualConsequence.load(name) for name in MODES}
    for seed in seeds(options.seed, options.cohorts):
        entry = dict(seed=seed, stages={})
        for index, name in enumerate(stages):
            arena = PlanarBounce(name, max_frames=options.max_frames, presentation=options.presentation)
            trial_seeds = seeds(seed + 19 * index, options.practice + options.held)
            practice, held = trial_seeds[:options.practice], trial_seeds[options.practice:]
            if set(practice) & set(held):
                raise ValueError('practice and held launches must be disjoint')
            passive = summarize([play(arena, case, None, priors['cold'], hold=True, stride=options.stride) for case in held])
            phase = dict(practice_seeds=practice, held_seeds=held, hold=passive, modes={})
            for mode in MODES:
                donor = None
                if options.carry and name == 'energy':
                    donor = np.asarray(entry['stages']['lateral']['modes'][mode]['actor'])
                phase['modes'][mode] = train(arena, practice, held, priors[mode], options=options, seed=seed + 110 + index,
                                             checkpoints=checkpoints, weights=donor)
            entry['stages'][name] = phase
        record['cohorts'].append(entry)
        record['elapsed_s'] = perf_counter() - begin
        _write(options.output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'both'), default='both')
    parser.add_argument('--presentation', choices=('fixed', 'varied'), default='fixed')
    parser.add_argument('--cohorts', type=int, default=3)
    parser.add_argument('--practice', type=int, default=48)
    parser.add_argument('--held', type=int, default=24)
    parser.add_argument('--seed', type=int, default=98483241)
    parser.add_argument('--max-frames', type=int, default=240)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--actor-rate', type=float, default=.075)
    parser.add_argument('--critic-rate', type=float, default=.09)
    parser.add_argument('--source-gain', type=float, default=1.)
    parser.add_argument('--rule', choices=('temporal', 'planner'), default='temporal')
    parser.add_argument('--checkpoints', default='4,8,16,24,48,96,192')
    parser.add_argument('--carry', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    options = parser.parse_args()
    if min(options.cohorts, options.practice, options.held, options.stride) < 1:
        parser.error('sample counts and stride must be positive')
    if options.carry and options.stage != 'both':
        parser.error('carry requires both planar stages')
    run(options)


if __name__ == '__main__':
    main()
