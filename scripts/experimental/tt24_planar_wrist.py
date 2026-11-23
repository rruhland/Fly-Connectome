"""Try planar-earned visual credit on the native camera of a different body."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import _write, seeds
from .tt24_motor import optical_sightings
from .tt24_planar_effects import OpticalEffects
from .tt24_planar_learning import LocalCredit, Retina
from .tt24_planar_transfer import CHECKPOINT
from .tt24_source import MODES, VisualConsequence


OUTPUT = Path('artifacts/tt24/planar-wrist.json.gz')


def appearance(observation):
    mover, actor = optical_sightings(observation.get('rgb'))
    if mover is None or actor is None:
        return None
    return actor.center, mover.center, actor.half_span


def play(arena, seed, learner, planner, prior, *, practice=False, hold=False, blind=False, stride=4, scale=.075):
    split = 'practice' if practice else 'held'
    use_camera = not (hold or blind)
    observation = arena.reset(seed=seed, split=split, render=use_camera)
    home = observation['joints']
    retina = Retina(prior, arena.dt * 2)
    optics = retina.measure(appearance(observation))
    planner.reset()
    planner.initial(optics)
    learner.reset()
    actions = []
    cost = 0.
    frames = 0
    done = False
    visible = int(optics.visible)
    while not done:
        before = optics
        effect = planner.hint(prior, stride * arena.dt)
        rate, activity = learner.choose(before, practice=practice, bias=effect)
        reward = 0.
        for offset in range(stride):
            correction = (home - observation['joints']) * .80 - .018 * observation['joint_velocity']
            action = np.zeros(arena.action_size) if hold else np.clip(scale * rate + correction, -.16, .16)
            frames += 1
            sampled = use_camera and frames % 2 == 0
            observation, outcome, done = arena.step(action, render=sampled)
            if sampled:
                optics = retina.measure(appearance(observation))
                visible += int(optics.visible)
            reward += outcome
            if not practice:
                actions.append(action.tolist())
            if done:
                break
        if practice:
            learner.update(before, optics, activity, reward, done)
        if offset == stride - 1:
            planner.observed(rate, optics, plastic=practice)
        cost += reward
    result = arena.judge_metrics() | dict(score=cost, frames=frames, visible=visible, actions=actions,
                                        model_queries=planner.predictions, supported=planner.supported)
    return result


def summary(episodes):
    counts = [row['rebounds'] for row in episodes]
    return dict(total=sum(counts), mean=float(np.mean(counts)), peak=max(counts), counts=counts,
                mean_frames=float(np.mean([row['frames'] for row in episodes])),
                model_support=sum(row['supported'] for row in episodes) / max(1, sum(row['model_queries'] for row in episodes)))


def experiment(arena, options):
    practice = seeds(options.seed, options.practice)
    held = seeds(options.seed + 1, options.held)
    if set(practice) & set(held):
        raise ValueError('wrist practice and held launches must be disjoint')
    rows = dict(schema='tt24-planar-wrist-1', seed=options.seed, checkpoint=str(options.checkpoint),
                checkpoint_sha256=sha256(options.checkpoint.read_bytes()).hexdigest(),
                practice_seeds=practice, held_seeds=held, modes={})
    for mode in MODES:
        prior = VisualConsequence.load(mode, path=options.checkpoint)
        learner = LocalCredit(arena.action_size, np.random.default_rng(options.seed + 2),
                              actor_rate=.035, source_gain=0.)
        planner = OpticalEffects(arena.action_size, source_gain=8., egocentric=True)
        training = [play(arena, launch, learner, planner, prior, practice=True) for launch in practice]
        paired = {name: [play(arena, launch, learner, planner, prior, **control) for launch in held]
                  for name, control in (('live', {}), ('blind', {'blind': True}), ('hold', {'hold': True}))}
        statistics = {name: summary(episodes) for name, episodes in paired.items()}
        print(mode, [row['rebounds'] for row in training], {name: stats['mean'] for name, stats in statistics.items()},
              flush=True)
        rows['modes'][mode] = dict(practice=training, summary=statistics, held=paired, actor=learner.actor.tolist(),
                                 actor_effects=planner.motor.weights.tolist(), mover_effects=planner.mover.weights.tolist())
        _write(options.output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--seed', type=int, default=78091805)
    parser.add_argument('--practice', type=int, default=12)
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.practice, args.held) < 1:
        parser.error('need positive practice and held sample sizes')
    arena = BounceArena(view='wrist', width=64, height=48)
    try:
        experiment(arena, args)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
