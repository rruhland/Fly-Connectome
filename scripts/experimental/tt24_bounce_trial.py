"""Run reward-only visual motor practice and paired held SO-101 evaluations."""

import argparse
from copy import deepcopy
from hashlib import sha256
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt24_motor import MotorEffects, MotorPolicy, RewardSearch, motor_feature_count
from .tt24_motor_portfolio import Portfolio, context


OUTPUT = Path('artifacts/tt24/motor-bounce.json.gz')


def play(arena, seed, weights, effects, *, split, prior=None, plastic=False, blind=False, hold=False,
         keep_actions=False, record_rgb=False, stride=2, regime=None, readout='position', retinal='absolute',
         credit_tau=None, portfolio=None, max_frames=None, vision='hue'):
    detector = None
    if vision == 'anonymous':
        if portfolio is not None:
            raise ValueError('anonymous vision needs a retinal history before choosing a contextual readout')
        if not blind:
            from .tt24_anonvision import AnonymousVision

            detector = AnonymousVision()
    elif vision != 'hue':
        raise ValueError('unknown detector')
    reset_options = {} if regime is None else {'regime': regime}
    render = not blind or record_rgb
    obs = arena.reset(seed=seed, split=split, render=render, **reset_options)
    visible = {} if blind else obs
    initial_context = context(visible) if vision == 'hue' else None
    if portfolio is not None:
        weights = portfolio.choose(visible)
    policy = MotorPolicy(weights, effects, readout=readout, retinal=retinal, detector=detector)
    credit, support, confidence_weights = [], [], []
    actions = []
    digest = sha256() if record_rgb else None
    total = 0.
    peak_reward = 0.
    frames = 0
    done = False
    while not done and (max_frames is None or frames < max_frames):
        sampled = frames % stride == 0
        if digest is not None and sampled:
            digest.update(obs['rgb'])
        action = np.zeros(arena.action_size) if hold else policy.command(obs, plastic=plastic, blind=blind,
                                                                         sampled=sampled)
        if keep_actions:
            actions.append(action.tolist())
        if sampled:
            predicted = policy.optics.consequence(prior, arena.dt * stride)
            if predicted is not None:
                delta, confidence = predicted
                credit.append(delta)
                support.append(confidence)
                confidence_weights.append(1. if credit_tau is None else np.exp(-frames * arena.dt / credit_tau))
        obs, reward, done = arena.step(action, render=render and (frames + 1) % stride == 0)
        total += float(reward)
        peak_reward = max(peak_reward, float(reward))
        frames += 1

    result = arena.judge_metrics() | dict(score=total, peak_reward=peak_reward, frames=frames,
                                         initial_context=None if initial_context is None else initial_context.tolist(),
                                         credit=float(np.average(credit, weights=confidence_weights)) if credit else 0.,
                                         source_support=float(np.mean(support)) if support else 0.,
                                         credit_windows=len(credit), ball_frames=policy.optics.visible,
                                         paired_frames=policy.optics.paired)
    if keep_actions:
        result['actions'] = actions
    if digest is not None:
        result['rgb_sha256'] = digest.hexdigest()
    return result


def verify_replay(arena, episode, *, stride=1, record_rgb=False):
    """Re-execute recorded commands and check every physical rebound event."""
    regime = episode.get('regime')
    reset_options = {} if regime is None else {'regime': regime}
    obs = arena.reset(seed=episode['seed'], split=episode['split'], render=record_rgb, **reset_options)
    digest = sha256() if record_rgb else None
    total = 0.
    done = False
    for tick, action in enumerate(episode['actions']):
        if done:
            raise ValueError('the recorded actions continue beyond the end of the drop')
        if digest is not None and tick % stride == 0:
            digest.update(obs['rgb'])
        obs, reward, done = arena.step(action, render=record_rgb and (tick + 1) % stride == 0)
        total += reward
    outcome = arena.judge_metrics()
    for key in ('steps', 'termination', 'paddle_contacts', 'rebounds'):
        if json.dumps(outcome[key], sort_keys=True) != json.dumps(episode[key], sort_keys=True):
            raise ValueError(f'action replay changed {key} on seed {episode["seed"]}')
    recorded = episode['rebound_events']
    measured = [{key: actual[key] for key in expected}
                for actual, expected in zip(outcome['rebound_events'], recorded, strict=True)]
    if json.dumps(measured, sort_keys=True) != json.dumps(recorded, sort_keys=True):
        raise ValueError(f'action replay changed rebound events on seed {episode["seed"]}')
    if not done or abs(total - episode['score']) > 1.e-9:
        raise ValueError('action replay changed completion or reward')
    if digest is not None:
        rgb = digest.hexdigest()
        if 'rgb_sha256' in episode and rgb != episode['rgb_sha256']:
            raise ValueError('action replay changed retinal images')
        return rgb
    return None


def search(arena, seeds, prior, effects, *, seed, weights=None, radius=.45, credit_weight=1., log=False,
           split='practice', regime=None, stride=2, readout='position', repeats=1, motor_modes=None, retinal='absolute',
           credit_tau=None, vision='hue'):
    learner = RewardSearch(arena.action_size, seed, weights=weights, radius=radius, motor_modes=motor_modes,
                           features=motor_feature_count(readout))
    rows = []
    for start in range(0, len(seeds), repeats):
        weights = learner.propose()
        trial_rows = [play(arena, launch, weights, effects, split=split, prior=prior, plastic=True, regime=regime,
                           stride=stride, readout=readout, retinal=retinal, credit_tau=credit_tau, vision=vision)
                      for launch in seeds[start:start + repeats]]
        learned_credit = credit_weight * np.mean([row['credit'] for row in trial_rows])
        reward = np.mean([row['score'] for row in trial_rows])
        survival = np.mean([.25 * min(row['frames'] / 120, 1.) for row in trial_rows])
        learner.update(weights, reward, learned_credit + survival)
        rows.extend(row | dict(weights=weights.tolist(), selected=float(learner.elite[0][0])) for row in trial_rows)
        if log:
            print('practice', len(rows), [row['seed'] for row in trial_rows], [row['score'] for row in trial_rows],
                  round(learned_credit, 3), round(float(learner.elite[0][0]), 3), flush=True)
    return learner.best.copy(), rows


def evaluate(arena, seeds, weights, effects, *, split='held', controls=True, keep_actions=False, record_rgb=False,
             regime=None, stride=2, readout='position', retinal='absolute', vision='hue'):
    conditions = {'live': {}}
    if controls:
        conditions |= {'blind': {'blind': True}, 'hold': {'blind': True, 'hold': True}}
    static = weights.copy()
    if controls and readout in ('position', 'rate'):
        conditions['static'] = {'blind': True}
        static[:, 0] += weights[:, 1]
    rows = {}
    for name, settings in conditions.items():
        rows[name] = [play(arena, seed, static if name == 'static' else weights, effects, split=split,
                           keep_actions=keep_actions, record_rgb=record_rgb, regime=regime, stride=stride, readout=readout,
                           retinal=retinal, vision=vision, **settings) for seed in seeds]
    return rows


def seeds(seed, n):
    return np.random.default_rng(seed).integers(1, 2 ** 31 - 1, n).tolist()


def summary(rows):
    records = {}
    for name, entries in rows.items():
        records[name] = {key: float(np.mean([episode[key] for episode in entries]))
                         for key in ('rebounds', 'score', 'frames', 'ball_frames', 'paired_frames')}
        records[name]['rebound_counts'] = [entry['rebounds'] for entry in entries]
        records[name]['max_rebounds'] = max(records[name]['rebound_counts'])
    return records


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        pending = Path(staging) / path.name
        with gzip.open(pending, 'wt') as stream:
            json.dump(record, stream, separators=(',', ':'))
        pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pong', 'cold', 'shuffled', 'none'), default='none')
    parser.add_argument('--view', choices=('wrist', 'external'), default='wrist')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--practice-seed', type=int, default=123)
    parser.add_argument('--held-seed', type=int)
    parser.add_argument('--practice', type=int, default=24)
    parser.add_argument('--practice-contexts', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1, help='native practice episodes per proposed readout')
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--changed-practice', type=int, default=0)
    parser.add_argument('--changed-arm', choices=('stock', 'alternate'), default='stock')
    parser.add_argument('--track-drive', action='store_true', help='continue adapting measured channel gains')
    parser.add_argument('--radius', type=float, default=.45)
    parser.add_argument('--readout', choices=('position', 'rate', 'flow', 'recurrent'), default='position')
    parser.add_argument('--retinal', choices=('absolute', 'relative'), default='absolute')
    parser.add_argument('--vision', choices=('hue', 'anonymous'), default='hue')
    parser.add_argument('--stride', type=int, default=2, help='sample RGB every N control ticks')
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=72)
    parser.add_argument('--source-credit', type=float, default=1.)
    parser.add_argument('--credit-tau', type=float, help='discount later optical credit by this many seconds')
    parser.add_argument('--motor-survey', action='store_true')
    parser.add_argument('--motor-concentration', type=float, default=.5, help='preference for quieter retinal modes')
    parser.add_argument('--portfolio', action='store_true', help='test a contextual selection of learned readouts')
    parser.add_argument('--initial', type=Path, help='continue an existing target readout')
    parser.add_argument('--apex-reward', type=float, default=0., help='capped post-rebound apex bonus weight')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.practice_contexts, args.held, args.stride, args.repeats) < 1 or min(args.practice, args.changed_practice) < 0:
        parser.error('held, stride and context settings must be positive, with nonnegative practice')
    if args.credit_tau is not None and args.credit_tau <= 0.:
        parser.error('optical credit time constant must be positive')
    if args.portfolio and not (args.repeats == args.practice_contexts and 2 <= args.repeats < args.practice):
        parser.error('a portfolio needs at least two complete, matched practice context blocks')
    if args.portfolio and args.vision == 'anonymous':
        parser.error('anonymous vision needs a retinal history before choosing a contextual readout')

    from .tt24_bounce import BounceArena

    prior = None
    if args.mode != 'none':
        from .tt24_source import VisualConsequence

        prior = VisualConsequence.load(args.mode)
    arena_options = {} if args.apex_reward == 0. else {'apex_reward_weight': args.apex_reward}
    arena = BounceArena(view=args.view, width=args.width, height=args.height, **arena_options)
    started = perf_counter()
    try:
        effects = MotorEffects(arena.action_size)
        motor_modes = receipt = None
        initial_weights = None
        previous = None
        if args.initial is not None:
            with gzip.open(args.initial, 'rt') as stream:
                previous = json.load(stream)
            for key in ('view', 'width', 'height', 'stride', 'retinal', 'readout'):
                if key in previous and previous[key] != getattr(args, key):
                    raise ValueError(f'initial checkpoint has different {key}')
            initial_weights = np.asarray(previous['weights'])
            effects.readout[:] = previous['motor_effects']
            effects.samples = previous['motor_samples']
            if 'motor_precision' in previous:
                effects.precision[:] = previous['motor_precision']
            if previous.get('motor_modes') is not None:
                motor_modes = np.asarray(previous['motor_modes'])
        if args.motor_survey:
            from .tt24_motor_modes import survey

            motor_modes, receipt = survey(arena, args.practice_seed + 22, concentration=args.motor_concentration)
        starts = seeds(args.practice_seed, args.practice_contexts)
        schedule = [starts[idx % len(starts)] for idx in range(args.practice)]
        if schedule:
            weights, training = search(arena, schedule, prior, effects, seed=args.seed, radius=args.radius, log=True,
                                       stride=args.stride, credit_weight=args.source_credit, readout=args.readout,
                                       repeats=args.repeats, motor_modes=motor_modes, retinal=args.retinal,
                                       credit_tau=args.credit_tau, weights=initial_weights, vision=args.vision)
        else:
            shape = (arena.action_size, motor_feature_count(args.readout))
            weights = np.zeros(shape) if initial_weights is None else initial_weights
            training = []
        initial_attempts = 0
        if previous:
            old_babble = previous.get('motor_survey') or {}
            initial_attempts = previous.get('total_initial_attempts', len(previous['training']) + old_babble.get('attempts', 0))
        new_attempts = len(training) + (receipt['attempts'] if receipt else 0)
        record = dict(schema='tt24-direct-motor-1', mode=args.mode, view=args.view, vision=args.vision,
                      seed=args.seed, stride=args.stride,
                      width=args.width, height=args.height, practice_seeds=schedule, training=training,
                      readout=args.readout, repeats=args.repeats, retinal=args.retinal, credit_tau=args.credit_tau,
                      source_credit=args.source_credit, radius=args.radius, apex_reward_weight=args.apex_reward,
                      initial=str(args.initial) if args.initial else None,
                      total_initial_attempts=initial_attempts + new_attempts,
                      motor_survey=receipt, motor_modes=None if motor_modes is None else motor_modes.tolist(),
                      weights=weights.tolist(), motor_effects=effects.readout.tolist(), motor_samples=effects.samples,
                      motor_precision=effects.precision.tolist())
        _write(args.output, record)
        held = seeds(args.held_seed if args.held_seed is not None else args.practice_seed + 1, args.held)
        if set(schedule) & set(held):
            raise ValueError('practice and held launches must be disjoint')
        evaluations = evaluate(arena, held, weights, deepcopy(effects), keep_actions=True, stride=args.stride,
                               readout=args.readout, retinal=args.retinal, vision=args.vision)
        if args.portfolio:
            portfolio = Portfolio(training, args.repeats)
            evaluations['portfolio'] = [play(arena, launch, weights, effects, split='held', stride=args.stride,
                                             readout=args.readout, retinal=args.retinal, keep_actions=True,
                                             portfolio=portfolio) for launch in held]
        scores = summary(evaluations)
        print('standard', scores, flush=True)
        record |= dict(held=evaluations, summary=scores, wall_seconds=perf_counter() - started)
        _write(args.output, record)
        if args.changed_practice:
            if args.changed_arm != 'stock':
                arena.close()
                arena = BounceArena(view=args.view, width=args.width, height=args.height, arm=args.changed_arm,
                                    **arena_options)
            changed_start = seeds(args.practice_seed + 2, args.practice_contexts)
            schedule = [changed_start[idx % len(changed_start)] for idx in range(args.changed_practice)]
            if set(schedule) & set(held):
                raise ValueError('changed practice and held launches must be disjoint')
            changed_effects = deepcopy(effects)
            if args.track_drive:
                changed_effects.track_change()
            adapted, attempts = search(arena, schedule, prior, changed_effects, seed=args.seed + 1, weights=weights,
                                       radius=.65 * args.radius, log=True, stride=args.stride, regime='changed',
                                       credit_weight=args.source_credit, readout=args.readout, repeats=args.repeats,
                                       motor_modes=motor_modes, retinal=args.retinal, credit_tau=args.credit_tau,
                                       vision=args.vision)
            adapted_rows = evaluate(arena, held, adapted, changed_effects, stride=args.stride, regime='changed',
                                    keep_actions=True, readout=args.readout, retinal=args.retinal, vision=args.vision)
            frozen = evaluate(arena, held, weights, effects, stride=args.stride, regime='changed', controls=False,
                              keep_actions=True, readout=args.readout, retinal=args.retinal, vision=args.vision)['live']
            changed = dict(online=adapted_rows.pop('live'), frozen=frozen, **adapted_rows)
            if args.track_drive:
                changed['gain_only'] = evaluate(arena, held, weights, changed_effects, stride=args.stride,
                                                regime='changed', controls=False, keep_actions=True,
                                                readout=args.readout, retinal=args.retinal, vision=args.vision)['live']
            changed_scores = summary(changed)
            print('changed', changed_scores, flush=True)
            record |= dict(changed=dict(arm=args.changed_arm, held=changed, summary=changed_scores, training=attempts,
                                        weights=adapted.tolist(), motor_effects=changed_effects.readout.tolist(),
                                        motor_samples=changed_effects.samples,
                                        motor_precision=changed_effects.precision.tolist(),
                                        gain=changed_effects.gain().tolist(), frozen_gain=effects.gain().tolist()),
                           wall_seconds=perf_counter() - started)
            _write(args.output, record)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
