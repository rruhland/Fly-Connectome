"""Use a frozen source visual consequence estimate to allocate short target practice."""

import argparse
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import _write, evaluate, play, seeds, summary
from .tt24_motor import MotorEffects, motor_feature_count
from .tt24_motor_modes import survey
from .tt24_source import MODES, VisualConsequence


def candidates(mode_basis, rng, size, readout, radius):
    axes = len(mode_basis)
    features = motor_feature_count(readout)
    proposals = [np.zeros((axes, features))]
    for _ in range(size // 2):
        motor = mode_basis @ rng.normal(size=axes)
        sensory = rng.normal(size=features)
        perturbation = np.outer(motor, sensory) * radius / np.sqrt(features)
        proposals += [perturbation, -perturbation]
    return proposals[:size]


def experiment(arena, *, seed, radius, n_candidates, n_full, n_held, probe_ticks):
    readout = 'rate'
    retinal = 'relative'
    effects = MotorEffects(arena.action_size)
    mode_basis, babble = survey(arena, seed + 4, concentration=1.4, effects=effects)
    proposals = candidates(mode_basis, np.random.default_rng(seed), n_candidates, readout, radius)
    probe_seed, workout_seed = seeds(seed + 1, 2)
    held = seeds(seed + 2, n_held)
    if set([probe_seed, workout_seed] + babble['launches']) & set(held):
        raise ValueError('practice and held launches must be disjoint')
    base_play = dict(split='practice', readout=readout, retinal=retinal, stride=1)
    partitions = {}
    for mode in MODES:
        prior = VisualConsequence.load(mode)
        probes = [play(arena, probe_seed, proposal, effects, prior=prior, max_frames=probe_ticks, **base_play)
                  for proposal in proposals]
        ranking = sorted(range(len(probes)), key=lambda i: -probes[i]['credit'])
        chosen = ranking[:n_full]
        trials = [play(arena, workout_seed, proposals[i], effects, **base_play) for i in chosen]
        best = max(range(len(trials)), key=lambda i: trials[i]['score'] + .25 * min(trials[i]['frames'] / 120, 1.))
        weights = proposals[chosen[best]]
        pairs = evaluate(arena, held, weights, effects, readout=readout, retinal=retinal, stride=1, keep_actions=True)
        aggregate = summary(pairs)
        partitions[mode] = dict(probes=probes, ranked=ranking, chosen=chosen, training=trials, winner=chosen[best],
                                weights=weights.tolist(), held=pairs, summary=aggregate)
        print(mode, 'probe rankings', chosen, 'rewards', [item['score'] for item in trials],
              'held rebounds', {name: score['rebounds'] for name, score in aggregate.items()}, flush=True)
    return dict(seed=seed, readout=readout, retinal=retinal, babble=babble, probe_ticks=probe_ticks,
                per_arm_target_attempts=babble['attempts'] + n_candidates + n_full,
                probe_seed=probe_seed, workout_seed=workout_seed, held_seeds=held,
                motor_effects=effects.readout.tolist(), motor_samples=effects.samples,
                proposals=np.asarray(proposals).tolist(), modes=partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=431202)
    parser.add_argument('--radius', type=float, default=.8)
    parser.add_argument('--candidates', type=int, default=9)
    parser.add_argument('--full', type=int, default=4)
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--probe-ticks', type=int, default=9)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt24/motor-prefix.json.gz'))
    args = parser.parse_args()
    if not 1 < args.full < args.candidates or args.held < 1 or args.probe_ticks < 5:
        parser.error('need enough probes and full trials to rank and select a readout')
    arena = BounceArena(view='wrist', width=64, height=48)
    try:
        data = experiment(arena, seed=args.seed, radius=args.radius, n_candidates=args.candidates, n_full=args.full,
                          n_held=args.held, probe_ticks=args.probe_ticks)
        _write(args.output, dict(schema='tt24-visual-probe-1', **data))
    finally:
        arena.close()


if __name__ == '__main__':
    main()
