"""Learn rebound-value priors from live, visually varied planar drops."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .tt24_bounce_trial import _write, seeds
from .tt24_planar import PlanarBounce
from .tt24_planar_effects import OpticalEffects
from .tt24_planar_learning import LocalCredit
from .tt24_planar_trial import play
from .tt24_source import FEATURES, MODES, VisualConsequence, basis, visual_features


ROOT = Path('artifacts/tt24')
SOURCE = ROOT / 'planar-planner.json.gz'
CHECKPOINT = ROOT / 'planar-value.json'


def visual_targets(trace, dt, horizon):
    future = int(round(horizon / dt))
    observations, rewards = zip(*trace, strict=True)
    rows = []
    for i in range(4, len(observations), 2):
        recent = observations[i - 4:i + 1]
        if any(item is None for item in recent):
            continue
        actor, mover, scales = zip(*recent, strict=True)
        cue = visual_features(np.asarray(actor), np.asarray(mover), dt=dt, scale=float(np.median(scales)))
        if np.isfinite(cue).all():
            rows.append((cue.tolist(), sum(rewards[i:i + future]) >= 1., i))
    return rows


def collect(controllers, launches, *, stride, horizon):
    prior = VisualConsequence.load('pong')
    arenas = {name: PlanarBounce(name, presentation='varied') for name in ('lateral', 'energy')}
    episodes = []
    for i, launch in enumerate(launches):
        stage, record = controllers[i % len(controllers)]
        arena = arenas[stage]
        learner = LocalCredit(arena.action_size, np.random.default_rng(launch), weights=np.asarray(record['actor']))
        dynamics = record['optical_effects']
        planner = OpticalEffects(arena.action_size, source_gain=8.)
        planner.motor.weights[:] = dynamics['actor']
        planner.mover.weights[:] = dynamics['mover']
        planner.motor.samples = dynamics['actor_samples']
        planner.mover.samples = dynamics['mover_samples']
        result = play(arena, launch, learner, prior, stride=stride, planner=planner, trace=True)
        rows = visual_targets(result['trace'], arena.dt * stride, horizon)
        episodes.append(dict(seed=launch, stage=stage, frames=result['frames'], rebounds=result['rebounds'], rows=rows))
    return episodes


def _joined(episodes):
    rows = [(x, label, tick, episode['stage'] == 'energy')
            for episode in episodes for x, label, tick in episode['rows']]
    return tuple(np.asarray(column) for column in zip(*rows, strict=True))


def fit(episodes, *, shuffle_seed=9765902):
    x, y, ticks, energy = _joined(episodes)
    prior = float(y.mean())
    center = np.median(x, axis=0)
    scale = np.maximum(x.std(axis=0), .12)
    projection = basis((x - center) / scale)
    penalty = np.eye(projection.shape[1]) * max(1., .002 * len(x))
    penalty[0, 0] *= .05
    shuffle = y.copy()
    rng = np.random.default_rng(shuffle_seed)
    blocks = np.column_stack((energy, ticks // 6, np.floor(x[:, 0])))
    for group in np.unique(blocks, axis=0):
        idx = np.flatnonzero(np.all(blocks == group, axis=1))
        shuffle[idx] = rng.permutation(y[idx])
    fit_targets = {'pong': y, 'shuffled': shuffle}
    covariance = projection.T @ projection + penalty
    weights = {name: np.linalg.solve(covariance, projection.T @ target).tolist()
               for name, target in fit_targets.items()}
    weights['cold'] = [prior] + [0.] * (projection.shape[1] - 1)
    core = np.quantile(x, (.015, .985), axis=0)
    fringe = np.maximum(.32 * np.ptp(core, axis=0), scale)
    return dict(schema='tt24-visual-duration-source-1', features=list(FEATURES), prior=prior,
                mean=center.tolist(), scale=scale.tolist(), weights=weights, core=core.tolist(),
                outer=(core + np.array((-1., 1.))[:, None] * fringe).tolist(), examples=len(y))


def qualify(checkpoint, episodes):
    x, y, _, _ = _joined(episodes)
    n = int(y.sum())
    if n in (0, len(y)):
        raise ValueError('held rebound-value corpus needs both outcomes')
    output = {}
    for mode in MODES:
        result = VisualConsequence.from_checkpoint(mode, checkpoint).score_features(x, details=True)
        score = result['score']
        rank = rankdata(score)[y].sum()
        auc = (rank - n * (n + 1) / 2) / (n * (len(y) - n))
        output[mode] = dict(auc=float(auc), brier=float(np.mean((score - y) ** 2)),
                            mean_support=float(result['support'].mean()), examples=len(y), positives=n)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controllers', type=Path, default=SOURCE)
    parser.add_argument('--train', type=int, default=192)
    parser.add_argument('--held', type=int, default=72)
    parser.add_argument('--horizon', type=float, default=.65)
    parser.add_argument('--seed', type=int, default=19784210)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=ROOT / 'planar-value-data.json.gz')
    args = parser.parse_args()
    with gzip.open(args.controllers, 'rt') as stream:
        evidence = json.load(stream)
    if evidence['settings']['rule'] != 'planner':
        parser.error('expected closed-loop optical-effects source controllers')
    controllers = [(stage, rollouts['modes']['pong']) for cohort in evidence['cohorts'][:4]
                   for stage, rollouts in cohort['stages'].items()]
    if not controllers:
        parser.error('need a completed controller cohort')
    stride = evidence['settings']['stride']
    launch_seeds = seeds(args.seed, args.train + args.held)
    train = collect(controllers, launch_seeds[:args.train], stride=stride, horizon=args.horizon)
    test = collect(controllers, launch_seeds[args.train:], stride=stride, horizon=args.horizon)
    checkpoint = fit(train)
    held = qualify(checkpoint, test)
    checkpoint |= dict(held=held, source_receipt_sha256=sha256(args.controllers.read_bytes()).hexdigest(),
                       label_horizon_s=args.horizon, training_drops=len(train), held_drops=len(test))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + '\n')
    _write(args.output, dict(schema='tt24-planar-value-corpus-1', horizon=args.horizon, train=train, held=test,
                             qualification=held))
    print(held)


if __name__ == '__main__':
    main()
