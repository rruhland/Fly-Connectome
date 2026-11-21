"""Learn a catch prior from previous table-to-cart handoffs, then test on another arm."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt7_surface_trial import DAMPING
from .tt8_transfer_continuity import SOURCE_NOISE, catcher, generic_controls, snapshots, summarize
from .tt8_transfer_model import RESTITUTION
from .tt8_transfer_rebound import DT, ReboundCourt, steer
from .tt8_transfer_trial import EARLY_FRAMES, play, verify


def target_example(court, row, observer):
    """Replay the recorded cart commands. Only the eight visible images and final image inform the fit."""
    opening = court.opening(row['fall'])
    cart = opening.cart_start
    observer.reset()
    commands = row['players']['generic_mixture']['commands']
    if len(commands) != len(opening.positions) - 1:
        raise ValueError('archive duration changed')
    for i, (ball, command) in enumerate(zip(opening.positions[:-1], commands, strict=True)):
        if i < EARLY_FRAMES:
            observer.act(court.frame(ball, cart))
        elapsed = opening.crossing_time - i * DT if i == len(commands) - 1 else DT
        cart = steer(cart, command, elapsed)
    observer.finish(court.frame(opening.positions[-1], cart))


def learn(archive, courts):
    """Fit a target distribution independently of the source for each four-fall session."""
    source_beliefs = {h['count']: h['priors']['chronological'] for h in archive['handoffs']}
    examples = []
    uniform = np.ones_like(RESTITUTION) / len(RESTITUTION)
    for row in archive['rows']:
        if row['round'] == 0:
            observer = catcher(uniform)
            examples.append(dict(source=source_beliefs[row['handoff']], target=None))
        target_example(courts[row['material']], row, observer)
        examples[-1]['target'] = observer.probability.tolist()
    return examples


def retrieve(examples, source, generic):
    # Shrink sparse local memories toward two generic sessions. The marginal
    # comparison uses exactly the same target observations and regularization.
    centers = np.array([e['source'] for e in examples]) @ RESTITUTION
    targets = np.array([e['target'] for e in examples])
    weights = np.exp(-.5 * ((centers - source @ RESTITUTION) / SOURCE_NOISE)**2)
    local = (weights @ targets + 2. * generic) / (weights.sum() + 2.)
    marginal = (targets.sum(axis=0) + 2. * generic) / (len(examples) + 2.)
    return local, marginal


def run(archive_path, source_path, visit, seed, checkpoint):
    archive = _read(archive_path)
    if archive['visit'] == visit or archive['source_sha256'] != _hash(source_path):
        raise ValueError('use different visits from the same source archive')
    courts = {name: ReboundCourt(name) for name in DAMPING}
    examples = learn(archive, courts)
    _write(checkpoint, dict(schema='tt9-source-target-memory-v1', source_sha256=_hash(archive_path),
                            examples=examples))
    examples = _read(checkpoint)['examples']
    rng = np.random.default_rng(seed)
    # Reassign observed sources across training sessions for the pairing control.
    order = np.random.default_rng(9156902).permutation(len(examples))
    shuffled = [dict(source=examples[j]['source'], target=e['target']) for e, j in zip(examples, order, strict=True)]
    points = _read(source_path)['visits'][visit]['points']
    measurements = [[s['ratio'] for s in p['table_bounce']['samples']] for p in points]
    generic = generic_controls()['generic_mixture']
    rows = []
    for count, beliefs, _ in snapshots(measurements, np.random.default_rng(810507)):
        current = beliefs['chronological']
        local, marginal = retrieve(examples, current, generic)
        scrambled, _ = retrieve(shuffled, current, generic)
        priors = dict(paired=local, pooled=marginal, shuffled=scrambled, direct=current,
                      half_direct=(current + generic) / 2., generic=generic)
        material = points[count - 1]['phase']
        for session, change in enumerate(rng.permutation([False, True] * 4)):
            target = str(rng.choice([m for m in courts if m != material])) if change else material
            agents = {k: catcher(v) for k, v in priors.items()}
            for i, fall in enumerate(rng.integers(0, 2**31 - 1, size=4).tolist()):
                opening = courts[target].opening(fall)
                actions = {k: play(courts[target], opening, a, visibility='curtain') for k, a in agents.items()}
                rows.append(dict(handoff=count, session=session, round=i, fall=fall, material=target,
                                 visibility='curtain', continuity='changed' if change else 'persisted',
                                 crossing_m=opening.endpoint, time_s=opening.crossing_time, players=actions))
    return dict(schema='tt9-conditional-catch-v1', seed=seed, source_visit=visit, training_visit=archive['visit'],
                examples=len(examples), checkpoint_sha256=_hash(checkpoint), table_archive_sha256=_hash(source_path),
                implementation_sha256=_hash(Path(__file__)), rows=rows, summary=summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt7/surface-read-only.json.gz')
    parser.add_argument('--visit', type=int, choices=(0, 1), required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    receipt = run(args.train, args.source, args.visit, args.seed, args.checkpoint)
    _write(args.output, receipt)
    print({k: v['catches'] for k, v in receipt['summary']['all']['scores'].items()})
    print(f"Verified {verify(_read(args.output))} physical cart tapes.")


if __name__ == '__main__':
    main()
