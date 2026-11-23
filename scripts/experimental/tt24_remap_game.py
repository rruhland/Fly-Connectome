"""Evaluate direct policies after empirical five-motor reidentification on real drops."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .tt24_effects import excitation, transmission
from .tt24_bounce import BounceArena
from .tt24_effects import MotorEffects
from .tt24_motor import MotorEffects as PolicyGains, MotorPolicy


def measure(arena, wiring, seed):
    """Collect paired command/encoder changes during real, charged practice drops."""
    schedule = excitation(np.random.default_rng(seed))
    model = MotorEffects(5)
    ticks = round(.1 / arena.dt)
    frames = drops = 0
    done = True
    for probe in schedule:
        while True:
            if done:
                obs = arena.reset(seed=seed + drops, split='practice', render=False)
                drops += 1
                prev = np.zeros(5)
            before = obs['joints'].copy()
            for tick in range(ticks * 2):
                act = probe if tick < ticks else np.zeros(5)
                obs, _, done = arena.step(wiring @ act, render=False)
                frames += 1
                if done:
                    break
            if tick == ticks * 2 - 1:
                delta = obs['joints'] - before
                model.update(probe, delta, prev)
                prev = delta
                break
    return model.coefficients()[:5].T, dict(frames=frames, drops=drops)


def run_policy(arena, seed, record, decoder, wiring, *, regime='standard', hold=False, blind=False):
    gains = PolicyGains(5)
    gains.readout[:] = record['motor_effects']
    policy = MotorPolicy(np.asarray(record['weights']), gains, readout=record.get('readout', 'position'),
                         retinal=record.get('retinal', 'absolute'))
    obs = arena.reset(seed=seed, split='held', regime=regime)
    reward = frames = 0
    done = False
    while not done:
        sampled = frames % record.get('stride', 1) == 0
        action = np.zeros(5) if hold else policy.command(obs, sampled=sampled, blind=blind)
        send = np.clip(decoder @ action, -1., 1.)
        obs, earned, done = arena.step(wiring @ send, render=(frames + 1) % record.get('stride', 1) == 0)
        reward += earned
        frames += 1
    judge = arena.judge_metrics()
    return dict(rebounds=int(judge['rebounds']), consecutive=judge['longest_consecutive_rebounds'], score=reward, frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bodies', type=int, default=4)
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--seed', type=int, default=777111)
    args = parser.parse_args()
    record = json.loads(gzip.decompress(args.checkpoint.read_bytes()))
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(1000000, 2**30, args.starts).tolist()
    arena = BounceArena(view='wrist', width=record.get('width', 96), height=record.get('height', 72))
    rows = []
    try:
        nominal, base_cost = measure(arena, np.eye(5), args.seed)
        native = [run_policy(arena, start, record, np.eye(5), np.eye(5)) for start in starts]
        for body in range(args.bodies):
            wiring = transmission(rng)
            observed, cost = measure(arena, wiring, args.seed + 100 + body)
            decoder = np.linalg.lstsq(observed, nominal, rcond=.01)[0]
            outcomes = {label: [run_policy(arena, start, record, candidate, wiring) for start in starts]
                        for label, candidate in [('frozen', np.eye(5)), ('identified', decoder),
                                                 ('oracle', np.linalg.inv(wiring))]}
            rows.append(dict(body=body, target_cost=cost, outcomes=outcomes, decoder=decoder.tolist(),
                             wiring=wiring.tolist(), effect=observed.tolist()))
            print(body, cost, {k: sum(x['rebounds'] for x in v) for k, v in outcomes.items()}, flush=True)
        out = dict(starts=starts, native=native, source_cost=base_cost, rows=rows,
                   source_checkpoint=args.checkpoint.name,
                   source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   nominal_effect=nominal.tolist())
        args.output.write_text(json.dumps(out, separators=(',', ':')) + '\n')
    finally:
        arena.close()


if __name__ == '__main__':
    main()
