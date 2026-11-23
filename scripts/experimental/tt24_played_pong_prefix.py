"""Carry motor knowledge acquired in the first seconds of reward-trained native Pong play."""
import numpy as np
import torch

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM, arcade
from .tt24_motor_bank import I, STEP
from .tt24_pong_arm import DIRECTION, NATIVE_CAP, PhysicalPaddle, bounce, wirings


class NativeJoystick(PhysicalPaddle):
    def step(self, drive):
        # Positive native motor drive points towards the upper edge of the screen.
        super().step(-drive)


def point(arena, wiring, seed, weights):
    joystick = None
    def attach(config):
        nonlocal joystick
        joystick = NativeJoystick(arena, wiring, config)
        return joystick
    score, hits, frames = arcade(weights, [seed], cap=NATIVE_CAP, opponent=.35, paddle_factory=attach)[0, 0]
    return dict(score=score, hits=hits, frames=frames, commands=joystick.commands, encoders=joystick.encoders)


def from_prefix(record, length, *, shuffled=False):
    u = np.array(record['commands'][:length]) @ DIRECTION
    y = np.array(record['encoders'][1:len(u) + 1])
    if shuffled:
        y = np.roll(y, len(y) // 2, axis=0)
    goal = STEP * np.cumsum(u)
    best = None
    for alpha in (.08, .13, .21, .34, .52, .72, 1.):
        latest = 0.
        filtered = []
        for sample in goal:
            latest += alpha * (sample - latest)
            filtered.append(latest)
        x = np.array(filtered)
        def fit(xs, ys):
            centered = xs - xs.mean()
            ridge = .0001
            column = (centered @ ys + ridge * DIRECTION) / (centered @ centered + ridge)
            return column, ys.mean(axis=0) - xs.mean() * column
        first = max(2, 2 * len(x) // 3)
        column, bias = fit(x[:first], y[:first])
        error = np.mean((y[first:] - x[first:, None] * column - bias) ** 2)
        if best is None or error < best[0]:
            best = (error, alpha, fit(x, y)[0])
    matrix = I.copy()
    matrix[:, 0] = best[2]
    return matrix, dict(frames=len(u), validation_rmse=float(np.sqrt(best[0])),
                        alpha=best[1], map=matrix.tolist())


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(1288130093)
    source_seeds = rng.integers(1, 2**31-1, 8).tolist()
    target_seeds = rng.integers(1, 2**31-1, 12).tolist()
    wiring = wirings()[0]
    archive = _read(WARM)
    if not archive['gate']:
        raise ValueError('source readout lacks native qualification')
    weights = archive['weights']
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    try:
        source = [point(arena, wiring, seed, weights) for seed in source_seeds]
        stationary = [point(arena, wiring, seed, np.zeros_like(weights)) for seed in source_seeds]
        for row, seed in zip(source, source_seeds, strict=True):
            if digest_json(point(arena, wiring, seed, weights)) != digest_json(row):
                raise ValueError('source replay changed')
        report = dict(source_seeds=source_seeds, target_seeds=target_seeds, source=source,
                      stationary=stationary, native_checkpoint_sha256=_hash(WARM), wrist_sha256=_hash(BUNDLE))
        target = ROOT / 'played-pong-prefix.json.gz'
        _write(target, report)
        counts = [sum(row['hits'] for row in group) for group in (source, stationary)]
        print('source', counts, flush=True)
        if counts[0] <= counts[1] or len(source[0]['commands']) < 120:
            return
        modes = {'stale': I}
        fits = {}
        for label, ticks, shuffle in [('half_second', 30, False), ('two_seconds', 120, False), ('shuffled_two', 120, True)]:
            modes[label], fits[label] = from_prefix(source[0], ticks, shuffled=shuffle)
        policy = CarriedWristPolicy()
        rows = []
        for seed in target_seeds:
            group = {}
            for label, matrix in modes.items():
                played = bounce(arena, policy, seed, wiring, matrix)
                repeated = bounce(arena, policy, seed, wiring, matrix, tape=played['actions'])
                if digest_json(played) != digest_json(repeated):
                    raise ValueError('wrist replay changed')
                group[label] = played
            rows.append(dict(seed=seed, modes=group))
            report.update(fits=fits, rows=rows)
            _write(target, report)
        for name in modes:
            scores = [r['modes'][name]['metrics']['longest_consecutive_rebounds'] for r in rows]
            print(name, scores, flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
