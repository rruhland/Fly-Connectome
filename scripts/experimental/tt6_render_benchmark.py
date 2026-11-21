"""Compare cached and native hall observations on paired physical rollouts."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from .tt4_transfer_trial import ROOT, _point
from .tt5_checkpoint import load, setup
from .tt6_body_arena import BODIES, TT6Arena
from .tt6_render import cache_hall_stereo


_MODES = ('native', 'cached', 'reflection', 'compiled')
_POINT_SEEDS = (833294100, 1175555228)


def _install(arena, mode):
    if mode != 'native':
        cache_hall_stereo(arena, reflections=mode in ('compiled', 'reflection'), compiled_geometry=mode == 'compiled')


def _capture(body, seed, distractor, mode, steps):
    arena = TT6Arena(seed, body=body, width=192, height=144, venue='hall',
                     baseline=.28, distractor=distractor)
    try:
        _install(arena, mode)
        frames = [arena.reset(seed=seed)]
        rng = np.random.default_rng(seed)
        cpu, wall = process_time(), perf_counter()
        for _ in range(steps):
            command = rng.uniform(-1., 1., arena.action_size)
            frame, _, terminated, truncated, _ = arena.step(command)
            frames.append(frame)
            if terminated or truncated:
                break
        steps_taken = len(frames) - 1
        return (np.stack(frames), arena.data.qpos.copy(),
                1000 * (process_time() - cpu) / steps_taken,
                1000 * (perf_counter() - wall) / steps_taken)
    finally:
        arena.close()


def _images(steps, bodies):
    for body in bodies:
        i = BODIES.index(body)
        seed = 69237 + i
        distractor = i % 2 == 1
        rotation = i % 2
        results = {mode: _capture(body, seed, distractor, mode, steps)
                   for mode in _MODES[rotation:] + _MODES[:rotation]}
        native, physical, _, _ = results['native']
        moving = np.any(native[1:] != native[:-1], axis=-1)
        for mode, (frames, qpos, cpu, wall) in results.items():
            np.testing.assert_array_equal(physical, qpos)
            difference = np.abs(native.astype(np.int16) - frames)
            changed = np.any(difference, axis=-1)
            yield dict(body=body, seed=seed, mode=mode, frames=len(frames), cpu_ms=cpu, wall_ms=wall,
                       rgb_mean_error=float(difference.mean()), rgb_max_error=int(difference.max()),
                       changed_pixels=int(changed.sum()), pixels=int(changed.size),
                       changed_moving_pixels=int(changed[1:][moving].sum()), moving_pixels=int(moving.sum()))


def _scored():
    root = ROOT / 'artifacts' / 'tt5'
    for mode in ('reflection', 'native'):
        _, _, wiring = setup(root / 'short-calibration.json.gz', 0)
        try:
            _install(wiring.arena, mode)
            for seed in _POINT_SEEDS:
                player = load(root / 'first-hall-player.npz')
                cpu, wall = process_time(), perf_counter()
                row = _point(wiring, player, seed)
                yield dict(mode=mode, seed=seed, action_sha256=row['action_sha256'],
                           evaluator={key: value for key, value in row['evaluator'].items() if key != 'events'},
                           cpu_seconds=process_time() - cpu, wall_seconds=perf_counter() - wall)
        finally:
            wiring.arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--body', choices=BODIES, action='append')
    parser.add_argument('--images-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    evidence = dict(images=list(_images(args.steps, args.body or BODIES)),
                    points=[] if args.images_only else list(_scored()))
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
