"""Audit the frozen game maps on fresh physical motions with matched transport."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt10_target import TargetArena
from .tt14_gain_hardware import GainWiring
from .tt14_gain_trial import load_map
from .tt14_motor_audit import displacement, move


def evaluate(folder, site, profile, seed, count):
    prefix = f'motor-{site}-{profile}'
    models, checksums = {}, {}
    data = None
    for mode in ('fixed', 'hybrid', 'guard', 'probe'):
        name = f'{prefix}-{mode}'
        path = folder / f'{name}.json.gz'
        if mode == 'probe' and not path.is_file():
            continue
        receipt = _read(path)
        checkpoint = folder / f'{name}-changed-player.json.gz'
        filesum = _hash(checkpoint)
        if filesum != receipt['games'][0]['checkpoint']:
            raise ValueError(f'changed game checkpoint diverged for {name}')
        if data is not None and (data['arena'], data['matrix']) != (receipt['arena'], receipt['matrix']):
            raise ValueError('the learned maps came from different hardware')
        data = receipt
        models[mode] = load_map(checkpoint)
        checksums[mode] = filesum
    settings = data['arena']
    arena = TargetArena(**dict(settings, render_mode='projected'))
    matrix = np.asarray(data['matrix'])
    models['known'] = models['fixed'][0], matrix
    original = GainWiring(arena, settings['seed'])
    changed = GainWiring(arena, settings['seed'], matrix)
    rows = []
    try:
        for motion_seed in np.random.default_rng(seed).integers(0, 2**31-1, count).tolist():
            actions = np.zeros((56, arena.action_size))
            commands = np.random.default_rng(motion_seed).uniform(-.42, .42, (4, arena.action_size))
            for tick, command in zip((0, 14, 28, 42), commands, strict=True):
                actions[tick:tick+3] = command
            previous = move(original, motion_seed, actions)
            current = move(changed, motion_seed, actions)
            if not np.array_equal(previous, move(original, motion_seed, actions)) or not np.array_equal(
                    current, move(changed, motion_seed, actions)):
                raise ValueError('physical pulse replay diverged')
            reference = displacement(*models['fixed'], actions)
            losses = {}
            for mode, model in models.items():
                prediction = displacement(*model, actions)
                losses[mode] = dict(absolute=float(np.linalg.norm(prediction - current, axis=1).mean()),
                                    change=float(np.linalg.norm(prediction - reference - current + previous, axis=1).mean()))
            rows.append(dict(seed=motion_seed, actions=actions.tolist(), original=previous.tolist(),
                             changed=current.tolist(), losses=losses))
    finally:
        arena.close()
    means = {mode: {stat: float(np.mean([row['losses'][mode][stat] for row in rows]))
                    for stat in ('absolute', 'change')} for mode in models}
    return dict(seed=seed, arena=settings, matrix=matrix.tolist(), checkpoint_sha256=checksums, means_m=means,
                rows=rows, implementation_sha256=_hash(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('site', choices=('gantry', 'five'))
    parser.add_argument('profile', choices=('gain', 'coupling'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=16)
    args = parser.parse_args()
    result = evaluate(args.folder, args.site, args.profile, args.seed, args.count)
    _write(args.folder / f'motor-{args.site}-{args.profile}-pulses.json.gz', result)
    print(result['means_m'])
