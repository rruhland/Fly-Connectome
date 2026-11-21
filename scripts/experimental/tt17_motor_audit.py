"""Evaluate frozen calibration on independent physical motion before table learning."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_memory import load
from .tt10_target import TargetArena
from .tt14_gain_hardware import GainWiring
from .tt14_motor_audit import displacement, move
from .tt17_motor_fit import initialise
from .tt17_motor_protocol import CASES, setup_path


def audit(spec, sample_count):
    path = ROOT / spec['donor']
    baseline = _read(path)
    source = load(path)[0].models[baseline['port_memory']['indices'][0]]
    signed, fitted, _, info = initialise(source, spec['actions'], spec['reference'], spec['active'])
    identity = np.eye(source.action_size)
    actual = np.asarray(spec['matrix'])
    models = dict(original=(source, identity), signed=(signed.base, signed.transform),
                  bounded=(fitted.base, fitted.transform), known=(source, actual))
    arena = TargetArena(**dict(spec['arena'], render_mode='projected'))
    native = GainWiring(arena, spec['wiring'], identity)
    changed = GainWiring(arena, spec['wiring'], actual)
    rows = []
    try:
        rng = np.random.default_rng([spec['game_seed'], 4850221])
        for seed in rng.integers(0, 2**31 - 1, sample_count).tolist():
            commands = np.zeros((56, source.action_size))
            rates = np.random.default_rng(seed).uniform(-.42, .42, (4, source.action_size))
            for frame, rate in zip((0, 14, 28, 42), rates, strict=True):
                commands[frame:frame+3] = rate
            before, after = move(native, seed, commands), move(changed, seed, commands)
            if not (np.array_equal(before, move(native, seed, commands)) and
                    np.array_equal(after, move(changed, seed, commands))):
                raise ValueError('independent physical motion changed on replay')
            predicted_before = displacement(*models['original'], commands)
            mistakes = {}
            for name, model in models.items():
                predicted = displacement(*model, commands)
                absolute = np.linalg.norm(predicted-after, axis=1)
                change = np.linalg.norm((predicted-predicted_before)-(after-before), axis=1)
                mistakes[name] = dict(absolute=float(absolute.mean()), change=float(change.mean()))
            rows.append(dict(seed=seed, commands=commands.tolist(), original=before.tolist(), current=after.tolist(),
                             error_m=mistakes))
    finally:
        arena.close()
    means = {model: {score: float(np.mean([r['error_m'][model][score] for r in rows]))
                     for score in ('absolute', 'change')} for model in models}
    return dict(rows=rows, mean_error_m=means, calibration=info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('the evaluation needs at least one physical motion')
    setup = setup_path(args.folder, args.case)
    result = audit(_read(setup), args.count)
    result.update(setup_sha256=_hash(setup), source_sha256=_hash(Path(__file__)))
    _write(args.folder / f'motor-{args.case}-motion-audit.json.gz', result)
    print(args.case, result['mean_error_m'], flush=True)
