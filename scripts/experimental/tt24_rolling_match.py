"""Carry the same native Pong motor memory into actual two-racket table tennis."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import course
from .tt24_donor_trial import ROOT
from .tt24_forward_wiring import BallFree
from .tt24_motor_settle import inspect
from .tt24_rolling_motor import RollingMotorLearner
from .tt24_settled_match import run_map, totals


def trial(item):
    i, inspect_drive = item
    archive = _read(ROOT/f'learned-tip-transfer-{i}.json.gz')
    checkpoint = ROOT/f'rolling-motor-{i}.npz'
    scenes = course({'a': 417560223, 'b': 1082740361})
    cases = scenes[i*8:i*8+8]
    if len(cases) != 8 or len(set(c['seed'] for c in cases)) != 8:
        raise ValueError('incorrect held match roster')
    settings = dict(index=i, truth=archive['wiring'], fit=RollingMotorLearner.load(checkpoint).map.tolist(),
                    held=cases, source=checkpoint.name, source_sha256=_hash(checkpoint))
    tag = 'rolling-match'
    if inspect_drive:
        tag += '-inspected'
        setup = BallFree('stock', 100)
        try:
            wire = np.array(settings['truth'])
            fitted, detail = inspect(setup, 514811+i, wire, [], 520991+i, 'standard')
            confirm, repeated = inspect(setup, 514811+i, wire, [], 520991+i, 'standard')
            if not np.array_equal(fitted, confirm) or detail != repeated:
                raise ValueError('setup measurement changed on replay')
            settings.update(fit=fitted.tolist(), inspection=detail)
        finally:
            setup.close()
    return run_map(settings, ROOT/f'{tag}-{i}.json.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    tag = 'rolling-match-inspected' if args.inspect else 'rolling-match'
    with ProcessPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(trial, [(i,args.inspect) for i in (0,1)]))
    runs = [_read(p) for p in paths]
    result = dict(scores=totals(runs), verified=sum(r['verified'] for r in runs),
                  cases=[dict(name=p.name, sha256=_hash(p)) for p in paths])
    _write(ROOT/f'{tag}-summary.json.gz', result)
    print(result['scores'], flush=True)
