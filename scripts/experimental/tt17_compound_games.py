"""Score three motor estimates on identical compound wiring and table games."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt16_transfer_bridge import costs
from .tt17_compound import path_for, recoded
from .tt17_motor_fit import initialise
from .tt17_motor_protocol import load_optical_player, make_arena, setup_path
from .tt17_motor_trial import play, prefix, verify_physics


def equivalent(left, right):
    return (left.coupled == right.coupled and np.array_equal(left.transform,right.transform)
            and all(np.array_equal(getattr(left.base,name), getattr(right.base,name))
                    for name in ('coefficients','kernels','lags')))


def qualify(folder, case):
    setup = setup_path(folder,case)
    spec = _read(setup)
    query = _read(path_for(folder,case))
    arena = make_arena(spec)
    try:
        _,_,_,archived = load_optical_player(spec,arena)
        source = archived.source
    finally:
        arena.close()
    signed, bounded, _, _ = initialise(source,spec['actions'],spec['reference'],spec['active'])
    aligned, evidence = recoded(source,query)
    previous = []
    results = {}
    for label, motor in [('signed', signed), ('bounded',bounded), ('recoded',aligned)]:
        movements = (spec['actions'] if label != 'recoded' else query['actions']+query['second_actions'])
        preparation = dict(optics=costs([a for p in spec['optics_probes'] for a in p['actions']]),
                           active=costs(movements))
        path = next((receipt for earlier, receipt in previous if equivalent(motor,earlier)), None)
        if path is not None:
            game = _read(path)
            verify_physics(game,spec)
        else:
            game = play(folder,case,'bounded_frozen',initial_motor=motor,preparation_override=preparation,label=label)
            path = prefix(folder,case,label).with_suffix('.json.gz')
            previous.append((motor,path))
        row=game['games'][0]
        results[label]=dict(game=path.name, digest=_hash(path), checkpoint=game['checkpoint'],
                            active_movement=preparation['active'], score=row['score'],
                            returns=row['metrics']['total_legal_returns'], multiple=row['metrics']['two_return_points'])
    _write(folder/f'compound-{case}-games.json.gz',dict(setup_sha256=_hash(setup),
           query_sha256=_hash(path_for(folder,case)),recoded=evidence,results=results))
    print(case,{k:(v['score'],v['returns']) for k,v in results.items()},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('case',choices=('gantry-0','gantry-1','five-0','five-1'))
    a=parser.parse_args()
    qualify(a.folder,a.case)
