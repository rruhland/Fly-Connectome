"""Test a motor decoder acquired in a different game from unassisted receiving starts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read, _write
from .tt23_sequence import course, verify
from .tt24_rally_bank import CACHE, DIMS
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_remap import driver
from .tt24_rally_trial import COACHED, Station, returns

DIR = Path('artifacts/tt24')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    args = p.parse_args()
    bridge = json.loads((DIR/'carried-physics.json').read_text())
    wire, learned = (np.asarray(bridge[key]) for key in ('wire','decoder'))
    eye = np.eye(5)
    settings = [('native',eye,eye),('frozen',wire,eye),('bounce',wire,learned),('known',wire,np.linalg.inv(wire))]
    starts = course(dict(first=13051117,second=9341991))
    station = Station(_read(COACHED)['metadata']['opening']['far_profile'],render_dims=DIMS)
    with np.load(CACHE) as data:
        if set(data['seed']) & {s['seed'] for s in starts}:
            raise ValueError('reused teacher launch')
        pupil = policy_for('retina_rate',data)
    cases = []
    try:
        for case in starts:
            results = {}
            for name,physical,decoded in settings:
                results[name] = station.play(case,name,pupil,live_opening=True,
                                               execute=lambda u:driver(physical,decoded,u))
            if len({r['prefix_rgb'] for r in results.values()}) != 1:
                raise ValueError('different initial cameras')
            cases.append(dict(**case,results=results))
            print(case['seed'],{k:len(returns(r,1)) for k,r in results.items()},flush=True)
        _write(args.out,dict(cases=cases,verified=verify(cases),target_experience=bridge['target_cost'],
                             source='carried-physics.json',table_tennis_exposure_after_adaptation=0))
    finally:
        station.arena.close()


if __name__=='__main__':
    main()
