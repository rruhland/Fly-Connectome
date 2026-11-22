"""Measure table practice on the eight after-service launches, holding the lateral controller fixed."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt22_after_service import IntentPlayer, verify_rows
from .tt22_ordinary_game import DIR, FrozenMemory, StandardMatch
from .tt23_optical_contact import OUT


def run():
    cohort=DIR/'after-service.json.gz'
    comparison=_read(cohort)
    seeds=np.random.default_rng(comparison['seed']).integers(2**31,size=8).tolist()
    path=OUT/'after-service-zero.json.gz'
    d=dict(condition='neutral cell, zero optical examples',seeds=seeds,rows=[])
    if path.exists():
        saved=_read(path)
        if saved['seeds']!=seeds:
            raise ValueError('incompatible matched seed schedule')
        d=saved
    station=StandardMatch(comparison['feed'])
    station.player.close()
    station.player=IntentPlayer(FlightLaw())
    station.player.gains=np.zeros(3)
    memory=FrozenMemory(guard_empty=True)
    try:
        for i,seed in enumerate(seeds):
            if i<len(d['rows']):
                if d['rows'][i]['seed']!=seed:
                    raise ValueError('wrong saved point')
                continue
            row=station.play(seed,1,memory)
            d['rows'].append(row)
            _write(path,d)
            print(i,row['longest'],flush=True)
        verify_rows(d['rows'])
        d['verified']=len(d['rows'])
        _write(path,d)
    finally:
        station.close()


if __name__=='__main__':
    run()
