"""Compare causal flight estimates for the same bounded physical value ranker."""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_flight_inverse import FlightInverse
from .tt22_imaging_trial import verify_live
from .tt22_rally_trial import CONFIG, OUTPUT
from .tt22_value_calibration import CALIBRATION
from .tt22_value_online import SEED as ONLINE_SEED
from .tt22_value_player import ValuePlayer
from .tt22_value_support import SupportedValue
from .tt22_value_trial import (
    CALIBRATED_SEED, REPLICATION_SEED, SEED as SOURCE_SEED, SOURCE, SUPPORT_SEED, Station, report,
)


SEED = 211469891
RECEIPT = OUTPUT/'value-inverse.json.gz'
MODES = ('inverse', 'stock')


class InverseValuePlayer(ValuePlayer):
    """Fit a recent incoming arc before proposing and forecasting each stroke."""

    def __init__(self,law,model,gains,horizon):
        super().__init__(law,model,gains,horizon)
        self.infer = FlightInverse()

    def search(self,estimate,observation,target):
        fit = self.infer.fit(self.history)
        if fit is None:
            return super().search(estimate,observation,target)
        _,velocity,spin,residual = fit
        revised = replace(observation,velocity=velocity,spin=spin)
        choice = super().search((estimate[0],velocity/30.),revised,target)
        if choice is not None:
            choice[1]['flight_fit'] = dict(velocity=velocity.tolist(),spin=spin.tolist(),rms_mm=residual)
        return choice


def run(count,output):
    started = perf_counter()
    checkpoint = _read(CALIBRATION)
    gains = _read(SOURCE)['candidates']['duration'][1:]
    starts = np.random.default_rng(SEED).integers(2**31,size=count).tolist()
    archive = set(checkpoint['archived_seeds'])
    seeds = (SOURCE_SEED,CALIBRATED_SEED,SUPPORT_SEED,REPLICATION_SEED,ONLINE_SEED)
    for seed in seeds:
        archive.update(np.random.default_rng(seed).integers(2**31,size=max(16,count)))
    if len(set(starts))!=count or archive.intersection(starts):
        raise ValueError('a flight estimate launch overlaps previous evidence')
    model = SupportedValue(checkpoint,'kernel')
    data = dict(schema='tt22-value-inverse-1',arena=CONFIG,source=CALIBRATION.name,
                source_sha256=_hash(CALIBRATION),spread=.30,pace=.82,rival='anchor',starts=starts,
                portable_gains=gains,pairs=[])
    args = (dict.fromkeys(MODES,model),gains,data['spread'],data['pace'],checkpoint['horizon'],True)
    stock = Station(*args)
    try:
        inverse = Station(*args,player_type=InverseValuePlayer)
        stations = dict(inverse=inverse,stock=stock)
        try:
            for index,seed in enumerate(starts):
                order = MODES if index%2==0 else MODES[::-1]
                pair = {mode:stations[mode].point(seed,mode) for mode in order}
                data['pairs'].append(pair)
                data['summary'] = {mode:report([rows[mode] for rows in data['pairs']]) for mode in MODES}
                _write(output,data)
                print('value with flight estimate',seed,{mode:(row['longest'],row['outcome']['legal_returns'],
                           sum('flight_fit' in shot for shot in row['shots'])) for mode,row in pair.items()},flush=True)
        finally:
            inverse.close()
    finally:
        stock.close()
    data['verified'] = verify_live(data)
    data['elapsed_seconds'] = perf_counter()-started
    _write(output,data)
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--output',type=Path,default=RECEIPT)
    args = parser.parse_args()
    if args.count<1:
        parser.error('positive paired count required')
    run(args.count,args.output)
