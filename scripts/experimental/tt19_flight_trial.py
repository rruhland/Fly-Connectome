"""Estimate and compare flight laws between disjoint table-tennis bodies."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw, HORIZONS, errors, fit_flight, windows


SOURCE = ROOT/'artifacts/tt18/combined-418336241-carried.json.gz'
CONTROL = ROOT/'artifacts/tt18/combined-349051097-carried.json.gz'


def traces(path):
    record = _read(path)
    games = record.get('games', [record])
    return [[frame[0] for frame in r['optical']] for game in games for r in game['points']]


def evaluate(law, data, bounds=(.7625, 1.37)):
    distance = np.linalg.norm(errors(law, data, bounds), axis=-1)
    return dict(samples=len(distance), median_m=np.median(distance, axis=0).tolist(),
                mean_m=distance.mean(axis=0).tolist(), p90_m=np.quantile(distance, .9, axis=0).tolist())


def source_fit(folder):
    data = windows(traces(SOURCE))
    law = fit_flight(data)
    control = windows(traces(CONTROL))
    report = dict(source_sha256=_hash(SOURCE), held_sha256=_hash(CONTROL), source_windows=len(data[0]),
                  horizon_frames=HORIZONS, source_law=law.state(), held_gantry={})
    for name, model in [('gantry', law), ('nominal', FlightLaw())]:
        report['held_gantry'][name] = evaluate(model, control)
    _write(folder/'flight-source.json.gz', report)
    return report


def transfer(folder):
    source = folder/'online-29413852-warm-wrist+fixed-setup24.json.gz'
    held = folder/'online-886230749-warm-wrist+fixed-setup24-strict.json.gz'
    def arm(path):
        return windows([[frame['ball'] for frame in row['optical_trace']]
                        for game in _read(path)['games'] for row in game['points']])
    bounds=(.424,.80)
    source_arm, held_arm=arm(source),arm(held)
    held_gantry=windows(traces(CONTROL))
    models={'gantry':FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law']),
            'arm':fit_flight(source_arm,bounds), 'nominal':FlightLaw()}
    result=dict(arm_source=source.name,arm_source_sha256=_hash(source),arm_source_windows=len(source_arm[0]),
                arm_test=held.name,arm_test_sha256=_hash(held),gantry_test_sha256=_hash(CONTROL),
                horizons=HORIZONS,models={k:v.state() for k,v in models.items()})
    for label,data,court in [('arm',held_arm,bounds),('gantry',held_gantry,(.7625,1.37))]:
        result[label]={k:evaluate(law,data,court) for k,law in models.items()}
        position,speed,correction,target=data
        straight=position[:,None,:]+np.array(HORIZONS)[None,:,None]*speed[:,None,:]
        result[label]['constant_velocity_median_m']=np.median(np.linalg.norm(straight-target,axis=-1),axis=0).tolist()
    _write(folder/'flight-transfer.json.gz',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--cross-body',action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    print(transfer(args.folder) if args.cross_body else source_fit(args.folder))
