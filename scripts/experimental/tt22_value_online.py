"""Carry earned physical trajectory rewards from one point into the next."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_imaging_trial import verify_live
from .tt22_rally_trial import CONFIG, OUTPUT
from .tt22_value_calibration import CALIBRATION, kinetics, stroke_outcomes
from .tt22_value_support import SupportedValue
from .tt22_value_trial import (CALIBRATED_RECEIPT, REPLICATION_RECEIPT, SOURCE, SUPPORT_RECEIPT,
                               Station, report, shared_first_choice)


RECEIPT = OUTPUT/'value-online.json.gz'
SEED = 1559276515
WEIGHT = 4.
MODES = ('online','frozen')


def without_rival_timing(row):
    rival = [{name:value for name,value in decision.items() if name!='wall_seconds'}
             for decision in row['rival']]
    return dict(row,rival=rival)


def observe(model,row):
    outcome = stroke_outcomes(row)
    examples = []
    for shot in row['shots']:
        actor = np.array(shot['actor_origin'])
        local = kinetics(np.array([shot['path']]),actor)[0]
        observation = np.r_[shot['features'],local]
        tick = shot['tick']
        label = outcome[tick]
        model.observe(observation,label['reward'],WEIGHT)
        examples.append(dict(seed=row['seed'],tick=tick,features=observation.tolist(),**label))
    return examples


def run(count,output):
    started = perf_counter()
    checkpoint = _read(CALIBRATION)
    gains = _read(SOURCE)['candidates']['duration'][1:]
    starts = np.random.default_rng(SEED).integers(2**31,size=count).tolist()
    prior_starts = set(checkpoint['archived_seeds'])
    for artifact in (CALIBRATED_RECEIPT,SUPPORT_RECEIPT,REPLICATION_RECEIPT):
        prior_starts.update(_read(artifact)['starts'])
    if len(set(starts))!=count or prior_starts.intersection(starts):
        raise ValueError('an online launch overlaps previous evidence')
    frozen = SupportedValue(checkpoint,'kernel')
    adaptive = SupportedValue.from_state(frozen.state())
    models = dict(online=adaptive,frozen=frozen)
    data = dict(schema='tt22-value-online-1',arena=CONFIG,source=CALIBRATION.name,
                source_sha256=_hash(CALIBRATION),spread=.30,pace=.82,rival='anchor',starts=starts,
                portable_gains=gains,weight_per_appended_shot=WEIGHT,donor_shots=len(frozen.labels),pairs=[],updates=[])
    station = Station(models,gains,data['spread'],data['pace'],checkpoint['horizon'],True)
    try:
        for index,seed in enumerate(starts):
            order = MODES if index%2==0 else MODES[::-1]
            pair = {mode:station.point(seed,mode) for mode in order}
            shared_first_choice(pair)
            if not index and without_rival_timing(pair['online'])!=without_rival_timing(pair['frozen']):
                raise ValueError('adaptive and frozen behavior must match before its first observation')
            examples_before = len(adaptive.labels)
            before = perf_counter()
            appended = observe(adaptive,pair['online'])
            update_seconds = perf_counter()-before
            data['pairs'].append(pair)
            data['updates'].append(dict(seed=seed,prior_examples=examples_before,added=appended,
                                        update_seconds=update_seconds,end_examples=len(adaptive.labels),
                                        new_native_frames=len(pair['online']['actions'])))
            data['summary'] = {mode:report([item[mode] for item in data['pairs']]) for mode in MODES}
            data['added_shots'] = len(adaptive.labels)-len(frozen.labels)
            data['total_update_seconds'] = sum(update['update_seconds'] for update in data['updates'])
            data['checkpoint'] = adaptive.state()
            _write(output,data)
            print('online value',seed,{mode:(row['longest'],row['outcome']['legal_returns'])
                                     for mode,row in pair.items()},'added',len(appended),flush=True)
    finally:
        station.close()
    data['verified'] = verify_live(data)
    data['elapsed_seconds'] = perf_counter()-started
    _write(output,data)
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--output',type=Path,default=RECEIPT)
    args = parser.parse_args()
    if args.count<1:
        parser.error('positive paired count required')
    run(args.count,args.output)
