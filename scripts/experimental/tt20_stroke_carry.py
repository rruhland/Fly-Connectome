"""Continue the same online stroke values across a physical camera change."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_camera_carry import EYES, OUTPUT, SETTINGS, Station, practice_schedule, verify

SWITCH = 97643315


class StrokeValues:
    """Update four program values with the public reward and optically estimated placement."""

    def __init__(self, factory_pick, state=None):
        self.count = np.zeros(4, dtype=int)
        self.total = np.zeros(4)
        self.prior = factory_pick
        if state is not None:
            self.count[:] = state['count']
            self.total[:] = state['merit_sum']

    def record(self, candidate, merit):
        self.count[candidate] += 1
        self.total[candidate] += merit

    def pick(self):
        if min(self.count) == 0:
            return self.prior
        return int(np.argmax(self.total/self.count + .03*(np.arange(4) == self.prior)))

    def state(self):
        return dict(count=self.count.tolist(), merit_sum=self.total.tolist(), pick=self.pick())


def restore(data, factory_pick):
    if 'state' in data:
        return StrokeValues(factory_pick, state=data['state'])
    values = StrokeValues(factory_pick)
    for row in data['points']:
        values.record(row['candidate'],row['merit'])
    if data['learning'][-1]['selected'] != values.pick():
        raise ValueError('source receipt and restored program values differ')
    return values


def adapt(destination, source_file, *, seed=SWITCH, blocks=2):
    old = _read(source_file)
    factory_pick = old['factory_pick']
    carried = restore(old,factory_pick)
    fresh = StrokeValues(factory_pick)
    station = Station(destination)
    rows, learning = [], []
    schedule = practice_schedule(seed,blocks)
    used = set(old.get('used_seeds',[r['seed'] for r in old['points']]))
    if used.intersection(s for s,c in schedule):
        raise ValueError('destination practice repeated a previous point')
    before = carried.state()
    try:
        for s, candidate in schedule:
            row = station.point(s,1,candidate)
            rows.append(row)
            for learner in (carried,fresh):
                learner.record(candidate,row['merit'])
            if len(rows)%4 == 0:
                learning.append(dict(points=len(rows),carried=carried.state(),destination_only=fresh.state()))
    finally:
        station.close()
    receipt = dict(schema='tt20-continuous-strokes-1', factory_pick=factory_pick, eye=destination,
                   arena=SETTINGS, factory_sha256=_hash(OUTPUT/'curriculum-factory.json.gz'),
                   source=source_file.name, source_sha256=_hash(source_file),
                   source_eye=old['eye'], source_values=before, state=carried.state(), seed=seed,
                   used_seeds=sorted(used | {s for s,c in schedule}), points=rows, learning=learning)
    receipt['physical_replays'] = verify(receipt)
    _write(OUTPUT/f'stroke-carry-{destination}-{seed}.json.gz',receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eye',choices=EYES)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--seed',type=int,default=SWITCH)
    parser.add_argument('--blocks',type=int,default=2)
    args=parser.parse_args()
    other=next(eye for eye in EYES if eye!=args.eye)
    previous=args.resume or OUTPUT/f'camera-carry-{other}-practice.json.gz'
    print(adapt(args.eye,previous,seed=args.seed,blocks=args.blocks)['learning'])
