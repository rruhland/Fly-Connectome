"""Remember stroke speeds that earned public points across physical installations."""
import argparse
from pathlib import Path
from statistics import median

from .tt4_transfer_trial import _hash, _read, _write


class EntryMemory:
    def __init__(self, station_speeds=()):
        self.station_speeds = list(station_speeds)
        self.won_speeds = []

    def observe(self, point):
        if point['opportunity'] and point['public_reward'] > 0:
            self.won_speeds.append(float(point['impact']['used_speed']))

    def finish_visit(self):
        if self.won_speeds:
            self.station_speeds.append(median(self.won_speeds))
        self.won_speeds.clear()

    def enter(self, racket):
        if self.station_speeds:
            racket.speed = median(self.station_speeds)
        return racket.speed

    @classmethod
    def from_state(cls, state):
        return cls(state['station_speeds'])

    def state(self):
        return dict(station_speeds=self.station_speeds.copy())


def learn(courses, output):
    memory = EntryMemory()
    previous = None
    for path in courses:
        for visit in _read(path)['visits']:
            if previous is not None and previous != visit['before']:
                raise ValueError('training visits do not form one carried course')
            previous = visit['after']
            for point in visit['points']:
                memory.observe(point)
            memory.finish_visit()
    _write(output, dict(**memory.state(), inputs_sha256=[_hash(path) for path in courses]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('courses', type=Path, nargs='+')
    args = p.parse_args()
    learn(args.courses, args.output)
