"""Score native RGB SO-101 exchanges with a physical five-axis rival."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _read, _write
from .tt19_flight import FlightLaw
from .tt20_natural import NaturalViewTransport
from .tt22_precision import PrecisionArena
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, OUTPUT, GentleDraws, Station


PILOT = 701433653
HELD = 1561049159


class PrecisionRallyArena(PrecisionArena):
    def __init__(self, spread, pace):
        self.spread, self.pace = spread, pace
        super().__init__(**CONFIG)

    def _opening(self):
        if self.server == 1 or (self.spread == self.pace == 1.):
            return super()._opening()
        rng = self._rng
        self._rng = GentleDraws(rng, self.spread, self.pace)
        try:
            return super()._opening()
        finally:
            self._rng = rng


class PrecisionStation(Station):
    def __init__(self, spread, pace):
        self.arena = PrecisionRallyArena(spread, pace)
        self.rival = self.arena.precision
        law = FlightLaw()
        self.player = RallyPlayer(law)
        self.optics = NaturalViewTransport('fixed', lens_fov=self.arena.lens_intrinsics(), law=law,
                                           flight=False, associate=True)

    def point(self, seed):
        result = super().point(seed)
        result['far_actions'] = self.arena.precision_actions.copy()
        return result


def replay(rows, spread, pace, *, feeds=None):
    arena = PrecisionRallyArena(spread, pace)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=-1, render=False)
            far = np.array(row['far_actions'])
            arena._aim = lambda *args: far[arena.steps]
            if len(far) != len(row['actions']):
                raise ValueError('the recorded near and far tapes must share the control clock')
            state = None
            target_y = arena.precision_anchor[1]
            for command in row['actions']:
                first_event = len(arena.events)
                _, reward, _, _, _ = arena.step(command, render=False)
                if feeds is None:
                    continue
                for event in arena.events[first_event:]:
                    if event['kind'] == 'legal_return' and event.get('side') == -1:
                        state = dict(seed=row['seed'], legal_tick=event['step'], crossing=None, learner_contact=None)
                        feeds.append(state)
                        previous = None
                    elif state is not None and event['kind'] == 'contact' and event.get('side') == 1:
                        state['learner_contact'] = dict(tick=event['step'],
                                                        position=(np.array(event['ball'])-[0., 0., TABLE_HEIGHT]).tolist(),
                                                        velocity=event['incoming'])
                        state = None
                if state is None:
                    continue
                sample = np.r_[arena.ball-[0., 0., TABLE_HEIGHT], arena.velocity]
                if (state['crossing'] is None and previous is not None and previous[1] > target_y >= sample[1]
                        and sample[4] < 0.):
                    fraction = (target_y-previous[1]) / (sample[1]-previous[1])
                    crossing = previous + fraction*(sample-previous)
                    state['crossing'] = dict(tick=float(arena.steps-1+fraction), position=crossing[:3].tolist(),
                                             velocity=crossing[3:].tolist())
                previous = sample
            if (row['outcome'] != arena.point_result() or row['reward'] != reward
                    or json.dumps(row['events']) != json.dumps(arena.events)):
                raise ValueError(f"two-arm physical replay changed for {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def metrics(rows):
    scores = [row['longest'] for row in rows]
    return dict(points=len(rows), alternating_returns=scores, longest=max(scores), ten_or_more=sum(n >= 10 for n in scores),
                near_returns=sum(row['outcome']['legal_returns'] for row in rows),
                far_returns=sum(row['outcome']['rival_returns'] for row in rows),
                far_confirmed=sum(plan['confirmed'] for row in rows for plan in row['rival']),
                far_trials=sum(len(row['rival']) for row in rows))


def run(path, *, seed, count, spread, pace):
    starts = np.random.default_rng(seed).integers(2**31, size=count).tolist()
    station = PrecisionStation(spread, pace)
    start = perf_counter()
    rows = []
    data = dict(schema='tt22-cartesian-precision-1', arena=CONFIG, seed=seed, starts=starts, spread=spread,
                pace=pace, target=station.arena.precision_anchor, rows=rows, verified=0,
                learner='native stereo only', initial_release='ordinary far serve' if spread==pace==1. else 'narrowed far serve')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for seed in starts:
            row = station.point(seed)
            rows.append(row)
            data['metrics'] = metrics(rows)
            data['wall_seconds'] = perf_counter() - start
            _write(path, data)
            print(seed, row['longest'], row['outcome'],
                  [(plan['windows'], plan['predicted_legal'], plan['confirmed'], plan.get('reachable'))
                   for plan in row['rival']], flush=True)
    finally:
        station.close()
    data['native_feeds'] = []
    data['verified'] = replay(rows, spread, pace, feeds=data['native_feeds'])
    _write(path, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held', action='store_true')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--spread', type=float, default=.30)
    parser.add_argument('--pace', type=float, default=.82)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.audit:
        data = _read(args.audit)
        data['native_feeds'] = []
        data['verified'] = replay(data['rows'], data['spread'], data['pace'], feeds=data['native_feeds'])
        _write(args.audit, data)
        print('replayed', data['verified'], 'legal feeds', len(data['native_feeds']))
    elif args.verify:
        saved = _read(args.verify)
        print('replayed', replay(saved['rows'], saved['spread'], saved['pace']))
    else:
        seed = args.seed or (HELD if args.held else PILOT)
        path = args.output or OUTPUT / ('precision-held.json.gz' if args.held else 'precision-pilot.json.gz')
        run(path, seed=seed, count=args.count, spread=args.spread, pace=args.pace)
