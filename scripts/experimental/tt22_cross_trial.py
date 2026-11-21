"""Reciprocally pair the ordinary stereo RallyPlayer with the physical cycle rival."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_flight import FlightLaw
from .tt20_natural import NaturalViewTransport
from .tt22_cycle import ARENA, CyclePlanner, World
from .tt22_cycle_optical import OpticalCalibration, OpticalReceiver, rally_optical
from .tt22_cycle_optical_trial import OUTPUT, verify_optical
from .tt22_rally import RallyPlayer


STARTS = 994120365
SOURCE = OUTPUT/'cycle-optical-held.json.gz'
DESTINATION = OUTPUT/'cross-rally.json.gz'
MODES = ('cycle_eye', 'rally_player')


class StereoRallyAdapter:
    """Expose the stock RallyPlayer through the two-arm cycle's optical port."""

    def __init__(self, lenses):
        self.lenses = lenses
        self.player = RallyPlayer(FlightLaw())
        self.calibration = OpticalCalibration(self.player.law)

    @property
    def decisions(self):
        return self.player.shots

    def reset(self):
        self.player.reset()
        self.tracker = NaturalViewTransport('fixed', lens_fov=self.lenses, law=self.player.law,
                                            flight=False, associate=True).tracker()
        self.optical = []

    def command(self, tick, images):
        self.tracker.update(images, tick=tick)
        ball = self.tracker.ball
        position = (np.asarray(ball.position) if ball is not None and ball.observed
                    and self.tracker.sources.get('ball_range')=='stereo' else None)
        if position is not None:
            self.optical.append((tick, position.tolist()))
        return self.player.command(tick, position)

    def close(self):
        self.player.close()


def _summarize(completed):
    result = {}
    for mode in MODES:
        rows = [pair[mode] for pair in completed.values() if mode in pair]
        lengths = [row['legal_alternating_returns'] for row in rows]
        result[mode] = dict(points=len(rows), alternating=lengths, ten_or_more=sum(n>=10 for n in lengths),
                            longest=max(lengths, default=0), total=sum(lengths),
                            near_returns=sum(row['result']['legal_returns'] for row in rows),
                            far_returns=sum(row['result']['rival_returns'] for row in rows),
                            wins=sum(row['reward']>0 for row in rows))
    return result


def run(output):
    source = _read(SOURCE)
    seeds = np.random.default_rng(STARTS).integers(2**31, size=4).tolist()
    used = previous_seeds(OUTPUT)
    if output.exists():
        used.difference_update(_read(output)['seeds'])
    if len(set(seeds)) != 4 or set(seeds).intersection(used):
        raise ValueError('the four native starts must be unique and previously untested')
    world = World()
    far = CyclePlanner(world)
    lenses = world.arena.lens_intrinsics()
    previous = OpticalReceiver(lenses, styles=[tuple(s) for s in source['styles']],
                               samples=source['rows']['own-practice'][-1]['online_samples'])
    players = dict(cycle_eye=previous, rally_player=StereoRallyAdapter(lenses))
    started = perf_counter()
    names = ('tt22_cross_trial.py','tt22_rally.py','tt22_cycle.py','tt22_cycle_optical.py','tt20_natural.py')
    data = dict(schema='tt22-cross-rally-v1', seed_schedule=STARTS, seeds=seeds, arena=ARENA,
                source_cycle=SOURCE.name, source_cycle_sha256=_hash(SOURCE),
                near_inputs=['native fixed stereo RGB', 'control clock', 'own normalized motor requests'],
                ordinary_native_serve=True, far='privileged cycle planner previously sustaining 24/10/32/22',
                new_rally_player_practice_points=0, online_updates_during_test=0,
                comparison={}, implementation={name:_hash(Path(__file__).with_name(name)) for name in names})
    try:
        for index, seed in enumerate(seeds):
            pair = data['comparison'][str(seed)] = {}
            for mode in MODES if index%2==0 else MODES[::-1]:
                policy = players[mode]
                row = rally_optical(world, far, policy, seed, verbose=False)
                if mode=='rally_player':
                    row['optical'] = policy.optical
                row['verified'] = verify_optical([row])
                pair[mode] = row
                data['summary'] = _summarize(data['comparison'])
                data['wall_s'] = perf_counter()-started
                _write(output, data)
                print('cross', seed, mode, row['legal_alternating_returns'], row['result'],
                      len(row['decisions']), flush=True)
    finally:
        for player in players.values():
            player.close()
        far.close()
        world.close()
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DESTINATION)
    args = p.parse_args()
    run(args.output)
