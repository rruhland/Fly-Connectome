"""Put the independent optical table-impact estimator inside the live rally cycle."""

import argparse
from pathlib import Path
from time import perf_counter

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_bounce_filter import state
from .tt22_cycle import CyclePlanner, World
from .tt22_cycle_optical import OpticalReceiver, rally_optical
from .tt22_cycle_optical_trial import OUTPUT, TEACHER, verify_optical


SOURCE = OUTPUT / 'cycle-optical-held.json.gz'
FILTER = OUTPUT / 'bounce-filter-pilot.json.gz'
FRESH = (1551328765, 868361554)
CANONICAL = 1134144980


class BounceReceiver(OpticalReceiver):
    def __init__(self, lenses, *, settings, **kwargs):
        super().__init__(lenses, **kwargs)
        self.settings = settings

    def reset(self):
        super().reset()
        self.fitted_bounce = None
        self.last_bounce_refine = -10
        self.filter_timeline = []

    def _estimate(self, tick):
        original = super()._estimate(tick)
        self.fitted_bounce = state(self.history, self.settings)
        if self.fitted_bounce is None:
            return original
        position, velocity, after, offset = self.fitted_bounce
        self.filter_timeline.append(dict(tick=tick, after=bool(after), impact_offset_frames=float(offset),
                                         vertical_m_s=float(velocity[2]*30.)))
        return position, velocity

    def _shadow(self, tick, estimate):
        super()._shadow(tick, estimate)
        if self.fitted_bounce is not None:
            self.world.arena._received_bounce = bool(self.fitted_bounce[2])

    def _refinement_phase(self, tick, position):
        fresh = (self.fitted_bounce is not None and self.fitted_bounce[2] and self.fitted_bounce[3] >= -2.1
                 and -.82 < position[1] < -.23 and tick-self.start >= 2 and tick-self.last_bounce_refine >= 8)
        if fresh:
            self.last_bounce_refine = tick
            self.replanned = True
            return 'bounce-refine'
        return super()._refinement_phase(tick, position)


def run(output, fresh):
    reference = _read(SOURCE)
    teacher = _read(TEACHER)
    filter_source = _read(FILTER)
    settings = filter_source['calibration']['parameters']
    prior = reference['rows']['own-practice'][-1]['online_samples']
    styles = [tuple(style) for style in reference['styles']]
    occupied = {r['seed'] for rows in reference['rows'].values() for r in rows} | {r['seed'] for r in teacher['rows']}
    occupied |= {r['seed'] for r in filter_source['rows']}
    if occupied.intersection(fresh):
        raise ValueError('fresh starts overlap a source schedule')
    world = World()
    far = CyclePlanner(world)
    lenses = world.arena.lens_intrinsics()
    stock = OpticalReceiver(lenses, styles=styles, samples=prior)
    filtered = BounceReceiver(lenses, styles=styles, samples=prior, settings=settings)
    started = perf_counter()
    data = dict(schema='tt22-cycle-bounce-1', source=SOURCE.name, source_sha256=_hash(SOURCE),
                filter_source=FILTER.name, filter_sha256=_hash(FILTER), filter_settings=settings,
                own_point_updates_during_held=0, far_policy='privileged unchanged physical cycle',
                evaluated_fresh_schedule=fresh, comparison={}, seconds=0.)

    def play(label, receiver, seed):
        row = rally_optical(world, far, receiver, seed, verbose=True)
        row['filter_timeline'] = receiver.filter_timeline if isinstance(receiver,BounceReceiver) else []
        row['verified'] = verify_optical([row])
        data['comparison'].setdefault(str(seed), {})[label] = row
        data['seconds'] = perf_counter()-started
        _write(output,data)
        print('bounce', seed, label, row['legal_alternating_returns'], row['result'], flush=True)

    try:
        old = next(r for r in reference['rows']['teacher-plus-own-optics'] if r['seed']==CANONICAL)
        data['comparison'][str(CANONICAL)] = dict(archived_stock=old)
        play('filter', filtered, CANONICAL)
        for seed in fresh:
            play('stock', stock, seed)
            play('filter', filtered, seed)
    finally:
        filtered.close()
        stock.close()
        far.close()
        world.close()
    data['implementation'] = {name:_hash(Path(__file__).with_name(name)) for name in (
        'tt22_cycle_bounce.py','tt22_bounce_filter.py','tt22_cycle_optical.py','tt22_cycle.py')}
    _write(output,data)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT/'cycle-bounce.json.gz')
    p.add_argument('--fresh', type=int, nargs='+', default=FRESH)
    a=p.parse_args(); run(a.output,a.fresh)
