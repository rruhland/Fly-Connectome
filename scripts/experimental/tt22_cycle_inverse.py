"""Use a causal table-and-ball inverse fit in the optical half of the physical cycle."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_cycle import ARENA, CyclePlanner, World
from .tt22_cycle_optical import OpticalReceiver, rally_optical
from .tt22_cycle_optical_trial import OUTPUT, verify_optical
from .tt22_flight_inverse import FlightInverse


MATCHED = OUTPUT/'cycle-arc-live.json.gz'
TRANSFER = OUTPUT/'cycle-optical-held.json.gz'
RESULT = OUTPUT/'cycle-inverse.json.gz'


class RecordedFlightInverse(FlightInverse):
    """Recover the chosen optical fit's virtual path without changing its optimizer."""

    def simulate(self, theta, times, xyz):
        states = super().simulate(theta, times, xyz)
        self.paths.append(states)
        return states

    def path(self, history):
        self.paths = []
        value = self.fit(history)
        if value is None:
            return None
        position, velocity, spin, residual_mm = value
        terminal = np.r_[position+[0.,0.,TABLE_HEIGHT],velocity]
        for batch in reversed(self.paths):
            ends = np.c_[batch[:,-1,1:4],batch[:,-1,8:11]]
            same = np.flatnonzero(np.max(abs(ends-terminal),axis=1)<1.e-9)
            if len(same):
                result = batch[same[0]]
                break
        else:
            raise ValueError('the chosen optical fit has no recorded virtual path')
        # Gravity alone cannot turn a downward velocity upward. Require the
        # reversal to occur near the surface in the learner's half of the table.
        z = result[:,3] - TABLE_HEIGHT
        middle = (result[:-1,1:3]+result[1:,1:3])/2.
        turns = ((result[:-1,10] < -.08) & (result[1:,10] > .08)
                 & (np.minimum(z[:-1],z[1:]) < .115) & (np.abs(middle[:,0]) < .445)
                 & (-.83 < middle[:,1]) & (middle[:,1] < -.03))
        return position, velocity/30., spin, residual_mm, bool(np.any(turns)), len(result)


class InverseReceiver(OpticalReceiver):
    def __init__(self, lenses, **kwargs):
        super().__init__(lenses, **kwargs)
        self.infer = RecordedFlightInverse()

    def reset(self):
        super().reset()
        self.stereo = []
        self.inverse = None
        self.seen_near_bounce = False
        self.fits = []
        self.shadow_fits = []

    def _estimate(self, tick):
        ordinary = super()._estimate(tick)
        point = self.history[-1][1] if self.tracker.sources.get('ball_range')=='stereo' else None
        self.stereo.append((tick,point))
        self.inverse = None
        if ordinary is None:
            return None
        pos, speed = ordinary
        if pos[1] > -.15 or speed[1] > .015:
            self.seen_near_bounce = False
        if speed[1] >= -.022 or pos[1] >= .85:
            return ordinary
        self.inverse = self.infer.path(self.stereo)
        if self.inverse is None:
            return ordinary
        center, velocity, spin, error, bounced, support = self.inverse
        self.seen_near_bounce |= bounced
        self.fits.append(dict(tick=tick, position=center.tolist(), velocity_m_s=(30*velocity).tolist(),
                              spin_rad_s=spin.tolist(), residual_mm=float(error), support=support,
                              bounce_in_fit=bounced, bounce_seen=self.seen_near_bounce))
        return center,velocity

    def _shadow(self, tick, estimate):
        super()._shadow(tick, estimate)
        if self.inverse is None:
            return
        arena = self.world.arena
        velocity = arena._ball_dof
        arena.data.qvel[velocity+3:velocity+6] = self.inverse[2]
        arena._received_bounce = self.seen_near_bounce
        self.shadow_fits.append(dict(tick=tick, bounce_seen=self.seen_near_bounce,
                                     spin_rad_s=self.inverse[2].tolist()))
        mujoco.mj_forward(arena.model, arena.data)


def summary(pairs):
    output = {}
    for mode in ('stock','inverse'):
        rows = [pair[mode] for pair in pairs.values() if mode in pair]
        counts = [row['legal_alternating_returns'] for row in rows]
        output[mode] = dict(points=len(rows), alternating=counts, total=sum(counts),
                            ten_or_more=sum(n>=10 for n in counts), twenty_or_more=sum(n>=20 for n in counts),
                            near=sum(r['result']['legal_returns'] for r in rows),
                            far=sum(r['result']['rival_returns'] for r in rows), wins=sum(r['reward']>0 for r in rows))
    return output


def run(path):
    source = _read(MATCHED)
    optical = _read(TRANSFER)
    seeds = source['schedule']
    if len(seeds)!=4 or any(source['comparison'][str(seed)]['stock']['verified']!=1 for seed in seeds):
        raise ValueError('expected four verified matched original-optics points')
    styles = [tuple(style) for style in optical['styles']]
    samples = optical['rows']['own-practice'][-1]['online_samples']
    world = World()
    far = CyclePlanner(world)
    near = InverseReceiver(world.arena.lens_intrinsics(), styles=styles, samples=samples)
    started = perf_counter()
    data = dict(schema='tt22-cycle-inverse-live-1', source=MATCHED.name, source_sha256=_hash(MATCHED),
                transfer=TRANSFER.name, transfer_sha256=_hash(TRANSFER), seeds=seeds, arena=ARENA, comparison={},
                causal_fit_inputs=['fixed stereo RGB positions through current frame','nominal table','nominal ball'],
                unchanged_far_policy='privileged physical long-cycle opponent', near_online_target_updates=0,
                implementation={name:_hash(Path(__file__).with_name(name)) for name in (
                    'tt22_cycle_inverse.py','tt22_flight_inverse.py','tt22_cycle_optical.py','tt22_cycle.py')})
    try:
        for seed in seeds:
            paired = data['comparison'][str(seed)] = dict(stock=source['comparison'][str(seed)]['stock'])
            inverse = rally_optical(world,far,near,seed,verbose=True)
            inverse['inverse_fits'] = near.fits
            inverse['inverse_shadow_decisions'] = near.shadow_fits
            inverse['verified'] = verify_optical([inverse])
            paired['inverse'] = inverse
            data['summary'] = summary(data['comparison'])
            data['seconds'] = perf_counter()-started
            _write(path,data)
            print('inverse cycle',seed,'old',paired['stock']['legal_alternating_returns'],'fitted',
                  inverse['legal_alternating_returns'],'accepted fits',len(near.fits),flush=True)
    finally:
        near.close()
        far.close()
        world.close()
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=RESULT)
    a=p.parse_args(); run(a.output)
