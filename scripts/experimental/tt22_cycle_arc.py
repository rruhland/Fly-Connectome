"""Test only the causal smooth pre-bounce velocity fit in fresh physical rallies."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_model_rollout import observed_ball
from .tt22_arc_observer import arc
from .tt22_cycle import CyclePlanner, World
from .tt22_cycle_optical import OpticalReceiver, rally_optical
from .tt22_cycle_optical_trial import OUTPUT, verify_optical


SCHEDULE = 813045290
SOURCE = OUTPUT/'cycle-optical-held.json.gz'
OUTPUT_FILE = OUTPUT/'cycle-arc-live.json.gz'


class ArcVelocityReceiver(OpticalReceiver):
    """Use the ordinary position and spin, changing one incoming velocity estimate."""

    def reset(self):
        super().reset()
        self.ordinary = None
        self.arc_active = False
        self.adjustments = []

    def _estimate(self, tick):
        self.ordinary = super()._estimate(tick)
        self.arc_active = False
        if self.ordinary is None:
            return None
        position, old_speed = self.ordinary
        # Match the existing near-bounce guard. Neither a physical bounce flag
        # nor a teammate's instrumented trajectory reaches the receiver.
        bounced = position[1] < -.32 and any(
            p is not None and p[1] < -.23 and p[2] < .081 for _, p in self.history[-7:])
        if bounced:
            return self.ordinary
        fit = arc(self.history, np.zeros(3))
        if fit is None:
            return self.ordinary
        self.arc_active = True
        speed = fit[1]
        self.adjustments.append(dict(tick=tick, delta_velocity_m_s=(30.*(speed-old_speed)).tolist()))
        return position, speed

    def _shadow(self, tick, estimate):
        super()._shadow(tick, estimate)
        if self.arc_active:
            arena = self.world.arena
            spin = observed_ball(self.history, self.ordinary, self.law).spin
            address = arena._ball_dof
            arena.data.qvel[address+3:address+6] = spin
            mujoco.mj_forward(arena.model, arena.data)


def _report(data):
    summary = {}
    for mode in ('stock', 'arc'):
        rows = [pair[mode] for pair in data['comparison'].values() if mode in pair]
        lengths = [r['legal_alternating_returns'] for r in rows]
        summary[mode] = dict(completed=len(rows), alternating=lengths, total=sum(lengths),
                             ten_or_more=sum(n>=10 for n in lengths),
                             learner_returns=sum(r['result']['legal_returns'] for r in rows),
                             rival_returns=sum(r['result']['rival_returns'] for r in rows),
                             frames=sum(r['result']['frames'] for r in rows))
    return summary


def run(output, count):
    reference = _read(SOURCE)
    samples = reference['rows']['own-practice'][-1]['online_samples']
    styles = [tuple(style) for style in reference['styles']]
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=count).tolist()
    occupied = previous_seeds(OUTPUT)
    if output.exists():
        occupied.difference_update(_read(output)['schedule'])
    if len(set(seeds)) != len(seeds) or occupied.intersection(seeds):
        raise ValueError('fresh optical cycle schedule overlaps a prior probe')
    world = World()
    far = CyclePlanner(world)
    lenses = world.arena.lens_intrinsics()
    players = {'stock': OpticalReceiver(lenses, styles=styles, samples=samples),
               'arc': ArcVelocityReceiver(lenses, styles=styles, samples=samples)}
    style_source = Path(__file__).with_name('tt22_arc_observer.py')
    data = dict(schema='tt22-cycle-arc-live-1', source=SOURCE.name, source_sha256=_hash(SOURCE),
                schedule_seed=SCHEDULE, schedule=seeds, practice_frames=201, styles=styles,
                online_updates_during_test=0, same_position_spin_and_motor_library=True,
                far_policy='same privileged physical cycle', comparison={},
                implementation={file.name: _hash(file) for file in (Path(__file__), style_source,
                                Path(__file__).with_name('tt22_cycle_optical.py'),
                                Path(__file__).with_name('tt22_cycle.py'))})
    started = perf_counter()
    try:
        for i, seed in enumerate(seeds):
            for label in (('stock','arc') if i%2==0 else ('arc','stock')):
                player = players[label]
                row = rally_optical(world, far, player, seed, verbose=True)
                row['adjustments'] = player.adjustments if label=='arc' else []
                row['verified'] = verify_optical([row])
                key = str(seed)
                data['comparison'].setdefault(key, {})[label] = row
                data['seconds'] = perf_counter()-started
                data['summary'] = _report(data)
                _write(output, data)
                print('live arc', seed, label, row['legal_alternating_returns'], row['result'], flush=True)
    finally:
        for player in players.values():
            player.close()
        far.close()
        world.close()
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT_FILE)
    p.add_argument('--count', type=int, default=4)
    a=p.parse_args(); run(a.output,a.count)
