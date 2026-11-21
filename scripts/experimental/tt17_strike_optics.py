"""Measure optical rally continuations on the original, matched practice tapes."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_trial import PointInterface
from .tt10_series_trial import _restore_tracker
from .tt16_transfer_protocol import make_arena, setup_path
from .tt17_strike import OpticalCrossings, rally_value


def measure(folder, site):
    fixture = site+'-0'
    receipt = folder/f'strike-{fixture}-practice-0.json.gz'
    practice = _read(receipt)
    spec = _read(setup_path(Path('artifacts/tt16'),fixture))
    arena = make_arena(spec)
    traces, samples = [], []
    try:
        cache_hall_stereo(arena,reflections=True)
        factory = _restore_tracker(arena.cameras,spec['optics'])
        wiring = TT6Wiring(arena,spec['wiring'])
        for row in practice['points']:
            interface = PointInterface(wiring,prepare_point(arena,row['phase']),server=row['server'],seed=practice['seed'])
            tracker, crossing = factory(), OpticalCrossings()
            frames = interface.reset(seed=row['seed'])
            depths = []
            for tick in range(len(row['actions'])+1):
                tracker.update(frames)
                crossing.observe(tracker.ball,tick)
                depths.append(float(tracker.ball.position[1]) if tracker.ball is not None and tracker.ball.observed else None)
                if tick < len(row['actions']):
                    frames, *_ = interface.step(row['actions'][tick])
            if interface.applied != row['applied'] or arena.point_result() != row['evaluator']:
                raise ValueError('calibration tape did not reproduce')
            traces.append(dict(depths=depths,far_crossings=crossing.clearances))
            if row['strike_choice'] is not None:
                samples.append(dict(row['strike_choice'],reward=rally_value(row['public_reward'],crossing.clearances)))
    finally:
        arena.close()
    _write(folder/f'strike-{site}-optical-practice.json.gz',dict(practice=_hash(receipt),traces=traces,
           learned=dict(observations=samples)))
    print(site,len(samples),sum(len(x['far_crossings']) for x in traces),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site',choices=('five','gantry'))
    a=p.parse_args()
    measure(a.folder,a.site)
