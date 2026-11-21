"""Paired honest-camera transfer and optical-only practice against a physical partner."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_cycle import ARENA, CyclePlanner, World
from .tt22_cycle_optical import OpticalReceiver, rally_optical, teacher_styles
from .tt22_cycle_trial import OUTPUT


TEACHER = OUTPUT/'cycle-envelope-development.json.gz'
PRACTICE = (1588384217, 1871206363)
FRESH = (1134144980, 602925087)
PILOT = (2059027790,)


def verify_optical(rows):
    world = World()
    try:
        for row in rows:
            world.reset(row['seed'])
            image = sha256()
            for near, far in row['actions']:
                for name, rgb in world.arena.render_cameras().items():
                    image.update(name.encode())
                    image.update(rgb)
                _, score, _, _, _ = world.step(near, far)
            if (image.hexdigest() != row['rgb_sha256'] or world.arena.point_result() != row['result']
                    or score != row['reward'] or json.dumps(world.arena.events) != json.dumps(row['events'])):
                raise ValueError(f'optical tape or strict result did not reproduce for {row["seed"]}')
    finally:
        world.close()
    return len(rows)


def run(target, *, pilot=False, verbose=False):
    teacher = _read(TEACHER)
    styles = teacher_styles(teacher['rows'])
    reference_seeds = {row['seed'] for row in teacher['rows']}
    if not styles or reference_seeds.intersection((*PRACTICE, *FRESH)) or set(PRACTICE).intersection(FRESH):
        raise ValueError('source, practice and assessment must be disjoint')
    world = World()
    opposite = CyclePlanner(world)
    lenses = world.arena.lens_intrinsics()
    receipts = dict(schema='tt22-cycle-stereo-1', arena=ARENA, teacher=TEACHER.name, teacher_sha256=_hash(TEACHER),
                    styles=styles, near_policy_inputs=['RGB left/right', 'clock', 'issued near commands'],
                    far_policy='privileged cooperative physical planner', training_credit='future RGB only',
                    pilot=pilot, rows={})
    began = perf_counter()

    def execute(label, seeds, *, source=(), learn=False, samples=()):
        receiver = OpticalReceiver(lenses, styles=source, learn=learn, samples=samples)
        try:
            for seed in seeds:
                row = rally_optical(world, opposite, receiver, seed, verbose=verbose)
                print(label, seed, row['legal_alternating_returns'], row['result'], len(row['online_samples']), flush=True)
                row['verified'] = verify_optical([row])
                receipts['rows'].setdefault(label, []).append(row)
                receipts['seconds'] = perf_counter()-began
                _write(target, receipts)
            return receiver.calibration.state()
        finally:
            receiver.close()

    try:
        if pilot:
            execute('teacher-pilot', PILOT, source=styles)
        else:
            calibration = execute('own-practice', PRACTICE, source=styles, learn=True)
            execute('teacher-frozen', FRESH, source=styles)
            execute('teacher-plus-own-optics', FRESH, source=styles, samples=calibration)
            execute('no-teacher', FRESH)
    finally:
        opposite.close()
        world.close()
    receipts['implementation'] = {name:_hash(Path(__file__).with_name(name)) for name in
                                 ('tt22_cycle_optical.py','tt22_cycle_optical_trial.py','tt22_cycle.py')}
    _write(target, receipts)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    filename = 'cycle-optical-pilot.json.gz' if args.pilot else 'cycle-optical-held.json.gz'
    run(args.output or OUTPUT/filename, pilot=args.pilot, verbose=args.verbose)
