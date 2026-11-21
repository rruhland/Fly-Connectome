"""Resume a changed-arm game with its saved motor fit and no dry calibration."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases
from .tt10_series_trial import PHASES
from .tt14_gain_model import MotorMap
from .tt14_gain_trial import load_map
from .tt17_allocation import AllocatedServo, NativeTravelServo
from .tt17_motor_protocol import setup_path
from .tt17_motor_trial import play, prefix

SERVOS = {'bounded': None, 'travel': NativeTravelServo, 'allocate': AllocatedServo}


def resume(folder, case, seed, method, source):
    spec = _read(setup_path(folder, case))
    stem = prefix(folder, case, source)
    checkpoint = stem.with_name(stem.name + '-player.json.gz')
    parent = _read(stem.with_suffix('.json.gz'))
    digest = _hash(checkpoint)
    if digest != parent['checkpoint']:
        raise ValueError('donor differs from the verified parent game')

    spec.update(case=case, game_seed=seed, donor=str(checkpoint.resolve().relative_to(ROOT)),
                donor_sha256=digest, starts=_seeds(seed, 6011, 0, 65)['tests'],
                phases=match_phases(PHASES, 65, [seed, 6011]),
                followups=_seeds(seed, 540917, 0, 65)['tests'])
    destination = folder / f'resume-{source}-{seed}'
    destination.mkdir(exist_ok=True)
    setup = setup_path(destination, case)
    if setup.exists():
        if _read(setup) != spec:
            raise ValueError('existing fixture has different provenance')
    else:
        # Multiple controllers can resume the same donor/seed concurrently.
        with TemporaryDirectory(dir=destination) as scratch:
            pending = Path(scratch) / setup.name
            _write(pending, spec)
            pending.replace(setup)
    base, transform = load_map(checkpoint)
    motor = MotorMap(base, transform, parent['final_map']['kind'])
    return play(destination, case, 'bounded_frozen', initial_motor=motor, preparation_override={},
                label=method, servo_factory=SERVOS[method])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('case')
    parser.add_argument('seed', type=int)
    parser.add_argument('method', choices=SERVOS)
    parser.add_argument('--source', default='travel')
    args = parser.parse_args()
    resume(args.folder, args.case, args.seed, args.method, args.source)
