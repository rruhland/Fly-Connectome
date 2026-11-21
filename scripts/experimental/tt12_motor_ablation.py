"""Separate the geometry, actuator timing and optics changes in short calibration."""
import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write


def make_starts(standard, recalled, folder):
    original = _read(standard)
    new = _read(recalled)
    if new['series']['arena'] != original['series']['arena']:
        raise ValueError('different installations')
    fitted = new['models'][new['series']['model']]
    for variant, fields in dict(geometry=('coefficients',),timing=('lags','kernels'),
                                both=('coefficients','lags','kernels'),optics=()).items():
        state = deepcopy(original)
        model = state['models'][state['series']['model']]
        for key in fields:
            model[key] = fitted[key]
        if variant == 'optics':
            state['series']['optics'] = new['series']['optics']
        state['motor_ablation'] = dict(variant=variant, standard=_hash(standard), recalled=_hash(recalled))
        _write(folder/f'gantry-{variant}-start.json.gz',state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('standard','recalled','folder'):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    make_starts(args.standard,args.recalled,args.folder)
