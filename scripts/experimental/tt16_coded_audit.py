"""Reproduce active-calibration decisions directly from portable optical evidence."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring, Disturbance
from .tt13_live_audit import assignment
from .tt16_coded_probe import decode, excitation, prediction_bank
from .tt16_coded_trial import fit_path
from .tt16_continuous_probe import continued_tape, decide
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path
from .tt16_transport_calibration import identify


def audit(folder, fixture):
    spec_path = setup_path(folder, fixture)
    spec = _read(spec_path)
    arena = make_arena(spec)
    counts = {}
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        src = memory.source
        plain = _read(fit_path(folder, fixture))
        for kind, path in (
                ('ordinary', fit_path(folder,fixture)), ('transport', folder/f'coded-{fixture}-transport.json.gz'),
                ('continuous',folder/f'coded-{fixture}-continuous.json.gz')):
            record = _read(path)
            if record['setup_sha256'] != _hash(spec_path):
                raise ValueError('inconsistent optics preparation')
            if kind == 'ordinary':
                commands = np.asarray(record['actions'])
                rows = [dict(record, installation=spec['wiring'], scene=record['seed'])]
            else:
                commands = np.asarray(record['commands'])
                rows = record['rows']
            planned = continued_tape(src) if kind=='continuous' else excitation(src)
            np.testing.assert_array_equal(commands,planned)
            bank = prediction_bank(src, commands)
            data = []
            for row in rows:
                install = row.get('installation',row.get('wiring_seed'))
                wiring = TT6Wiring(arena,install)
                if kind != 'ordinary':
                    transport = row.get('schedule',row.get('transport'))
                    profile = Disturbance(**{k:tuple(map(tuple,v)) for k,v in transport.items()})
                    wiring = DelayedWiring(wiring,profile,seed=row['noise_seed'])
                history = collect_probe(wiring, row['scene'], commands)
                _, (_, ticks), seen = _samples([history],player.tracker_factory)
                np.testing.assert_array_equal(ticks,row['ticks'])
                np.testing.assert_array_equal(seen,np.asarray(row['seen'],dtype=float))
                truth = assignment(src.action_size, spec['original_wiring'], install)
                if kind=='ordinary':
                    key, fit = decode(src,commands,ticks,seen)
                    np.testing.assert_allclose([fit[k] for k in fit],[plain['fit'][k] for k in fit],rtol=1e-12)
                    if plain['key'] != key:
                        raise ValueError('ordinary RGB decision diverged')
                    data.append(dict(exact=key==truth))
                elif kind=='transport':
                    result = identify(src,bank,ticks,seen)
                    if result != row['decisions'] or truth != row['truth']:
                        raise ValueError('transport decision or physical assignment diverged')
                    data.append(dict(condition=row['condition'], zero=result['zero']==truth,
                                     timing=result['timing']==truth))
                else:
                    result = decide(src,bank,ticks,seen)
                    if result != row['decision'] or truth != row['truth']:
                        raise ValueError('continued movement decision or physical assignment diverged')
                    data.append(dict(condition=row['mode'], early=result['first']==truth,
                                     adaptive=result['adaptive']==truth, always=result['always']==truth,
                                     used_frames=24 if result['use_second'] else 12))
            counts[kind]=dict(sha256=_hash(path),physical_replays=len(data),cases=data)
    finally:
        arena.close()
    _write(folder/f'coded-{fixture}-audit.json.gz',counts)
    print(fixture, {k:v['physical_replays'] for k,v in counts.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('fixture', choices=FIXTURES)
    args=parser.parse_args()
    audit(args.folder,args.fixture)
