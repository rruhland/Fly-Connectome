"""Replay the first two held gantry tosses with and without motor-map recall."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _read
from .tt9_serve_replay import pair_movie


def movie(folder, output):
    sides = []
    for name, title in [('gantry-serves','Usual calibration'),('gantry-recalled-serves','Two probes + motor memory')]:
        receipt = _read(folder/f'{name}.json.gz')
        receipt['seed'] = receipt['held_seed']
        rows = receipt['study']['fresh']['registered']['points']
        sides.append((receipt,rows,title))
    seeds = [r['seed'] for r in sides[0][1][:2]]
    if [(r['seed'],r['phase']) for r in sides[0][1]] != [(r['seed'],r['phase']) for r in sides[1][1]]:
        raise ValueError('held tosses differ')
    pair_movie(sides,output,seeds)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    movie(a.folder,a.output)
