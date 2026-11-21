"""Evaluate material selection from public RGB after unannounced plate switches."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_transfer_model import Catcher, optical_prior
from .tt8_transfer_rebound import ReboundCourt
from .tt8_transfer_trial import MATERIALS, play, verify


CHANGE_HAZARD = .17


class SwitchingCatcher(Catcher):
    def __init__(self, prior):
        self.initial_mass = prior
        self.log_probability = np.log(prior)
        self.learn = True

    def reset(self):
        recovered = (1. - CHANGE_HAZARD) * self.probability + CHANGE_HAZARD * self.initial_mass
        self.log_probability = np.log(recovered)
        super().reset()


def controllers(donors):
    def mass(center, spread, diffuse=0.):
        return Catcher(center, spread, learn=False, diffuse=diffuse).probability

    fitted = np.mean([mass(data['center'], data['spread']) for data in donors.values()], axis=0)
    generic = .5 * (mass(.94, .055) + mass(.65, .070))
    priors = dict(source_mixture=fitted, generic_mixture=generic, generic_hard=mass(.94, .025, .40),
                  generic_broad=mass(.75, .15))
    return {name: SwitchingCatcher(prior) for name, prior in priors.items()}


def summarize(rows, stations, phase_length):
    matrix = {}
    for name in rows[0]['players']:
        catches = np.array([row['players'][name]['catch'] for row in rows], dtype=int)
        matrix[name] = catches.reshape(stations, 2, phase_length)
    rng = np.random.default_rng(78945501)
    bootstrap = rng.integers(0, stations, (5000, stations))
    summary = {}
    for name, counts in matrix.items():
        delta = matrix['source_mixture'] - counts
        intervals = np.percentile(100. * delta[bootstrap].mean(axis=(1, 2, 3)), [2.5, 97.5])
        summary[name] = dict(
            catches=int(counts.sum()),
            first_at_entry=int(counts[:, 0, 0].sum()),
            first_after_switch=int(counts[:, 1, 0].sum()),
            next_two_after_switch=int(counts[:, 1, 1:3].sum()),
            source_difference_pp=float(100. * delta.mean()),
            source_interval_pp=intervals.tolist(),
        )
    return summary


def run(source, seed, stations, phase_length):
    receipt = _read(source)
    donors = {name: optical_prior(receipt, name) for name in MATERIALS}
    rng = np.random.default_rng(seed)
    rows = []
    courts = {name: ReboundCourt(name) for name in MATERIALS}
    assigned = np.resize(np.array(MATERIALS), stations)
    rng.shuffle(assigned)
    for station, start in enumerate(assigned.tolist()):
        end = next(name for name in MATERIALS if name != start)
        materials = [start] * phase_length + [end] * phase_length
        rounds = rng.integers(0, 2**31 - 1, size=len(materials)).tolist()
        group = controllers(donors)
        for t, (material, fall) in enumerate(zip(materials, rounds, strict=True)):
            court = courts[material]
            opening = court.opening(fall)
            tapes = {name: play(court, opening, player, visibility='curtain')
                     for name, player in group.items()}
            rows.append(dict(station=station, fall=fall, material=material, visibility='curtain',
                             shift=t == phase_length, crossing_m=opening.endpoint, time_s=opening.crossing_time,
                             players=tapes))
    return dict(schema='tt8-blind-bank-catch-v1', seed=seed, stations=stations, phase_length=phase_length,
                source_sha256=_hash(source), donors=donors, h=CHANGE_HAZARD, rows=rows,
                implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                       ('tt4_arena.xml', 'tt7_surface_trial.py', 'tt8_transfer_rebound.py',
                                        'tt8_transfer_model.py', 'tt8_transfer_trial.py', 'tt8_transfer_switch.py')},
                summary=summarize(rows, stations, phase_length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt7/surface-read-only.json.gz')
    parser.add_argument('--seed', type=int, default=81266014)
    parser.add_argument('--stations', type=int, default=16)
    parser.add_argument('--phase-length', type=int, default=6)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify(_read(args.output))} rebound catch action tapes.')
        return
    if args.stations < 2 or args.stations % 2 or args.phase_length < 3:
        parser.error('use an even number of stations and a phase of at least three falls')
    receipt = run(args.source, args.seed, args.stations, args.phase_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, receipt)
    print(receipt['summary'])


if __name__ == '__main__':
    main()
