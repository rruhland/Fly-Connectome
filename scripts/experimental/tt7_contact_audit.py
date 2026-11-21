"""Screen optical timing edits on old command tapes using a separate referee."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_contact_feasibility import replay
from .tt6_racket import set_racket_response


DIRECTIONS = [(0., .040, 0.), (.012, .035, 0.), (.035, .047, 0.), (.056, .028, 0.),
              (.035, .035, -.19), (.035, .035, .19), (0., 0., -.20), (0., 0., .20),
              (-.02, .045, 0.), (.045, -.02, 0.)]
DEFAULT_DEVELOPMENT = ROOT / 'artifacts/tt7/contact-development.json.gz'


def _candidates():
    for choice in product(range(len(DIRECTIONS)), (-4, -3, -2, -1, 0), (1, 2, 3), (0, 1), (.65, 1.)):
        direction, _, duration, solver, _ = choice
        if solver == 0 or (direction in (1, 2, 3, 8) and duration < 3):
            yield choice


def _edit(actions, design, weight, anchor, candidate):
    index, relative_tick, duration, solver, strength = candidate
    bend = np.array([0., *DIRECTIONS[index]]) * weight
    target = np.r_[bend * strength, np.zeros(actions.shape[1])]
    first = round(anchor) + relative_tick
    frames = slice(first, first + duration)
    edited = actions.copy()
    if solver == 0:
        offset = np.linalg.lstsq(design, target, rcond=None)[0]
        edited[frames] = np.clip(edited[frames] + offset, -1., 1.)
    else:
        for step in range(first, first + duration):
            rates = edited[step]
            offset = lsq_linear(design, target, bounds=(-1.00001 - rates, 1.00001 - rates), method='bvls').x
            edited[step] = np.clip(rates + offset, -1., 1.)
    return edited


def _station(entry):
    geometry, station = entry
    calibration = station['commissioning']
    home = calibration['installation']
    arena = TT6Arena(home['arena_seed'], body='rotary', view_seed=home['view_seed'], **geometry)
    wiring = TT6Wiring(arena, home['arena_seed'])
    weight = np.array([1., 1., 1., .13])
    jacobian = np.asarray(calibration['fit']['response_per_command'])
    design = np.vstack((jacobian * weight[:, None], .013 * np.eye(wiring.arena.action_size)))
    outcomes = []
    try:
        set_racket_response(arena, 'medium')
        baseline = next(row for row in station['runs'] if row['candidate'] == -1)
        for row in baseline['points']:
            original = replay(arena, wiring, row, stop_on_return=False)
            if original['evaluator'] != row['evaluator']:
                raise ValueError(f"development source did not replay for {row['seed']}")
            anchor = row['optical_anchor']
            trials = []
            if anchor is not None:
                actions = np.asarray(row['actions'])
                for candidate in _candidates():
                    edited = _edit(actions, design, weight, anchor, candidate)
                    outcome = replay(arena, wiring, dict(seed=row['seed'], actions=edited.tolist()))
                    trials.append((candidate, outcome['landing'] is not None))
            outcomes.append(dict(seed=row['seed'], anchor=anchor, baseline=original['landing'] is not None,
                                 tested=trials))
        return dict(seed=station['seed'], points=outcomes)
    finally:
        arena.close()


def run(path, development, workers):
    data = _read(development)
    jobs = [(data['geometry'], station) for station in data['stations'] if station['family'] == 'rate']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stations = list(pool.map(_station, jobs))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, dict(schema='tt7-optical-rate-edit-audit-v1', evaluator_only=True,
                      development_sha256=_hash(development), directions=DIRECTIONS, stations=stations))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--development', type=Path, default=DEFAULT_DEVELOPMENT)
    cli.add_argument('--workers', type=int, default=2)
    args = cli.parse_args()
    run(args.output, args.development, args.workers)
