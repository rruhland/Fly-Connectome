"""Identify control wiring and unknown motor delay using a shared short RGB tape."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring, Disturbance
from .tt13_live_audit import assignment
from .tt16_coded_probe import correspondence_cost, excitation, prediction_bank
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path


def clock_costs(source, bank, ticks, observations, midpoint):
    """Retain the best timing explanation for every motor correspondence."""
    keys, poses = bank
    clock = np.arange(poses.shape[1])
    best = np.full(len(poses), np.inf)
    timings = np.zeros((len(poses), 2), dtype=int)
    candidates = [(d, d) for d in range(4)] + [(i, j) for i in (0, 3) for j in (0, 3) if i != j]
    for before, after in candidates:
        delay = np.where(clock <= midpoint, before, after)
        # A longer delay holds the last command; it cannot undo motion.
        time = np.maximum.accumulate(np.maximum(0, clock - delay))
        costs = correspondence_cost(source, poses[:, time[ticks]], observations)
        if (before, after) == (0, 0):
            zero = keys[costs.argmin()]
        if before != after:
            costs += .001
        improved = costs < best
        best[improved] = costs[improved]
        timings[improved] = [before, after]
    return best, timings, zero


def identify(source, bank, ticks, observations):
    if len(ticks) < 3:
        identity = [list(range(source.action_size)), [1] * source.action_size]
        return dict(zero=identity, timing=identity, inferred_clock=None)
    keys, poses = bank
    best, timings, zero = clock_costs(source, bank, ticks, observations, poses.shape[1] // 2)
    winner = best.argmin()
    return dict(zero=zero, timing=keys[winner], inferred_clock=timings[winner].tolist())


def conditions(index):
    fast_first = index % 2 == 0
    a, b = (0, 3) if fast_first else (3, 0)
    return dict(clean=Disturbance(),
                delayed=Disturbance(delays=((0, 2),), noise=((0, .045),)),
                changing=Disturbance(delays=((0, a), (4 if fast_first else 8, b)),
                                     noise=((0, .035), (6, .065))))


def run(folder, fixture):
    spec = _read(setup_path(folder, fixture))
    arena = make_arena(spec)
    rows = []
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        commands = excitation(memory.source)
        bank = prediction_bank(memory.source, commands)
        rng = np.random.default_rng([spec['camera_seed'], 527738])
        for index in range(3):
            installation = spec['wiring'] if index == 0 else int(rng.integers(0, 2**31-1))
            scene, noise_seed = rng.integers(0, 2**31-1, size=2).tolist()
            # The scoring and learning functions receive only commanded actions
            # and optical pose samples. The installation and clock stay here.
            for phase, schedule in conditions(index).items():
                interface = DelayedWiring(TT6Wiring(arena, installation), schedule, seed=noise_seed)
                history = collect_probe(interface, scene, commands)
                if len(history) != len(commands) + 1:
                    raise ValueError('incomplete diagnostic movement')
                _, (_, ticks), seen = _samples([history], player.tracker_factory)
                result = identify(memory.source, bank, ticks, seen)
                truth = assignment(arena.action_size, spec['original_wiring'], installation)
                rows.append(dict(index=index, installation=installation, scene=scene, noise_seed=noise_seed,
                                 condition=phase, schedule=asdict(schedule), ticks=ticks.tolist(),
                                 seen=[[float(x) if np.isfinite(x) else None for x in row] for row in seen],
                                 decisions=result, truth=truth))
    finally:
        arena.close()
    _write(folder / f'coded-{fixture}-transport.json.gz',
           dict(fixture=fixture, setup_sha256=_hash(setup_path(folder, fixture)),
                commands=commands.tolist(), rows=rows))
    print(fixture, [(r['condition'], r['decisions']['zero']==r['truth'], r['decisions']['timing']==r['truth'])
                    for r in rows], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('fixture', choices=FIXTURES)
    args = p.parse_args()
    run(args.folder, args.fixture)
