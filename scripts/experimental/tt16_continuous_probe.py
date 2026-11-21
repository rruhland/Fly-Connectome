"""Test an optional second movement without resetting the arm or transport clock."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring, Disturbance
from .tt13_live_audit import assignment
from .tt16_coded_probe import excitation, prediction_bank
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path
from .tt16_transport_calibration import clock_costs


def continued_tape(source):
    first = excitation(source)
    n = source.action_size
    rng = np.random.default_rng(9321357 + n)
    # The second half begins at the pose reached by the first. Enforce the
    # excursion limit across the entire trajectory, including the transition.
    draws = rng.choice([-.7,.7], size=(8192, 5, n))
    extensions = np.pad(draws.repeat(2, axis=1), ((0,0),(0,2),(0,0)))
    bounded = np.max(abs(extensions.cumsum(axis=1) + first.sum(axis=0)), axis=(1,2)) < 3.
    clipped = extensions[bounded]
    candidates = np.concatenate((np.broadcast_to(first, (len(clipped), *first.shape)), clipped),axis=1)
    cumulative = candidates.cumsum(axis=1)
    lag = lfilter(source.kernels.mean(axis=0)-1., [1.], candidates, axis=1)
    signal = cumulative + lag
    singular = np.linalg.svd(signal - signal.mean(axis=1,keepdims=True), compute_uv=False)
    return candidates[singular[:, -1].argmax()]


def decide(source, bank, ticks, seen):
    halves = [ticks <= 12, ticks > 12]
    scores, counts = [], []
    for keep, mid in zip(halves, [6,18], strict=True):
        count = int(keep.sum())
        values = (clock_costs(source, bank, ticks[keep], seen[keep], mid)[0]
                  if count >= 3 else np.zeros(len(bank[0])))
        scores.append(values)
        counts.append(count if count >= 3 else 0)
    keys = bank[0]
    first = scores[0]
    order = np.argsort(first, kind='stable')
    margin = first[order[1]] - first[order[0]]
    uncertain = counts[0] < 8 or margin < .012 or first[order[0]] > .045
    combined = scores[0]*counts[0] + scores[1]*counts[1]
    identity = [list(range(source.action_size)), [1]*source.action_size]
    early = keys[order[0]] if counts[0] else identity
    extended = keys[combined.argmin()] if sum(counts) else identity
    return dict(first=early, always=extended, adaptive=extended if uncertain else early,
                use_second=bool(uncertain), sightings=counts, first_margin_m=float(margin),
                first_error_m=float(first[order[0]]) if counts[0] else None)


def run(folder, fixture):
    spec = _read(setup_path(folder, fixture))
    arena = make_arena(spec)
    rows = []
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        tape = continued_tape(memory.source)
        bank = prediction_bank(memory.source, tape)
        rng = np.random.default_rng([spec['camera_seed'], 8476306])
        for index in range(6):
            wiring_seed, scene, noise = rng.integers(0,2**31-1,size=3).tolist()
            mode = ('ordinary', 'delayed', 'shifting')[index%3]
            if mode == 'ordinary':
                profile = Disturbance()
            elif mode == 'delayed':
                profile = Disturbance(delays=((0, 1 if index == 1 else 3),), noise=((0,.045),))
            else:
                start, middle = (3,0) if index == 2 else (0,3)
                profile = Disturbance(delays=((0,start),(5,middle),(16,start)),
                                      noise=((0,.035),(6,.065),(19,.05)))
            interface = DelayedWiring(TT6Wiring(arena,wiring_seed),profile,seed=noise)
            history = collect_probe(interface,scene,tape)
            if len(history) != 25:
                raise ValueError('24-command counterfactual tape did not complete')
            _,(_,ticks),seen = _samples([history],player.tracker_factory)
            decision = decide(memory.source,bank,ticks,seen)
            rows.append(dict(wiring_seed=wiring_seed, scene=scene, noise_seed=noise, mode=mode,
                             transport=asdict(profile), ticks=ticks.tolist(),
                             seen=[[float(x) if np.isfinite(x) else None for x in row] for row in seen],
                             decision=decision, truth=assignment(arena.action_size,spec['original_wiring'],wiring_seed)))
    finally:
        arena.close()
    _write(folder / f'coded-{fixture}-continuous.json.gz',dict(fixture=fixture, commands=tape.tolist(),rows=rows,
           setup_sha256=_hash(setup_path(folder,fixture))))
    print(fixture,[(r['mode'],r['decision']['use_second'],*[r['decision'][k]==r['truth'] for k in ('first','adaptive','always')]) for r in rows],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('fixture',choices=FIXTURES)
    a=p.parse_args()
    run(a.folder,a.fixture)
