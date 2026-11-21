"""Paired trials of a moving visual target across physical arms and camera mounts."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, PULSE, TOLERANCE
from .tt10_target_trial import SOURCES, targets
from .tt11_tracking import PursuitPlayer, TrackingArena, starting_probes


def course(arena, seed):
    anchors = targets(arena, seed, 9)
    phase = .5 - .5 * np.cos(np.pi * np.arange(24)[:, None] / 24.)
    return np.concatenate([a + phase * (b - a) for a, b in zip(anchors, anchors[1:])])


def play(arena, wiring_seed, seed, path, view, policy=None, tape=None):
    hardware = DelayedWiring(TT6Wiring(arena, wiring_seed),
                             Disturbance(delays=((0, 0), (48, 2), (120, 1)),
                                         noise=((0, 0.), (48, .025))), seed=seed)
    hardware.reset(seed=seed, render=False)
    arena.cursor_axes = view
    actions, inside, distance = [], [], []
    for t, goal in enumerate(path):
        arena.set_target(goal)
        if tape is not None:
            action = np.array(tape[t])
        else:
            if t % BLOCK == 0:
                pulse = policy.act(arena.frame())
            action = pulse if (t % BLOCK) < PULSE else np.zeros_like(pulse)
        if not np.isfinite(action).all() or np.max(abs(action)) > 1.:
            raise ValueError('invalid motor command')
        hardware.step(action, render=False)
        d = float(np.linalg.norm(arena.paddle - goal))
        actions.append(action.tolist())
        inside.append(int(d <= TOLERANCE))
        distance.append(d)
    bits = np.array(inside).reshape(-1, 24)
    return dict(frames=int(bits.sum()), held_segments=int(np.any(bits[:, 1:] * bits[:, :-1], axis=1).sum()),
                by_segment=bits.sum(axis=1).tolist(), mean_distance_m=float(np.mean(distance)), actions=actions)


def run(site, seed, count, test):
    donor, game = SOURCES[site]
    folder = ROOT/'artifacts/tt9'
    source = folder/f'{donor}.json.gz'
    model = load(source)[0].models[-1]
    jac = model.jacobian(np.zeros(model.action_size))[:3]
    probes = starting_probes(jac)
    setup = _read(folder/f'{game}.json.gz')
    arena = TrackingArena(**dict(setup['arena'], render_mode='projected'))
    archive = []
    try:
        for trial in np.random.default_rng([seed, list(SOURCES).index(site)]).integers(0, 2**31-1, count).tolist():
            path = course(arena, trial)
            views = dict(original=np.eye(3), rotated=Rotation.random(random_state=trial).as_matrix())
            for name, view in views.items():
                all_axes = probes + [i for i in range(model.action_size) if i not in probes]
                pilots = dict(aligned=PursuitPlayer(jac, probe_axes=probes),
                              ordinary=PursuitPlayer(jac, align=False, probe_axes=probes),
                              cold=PursuitPlayer(np.zeros_like(jac), align=False,
                                                 probe_axes=all_axes),
                              cold_fast=PursuitPlayer(np.zeros_like(jac), probe_axes=all_axes, support=.25),
                              online=PursuitPlayer(jac),
                              shuffled=PursuitPlayer(jac[:, ::-1], probe_axes=probes),
                              reactive=PursuitPlayer(jac, probe_axes=probes, lead=False),
                              direct=PursuitPlayer(jac, align=False),
                              frozen=PursuitPlayer(jac, probe_axes=probes, adapt=False))
                outcomes = {}
                for key, player in pilots.items():
                    case = play(arena, setup['installation_seed'], trial, path, view, policy=player)
                    if test and play(arena, setup['installation_seed'], trial, path, view, tape=case['actions']) != case:
                        raise RuntimeError('physical cursor replay changed')
                    outcomes[key] = case
                archive.append(dict(seed=trial, view=name, rotation=view.tolist(), path=path.tolist(), outcomes=outcomes))
    finally:
        arena.close()
    return dict(schema='tt11-moving-target-v1', site=site, seed=seed, samples=count,
                donor=_hash(source), learner=_hash(Path(__file__).with_name('tt11_tracking.py')), implementation=_hash(Path(__file__)), replay=test, rows=archive)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--count', default=10, type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    _write(args.output, run(args.site, args.seed, args.count, args.replay))
