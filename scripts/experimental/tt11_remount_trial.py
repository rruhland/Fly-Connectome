"""Test fast visual re-registration on two surprise camera changes without an arm reset."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, PULSE, TOLERANCE
from .tt10_target_trial import SOURCES
from .tt11_remount import RemountPlayer, restore, state
from .tt11_tracking import TrackingArena
from .tt11_tracking_trial import course


def play(arena, installation, seed, path, views, switches, *, player=None, tape=None, saved=None):
    hardware = DelayedWiring(TT6Wiring(arena, installation),
                             Disturbance(delays=((0, 0), (48, 2), (120, 1)),
                                         noise=((0, 0.), (48, .025))), seed=seed)
    hardware.reset(seed=seed, render=False)
    actions, scores, distances = [], [], []
    middle = None
    cut = len(path) // 2 + 1
    for t, goal in enumerate(path):
        arena.set_target(goal)
        if tape is None:
            arena.cursor_axes = views[sum(t >= start for start in switches)]
            rgb = arena.frame()
            if t == cut:
                middle = state(player)
                if saved is not None:
                    if middle != saved:
                        raise ValueError('checkpoint prefix diverged')
                    player = restore(saved)
                    pulse = player.command / PULSE
            player.observe(rgb)
            if t % BLOCK == 0:
                pulse = player.act(rgb)
            action = pulse if t % BLOCK < PULSE else np.zeros_like(pulse)
        else:
            action = np.array(tape[t])
        hardware.step(action, render=False)
        d = float(np.linalg.norm(arena.paddle - goal))
        scores.append(int(d <= TOLERANCE))
        distances.append(d)
        actions.append(action.tolist())
    split = np.split(scores, switches)
    return dict(frames=[int(part.sum()) for part in split], actions=actions,
                distance_m=float(np.mean(distances))), middle, player.events if player else []


def run(site, seed, count, checkpoint, output):
    donor, game = SOURCES[site]
    folder = ROOT/'artifacts/tt9'
    src = folder/f'{donor}.json.gz'
    model = load(src)[0].models[-1]
    jac = model.jacobian(np.zeros(model.action_size))[:3]
    setup = _read(folder/f'{game}.json.gz')
    arena = TrackingArena(**dict(setup['arena'], render_mode='projected'))
    rows = []
    try:
        for i, trial in enumerate(np.random.default_rng([seed, list(SOURCES).index(site)]).integers(
                0, 2**31-1, count).tolist()):
            rng = np.random.default_rng(trial)
            path = course(arena, trial)
            # A single target teleport tests false remount detections.
            jump = int(rng.integers(32, 160))
            path[jump:] = path[jump:][::-1]
            switches = [int(rng.integers(49, 76)), int(rng.integers(119, 144))]
            start, second, third = Rotation.random(3, random_state=rng).as_matrix()
            views = [start, second, start if i % 2 == 0 else third]
            cases = {}
            for method in ('transport', 'restart_fit', 'continuous'):
                player = RemountPlayer(jac, method)
                args = (arena, setup['installation_seed'], trial, path, views, switches)
                result, mid, events = play(*args, player=player)
                if play(*args, tape=result['actions'])[0] != result:
                    raise ValueError('physical replay diverged')
                if method == 'transport' and i == 0:
                    full = _read(src)
                    full['tracking'] = mid
                    _write(checkpoint, full)
                    again = play(*args, player=RemountPlayer(jac, method), saved=_read(checkpoint)['tracking'])
                    if again != (result, mid, events):
                        raise ValueError('restored learner diverged')
                cases[method] = dict(result, detected=events)
            rows.append(dict(seed=trial, views=np.array(views).tolist(), switches=switches, jump=jump,
                             returns_to_first=i % 2 == 0, path=path.tolist(), cases=cases))
    finally:
        arena.close()
    _write(output, dict(schema='tt11-mid-pursuit-remount-v1', site=site, seed=seed, rows=rows, donor=_hash(src),
                        checkpoint=_hash(checkpoint), implementation=_hash(Path(__file__)),
                        learner=_hash(Path(__file__).with_name('tt11_remount.py'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--count', type=int, default=10)
    args = p.parse_args()
    run(args.site, args.seed, args.count, args.checkpoint, args.output)
