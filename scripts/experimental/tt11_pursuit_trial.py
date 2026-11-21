"""Compare camera-rate arm pursuit under equal command bounds on independent courses."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _response_bank
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import TOLERANCE
from .tt10_target_trial import SOURCES
from .tt11_pursuit_servo import PursuitServo
from .tt11_tracking import TrackingArena, PursuitPlayer
from .tt11_tracking_trial import play as block_play, course


def fast_play(arena, wiring, seed, path, view, policy):
    interface = DelayedWiring(TT6Wiring(arena, wiring), Disturbance(delays=((0, 0), (48, 2), (120, 1)),
                                                                  noise=((0, 0.), (48, .025))), seed=seed)
    interface.reset(seed=seed, render=False)
    arena.cursor_axes = view
    tape, scores, distances = [], [], []
    for goal in path:
        arena.set_target(goal)
        action = policy.act(arena.frame())
        interface.step(action, render=False)
        d = float(np.linalg.norm(arena.paddle - goal))
        distances.append(d)
        scores.append(int(d <= TOLERANCE))
        tape.append(action.tolist())
    bits = np.array(scores).reshape(-1, 24)
    return dict(actions=tape, by_segment=bits.sum(axis=1).tolist(), frames=int(bits.sum()),
                held_segments=int(np.any(bits[:, 1:] * bits[:, :-1], axis=1).sum()),
                mean_distance_m=float(np.mean(distances)))


def run(site, seed, count, output):
    source, game = SOURCES[site]
    root = ROOT/'artifacts/tt9'
    donor = root/f'{source}.json.gz'
    fit = load(donor)[0].models[-1]
    setup = _read(root/f'{game}.json.gz')
    arena = TrackingArena(**dict(setup['arena'], render_mode='projected'))
    jac = fit.jacobian(np.zeros(fit.action_size))[:3]
    common = np.broadcast_to(np.mean(_response_bank()[0], axis=0), fit.kernels.shape)
    models = dict(table=fit, instant=replace(fit, kernels=np.ones_like(fit.kernels)),
                  generic=replace(fit, kernels=common), reactive=fit)
    if site in ('offset_a','offset_b'):
        cursor = ROOT/'artifacts/tt10'/f'target-time-{site[-1]}-player.json.gz'
        models['cursor'] = load(cursor)[0].models[-1]
    rows = []
    try:
        for trial in np.random.default_rng([seed, list(SOURCES).index(site)]).integers(0, 2**31-1, count).tolist():
            path = course(arena, trial)
            for view_name, view in [('original', np.eye(3)), ('rotated', Rotation.random(random_state=trial).as_matrix())]:
                params = arena, setup['installation_seed'], trial, path, view
                cases = dict(block=block_play(*params, policy=PursuitPlayer(jac)))
                for key, model in models.items():
                    cases[key] = fast_play(*params, PursuitServo(model, forecast=key != 'reactive'))
                for result in cases.values():
                    if block_play(*params, tape=result['actions']) != result:
                        raise ValueError('physical replay differed')
                rows.append(dict(seed=trial, view=view_name, path=path.tolist(), rotation=view.tolist(), cases=cases))
    finally:
        arena.close()
    _write(output, dict(site=site, seed=seed, donor=_hash(donor), implementation=_hash(Path(__file__)),
                        controller=_hash(Path(__file__).with_name('tt11_pursuit_servo.py')), rows=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    run(args.site, args.seed, args.count, args.output)
