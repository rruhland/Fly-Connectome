"""Learn a local motor map during high-rate pursuit with or without a table prior."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_control import MOTOR_BUDGET, _response_bank
from .tt6_memory import load
from .tt10_target import read_markers
from .tt10_target_trial import SOURCES
from .tt11_pursuit_trial import fast_play
from .tt11_tracking import TrackingArena, orientation
from .tt11_tracking_trial import course, play


class LocalServo:
    def __init__(self, prior, probe=True):
        self.prior = prior.copy()
        self.axes = prior.shape[1]
        self.curve = _response_bank()[0].mean(axis=0)
        self.history = np.zeros((len(self.curve), self.axes))
        self.requested = np.zeros(self.axes)
        self.gram = np.zeros((self.axes, self.axes))
        self.cross = np.zeros((3, self.axes))
        self.view = np.eye(3)*.0004
        self.jac = prior.copy()
        self.last = None
        self.goal = None
        self.velocity = np.zeros(3)
        self.tick = 0
        self.probe = probe

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        eff = self.requested + (self.curve - 1.) @ self.history
        if self.tick % 4 == 0:
            if self.last is not None:
                delta = eff - self.last[0]
                move = pos - self.last[1]
                self.gram += np.outer(delta, delta)
                self.cross += np.outer(move, delta)
                self.view += np.outer(move, self.prior @ delta)
                prior = orientation(self.view) @ self.prior
                self.jac = np.linalg.solve(self.gram + .35*np.eye(self.axes), (self.cross + .35*prior).T).T
            self.last = eff, pos
        if self.goal is not None:
            self.velocity = .65*self.velocity + .35*(goal - self.goal)
        self.goal = goal
        future = self.requested + (self.curve[6:] - 1.) @ self.history[:-6]
        motion = self.jac * self.curve[:6].sum()
        err = goal + 6*self.velocity - pos - self.jac @ (future - eff)
        cmd = motion.T @ np.linalg.solve(motion @ motion.T + .037**2*np.eye(3), err)
        if self.probe and self.tick < 4*self.axes:
            cmd[:] = 0.
            cmd[self.tick//4] = .25
        action = np.clip(cmd, np.maximum(-.25, -MOTOR_BUDGET-self.requested),
                         np.minimum(.25, MOTOR_BUDGET-self.requested))
        self.requested += action
        self.history[1:] = self.history[:-1]
        self.history[0] = action
        self.tick += 1
        return action


def run(site, seed, count, output):
    donor, game = SOURCES[site]
    folder = ROOT/'artifacts/tt9'
    src = folder/f'{donor}.json.gz'
    model = load(src)[0].models[-1]
    prior = model.jacobian(np.zeros(model.action_size))[:3]
    info = _read(folder/f'{game}.json.gz')
    arena = TrackingArena(**dict(info['arena'], render_mode='projected'))
    rows = []
    try:
        for s in np.random.default_rng([seed, list(SOURCES).index(site)]).integers(0, 2**31-1, count).tolist():
            path = course(arena, s)
            for name, view in [('original', np.eye(3)), ('rotated', Rotation.random(random_state=s).as_matrix())]:
                args = arena, info['installation_seed'], s, path, view
                cases = {}
                for k, agent in dict(table=LocalServo(prior), cold=LocalServo(np.zeros_like(prior)),
                                     direct=LocalServo(prior, probe=False)).items():
                    result = fast_play(*args, agent)
                    if play(*args, tape=result['actions']) != result:
                        raise ValueError('physical replay differed')
                    cases[k] = result
                rows.append(dict(seed=s, view=name, cases=cases))
    finally:
        arena.close()
    _write(output, dict(site=site, seed=seed, donor=_hash(src), implementation=_hash(Path(__file__)), rows=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    run(args.site, args.seed, args.count, args.output)
