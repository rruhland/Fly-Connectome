"""Re-identify local motor directions when a cursor moves contrary to the fitted response."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_control import MOTOR_BUDGET
from .tt6_memory import load
from .tt10_target import CursorPlayer, PULSE, TargetArena, read_markers
from .tt10_target_trial import SOURCES, targets


class ReidentifyingPlayer:
    def __init__(self, prior, *, recall):
        self.bank = [prior.copy()]
        self.gram = np.eye(prior.shape[1]) * 1.5
        self.cross = self.gram @ prior.T
        self.requested = np.zeros(prior.shape[1])
        self.previous = None
        self.command = self.requested.copy()
        self.forecast = np.zeros(3)
        self.recall = recall
        self.probe_observations = []
        self.probing = False
        self.bad = 0
        self.epoch = 0
        self.changes = []
        self.reused = []

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        if self.previous is not None:
            motion = pos - self.previous
            if self.probing:
                self.probe_observations.append((self.command.copy(), motion))
                if len(self.probe_observations) == len(self.command):
                    cmd, moved = (np.asarray(a) for a in zip(*self.probe_observations, strict=True))
                    jac = np.linalg.lstsq(cmd, moved, rcond=None)[0].T
                    support = np.zeros_like(jac)
                    if self.recall:
                        loss = [np.mean(np.linalg.norm(cmd @ old.T - moved, axis=1)) for old in self.bank]
                        if min(loss) < .026:
                            choice = int(np.argmin(loss))
                            support = self.bank[choice]
                            self.reused.append((self.epoch, choice))
                    self.gram = cmd.T @ cmd + 1.5 * np.eye(len(self.command))
                    self.cross = cmd.T @ moved + 1.5 * support.T
                    self.probing = False
                    self.bad = 0
            else:
                error = np.linalg.norm(motion - self.forecast)
                self.bad = max(0, self.bad + (2 if error > .028 + .65 * max(np.linalg.norm(motion),
                                                                           np.linalg.norm(self.forecast)) else -1))
                if self.bad >= 3:
                    self.bank.append(np.linalg.solve(self.gram, self.cross).T)
                    self.changes.append(self.epoch)
                    self.probing = True
                    self.probe_observations.clear()
                else:
                    self.gram += np.outer(self.command, self.command)
                    self.cross += np.outer(self.command, motion)
        jac = np.linalg.solve(self.gram, self.cross).T
        lower = np.maximum(-PULSE, -MOTOR_BUDGET - self.requested)
        upper = np.minimum(PULSE, MOTOR_BUDGET - self.requested)
        if self.probing:
            axis = len(self.probe_observations)
            command = np.zeros_like(self.command)
            command[axis] = 1.6 if self.requested[axis] < 0. else -1.6
        else:
            command = jac.T @ np.linalg.solve(jac @ jac.T + .014**2 * np.eye(3), goal - pos)
        self.command = np.clip(command, lower, upper)
        self.requested += self.command
        self.previous = pos
        self.forecast = jac @ self.command
        self.epoch += 1
        return self.command / PULSE


def run(site, seed, count, output):
    from .tt6_body_arena import TT6Wiring
    from .tt6_timing import DelayedWiring, Disturbance
    from .tt10_target import BLOCK, BLOCKS_PER_TARGET
    name, game = SOURCES[site]
    src = ROOT/'artifacts/tt9'/f'{name}.json.gz'
    fit = load(src)[0].models[-1]
    jac = fit.jacobian(np.zeros(fit.action_size))[:3]
    start = _read(ROOT/'artifacts/tt9'/f'{game}.json.gz')
    native = start['installation_seed']
    arena = TargetArena(**dict(start['arena'], render_mode='projected'))
    rows = []
    try:
        for course in np.random.default_rng(seed).integers(0, 2**31 - 1, size=count).tolist():
            goals = targets(arena, course, 16)
            policies = dict(recall=ReidentifyingPlayer(jac, recall=True),
                            reidentify=ReidentifyingPlayer(jac, recall=False),
                            continuous=CursorPlayer(jac, exploration='evidence'),
                            cold=CursorPlayer(np.zeros_like(jac), exploration='probe'))
            trials = {}
            for key, policy in policies.items():
                wiring = DelayedWiring(TT6Wiring(arena, native), Disturbance(delays=((0, 0), (90, 2), (240, 1)),
                                                                            noise=((0, 0.), (90, .025))), seed=course)
                wiring.reset(seed=course, render=False)
                scores, tapes = [], []
                for i, goal in enumerate(goals):
                    if i in (4, 10):
                        wiring.wiring = TT6Wiring(arena, course if i == 4 else native)
                    arena.set_target(goal)
                    score = 0
                    for _ in range(BLOCKS_PER_TARGET):
                        pulse = policy.act(arena.frame())
                        for tick in range(BLOCK):
                            action = pulse if tick < PULSE else np.zeros_like(pulse)
                            _, reward, _, _, _ = wiring.step(action, render=False)
                            score += reward
                            tapes.append(action.tolist())
                    scores.append(score)
                trials[key] = dict(scores=scores, actions=tapes, changes=getattr(policy, 'changes', []),
                                   recalled=getattr(policy, 'reused', []))
            rows.append(dict(seed=course, goals=[g.tolist() for g in goals], trials=trials))
    finally:
        arena.close()
    _write(output, dict(site=site, rows=rows, seed=seed, source=_hash(src), implementation=_hash(Path(__file__))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sessions', type=int, default=10)
    args = p.parse_args()
    run(args.site, args.seed, args.sessions, args.output)
