"""Identify which public motor channels produce familiar physical movement directions."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_control import MOTOR_BUDGET
from .tt6_memory import load
from .tt10_target import PULSE, TOLERANCE, CursorPlayer, TargetArena, read_markers
from .tt10_target_trial import SOURCES, play, targets


class MatchedMotorPlayer:
    def __init__(self, prior, *, match=True):
        self.prior = prior
        self.axes = prior.shape[1]
        self.match = match
        self.measured = {}
        self.covariance = 1.5 * np.eye(self.axes)
        self.cross = np.zeros((self.axes, 3))
        self.requested = np.zeros(self.axes)
        self.last_command = self.requested.copy()
        self.previous_pos = None
        self.test_axis = None
        self.probes = 0

    def projected_map(self):
        estimate = np.zeros_like(self.prior)
        if not self.measured or not self.match:
            return estimate
        channels = list(self.measured)
        observed = np.stack([self.measured[k] for k in channels])
        positive = np.linalg.norm(observed[:, None, :] - self.prior.T, axis=2)
        negative = np.linalg.norm(observed[:, None, :] + self.prior.T, axis=2)
        src, dest = linear_sum_assignment(np.minimum(positive, negative))
        for row, col in zip(src, dest, strict=True):
            estimate[:, channels[row]] = self.prior[:, col] * (1. if positive[row, col] <= negative[row, col] else -1.)
        return estimate

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        if self.previous_pos is not None:
            move = pos - self.previous_pos
            self.covariance += np.outer(self.last_command, self.last_command)
            self.cross += np.outer(self.last_command, move)
            if self.test_axis is not None:
                self.measured[self.test_axis] = move / self.last_command[self.test_axis]
        matrix = np.linalg.solve(self.covariance, self.cross + 1.5 * self.projected_map().T).T
        offset = goal - pos
        lower = np.maximum(-PULSE, -MOTOR_BUDGET - self.requested)
        upper = np.minimum(PULSE, MOTOR_BUDGET - self.requested)
        planned = matrix.T @ np.linalg.solve(matrix @ matrix.T + .014**2 * np.eye(3), offset)
        command = np.clip(planned, lower, upper)
        missing = next((i for i in range(self.axes) if i not in self.measured), None)
        self.test_axis = None
        # Spend an in-game probe only while the known directions cannot cover
        # most of the current requested movement in one pulse.
        if missing is not None and np.linalg.norm(offset - matrix @ command) > max(TOLERANCE, .42 * np.linalg.norm(offset)):
            self.test_axis = missing
            command[:] = 0.
            command[missing] = 1.6 if self.requested[missing] <= 0. else -1.6
            self.probes += 1
        self.requested += command
        self.last_command = command
        self.previous_pos = pos
        return command / PULSE


def run(seed, sessions):
    rows = []
    for i, (site, (name, game)) in enumerate(SOURCES.items()):
        archive = ROOT / 'artifacts/tt9'
        models = load(archive / f'{name}.json.gz')[0].models
        fit = models[-1]
        prior = fit.jacobian(np.zeros(fit.action_size))[:3]
        start = _read(archive / f'{game}.json.gz')
        arena = TargetArena(**dict(start['arena'], render_mode='projected'))
        try:
            for j in range(sessions):
                course = int(np.random.SeedSequence([seed, i, j]).generate_state(1)[0])
                goals = targets(arena, course, 8)
                for changed, wiring in [(False, start['installation_seed']), (True, course)]:
                    group = dict(matched=MatchedMotorPlayer(prior), empty=MatchedMotorPlayer(prior, match=False),
                                 probe_first=CursorPlayer(np.zeros_like(prior), exploration='probe'),
                                 direct=CursorPlayer(prior, exploration='evidence'),
                                 rotated=MatchedMotorPlayer(np.roll(prior, 1, axis=0)))
                    cases = {}
                    for key, agent in group.items():
                        case = play(arena, goals, agent, wiring, course)
                        repeated = play(arena, goals, None, wiring, course, tape=case['actions'])
                        if case['scores'] != repeated['scores']:
                            raise ValueError('physical replay differed')
                        cases[key] = dict(case, probes=agent.probes)
                    rows.append(dict(site=site, session=course, rewired=changed, wiring=wiring,
                                     goals=[p.tolist() for p in goals], runs=cases))
        finally:
            arena.close()
    return dict(schema='tt10-rebound-motor-directions-v1', rows=rows, seed=seed, sessions=sessions,
                implementation_sha256=_hash(Path(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, default=746818843)
    p.add_argument('--sessions', type=int, default=8)
    a = p.parse_args()
    result = run(a.seed, a.sessions)
    _write(a.output, result)
    for changed in [False, True]:
        rows = [r for r in result['rows'] if r['rewired'] == changed]
        print(changed, {k: sum(sum(r['runs'][k]['scores']) for r in rows) for k in rows[0]['runs']})
