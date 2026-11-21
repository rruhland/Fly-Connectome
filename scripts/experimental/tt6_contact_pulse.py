"""Try an optically timed up-and-forward burst on fresh softened rotary serves."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _write, load_sources
from .tt6_contact import BASELINE_CONTACT, ContactPlayer
from .tt6_continuous_trial import point_condition
from .tt6_racket import set_racket_response
from .tt6_rotary_adapt_trial import commission
from .tt6_timing import DelayedWiring


class CameraPulsePlayer(ContactPlayer):
    """Convert a brief desired pose increment to anonymous actuator rates."""

    def reset(self):
        super().reset()
        self.pulse = None
        self.expected = None

    def _stroke_goals(self, estimate):
        result = super()._stroke_goals(estimate)
        horizon = result[-1]
        if horizon is not None:
            self.expected = self.scene.tick + horizon
        return result

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        if self.pulse is not None or self.expected is None:
            return command
        horizon = self.expected - self.scene.tick
        if not -.6 <= horizon <= 1.35:
            return command
        jacobian = self.adapter.jacobian(self.servo.effective)
        weight = np.array((1., 1., 1., .20))
        design = jacobian * weight[:, None]
        # Preserve lateral position and pitch while moving briefly up and forward.
        desired = np.array((0., .026, .034, 0.))
        delta = np.linalg.solve(design.T @ design + .0006 * np.eye(design.shape[-1]), design.T @ (weight * desired))
        revised = np.clip(command + np.clip(delta, -.60, .60), -1., 1.)
        self.pulse = dict(tick=self.scene.tick, horizon=horizon, delta=(revised - command).tolist())
        return revised

    def finish(self, reward):
        result = super().finish(reward)
        result['camera_pulse'] = self.pulse
        return result


def run(output, seed, count):
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=140)
    arena, wiring, fit, factory, commissioning = commission(seed, count, geometry)
    outputs = {'baseline': ContactPlayer, 'pulse': CameraPulsePlayer}
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    report = dict(schema='tt6-optical-preimpact-pulse-v1', seed=seed, geometry=geometry,
                  commissioning=commissioning, renderer='cached native with reflections', input_sha256=hashes,
                  implementation_sha256=_hash(Path(__file__)), runs={}, summary={})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, player_type in outputs.items():
            prior = TransferPrior('first_person_mini', native, seed, mini=states['first_person_mini'])
            stroke, follow = BASELINE_CONTACT
            player = player_type(fit, prior, factory, stroke=stroke, follow=follow)
            rows = []
            for start in commissioning['schedule']['tests']:
                response, disturbance = point_condition('soft')
                set_racket_response(arena, response)
                row = _point(DelayedWiring(wiring, disturbance, seed=seed), player, start)
                row.update(_counts(row), phase='soft')
                rows.append(row)
                report['runs'][name] = rows
                report['summary'][name] = _metrics(rows)
                _write(output, report)
                print(name, start, row['contacts'], row['player_returns'], row.get('camera_pulse'), flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=126081551)
    parser.add_argument('--points', type=int, default=10)
    args = parser.parse_args()
    run(args.output, args.seed, args.points)
