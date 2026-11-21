"""Test whether accounting for ball arrival time improves scoring via an arm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from arm_plays_pong import ArmPongController, MatchCase, run_match
from button_pong_skill import CONTACT_X, DesiredPongPolicy
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor


class ArrivalController(ArmPongController):
    """Brake for an incoming ball's crossing time with an optional allowance for release."""
    def __init__(self, policy, motor, seed, *, release):
        super().__init__(policy, motor, seed)
        self.release = release
        self.pressed = 0.

    def begin_point(self):
        super().begin_point()
        self.pressed = 0.

    def occupied_button(self, panel):
        button = super().occupied_button(panel)
        down = self.binding.down_button
        sign = 0 if button is None or down is None else (1 if button == down else -1)
        self.pressed += .10 * (sign - self.pressed)
        return button

    def aim(self, screen, tick):
        goal = self.policy.target(screen, tick)
        horizon = float('inf')
        speed = self.policy.ball_velocity
        if screen.ball is not None and speed is not None and speed[0] < -.15:
            horizon = max(0., (CONTACT_X - screen.ball[0]) / speed[0])
        # Velocity decays after release. At a near crossing the paddle can
        # cover just a fraction of its eventual stopping distance.
        coast = .30 * -math.expm1(-horizon / .30)
        error = goal - (screen.paddle_y + coast * self.policy.paddle_velocity)
        if self.release:
            # Optical occupation outlives a request to leave the button;
            # allow some travel even when the paddle is only starting to move.
            error -= .047 * self.pressed * (-math.expm1(-horizon / .16))**2
        return int(error > .012) - int(error < -.012)


def cases(seed, count):
    rng = np.random.default_rng(seed)
    wirings = rng.permutation([0, 3]*(count//2))
    return [MatchCase(int(rng.integers(1_710_000_000, 1_790_000_000)), int(rng.integers(2**31)),
                      int(wiring), int(rng.integers(2**31))) for wiring in wirings]


def match(job):
    case, mode, checkpoint, memory, limit, score = job
    policy = (DesiredPongPolicy() if mode.endswith('_fresh')
              else DesiredPongPolicy.load(checkpoint, learn=True))
    motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=memory)
    controller = None if mode == 'baseline' else ArrivalController(
        policy, motor, case.controller_seed, release=mode.startswith('release'))
    result = run_match(case, policy, motor, first_to=score, max_ticks=limit*120, opponent_speed=.35,
                       panel_kwargs=dict(gear=SOURCE_GEAR), controller=controller)
    return mode, {key:result[key] for key in ('case','winner','seconds','points_won','points_lost',
                                               'scored_after_own_return','own_returns','own_opportunities',
                                               'arm_actions_sha256','drives_sha256','points')}


def summarize(outcomes):
    """Aggregate native matches and resample complete paired starts."""
    totals = {}
    for mode, rows in outcomes.items():
        totals[mode] = {key: sum(row[key] for row in rows) for key in
                        ('points_won', 'points_lost', 'scored_after_own_return', 'own_returns', 'own_opportunities')}
        totals[mode].update(wins=sum(row['winner'] == 'own' for row in rows),
                            losses=sum(row['winner'] == 'opponent' for row in rows),
                            caps=sum(row['winner'] is None for row in rows),
                            median_seconds=float(np.median([row['seconds'] for row in rows])))
    contrasts = {}
    if 'release' in outcomes:
        original = outcomes['release']
        draws = np.random.default_rng(236183).integers(len(original), size=(10000, len(original)))
        for other in ('baseline', 'release_fresh'):
            if other not in outcomes:
                continue
            reference = outcomes[other]
            if [row['case'] for row in original] != [row['case'] for row in reference]:
                raise ValueError('unmatched cases in paired comparison')
            differences = {
                'win_fraction': [int(a['winner'] == 'own') - int(b['winner'] == 'own')
                                 for a, b in zip(original, reference, strict=True)],
                'earned_goals': [a['scored_after_own_return'] - b['scored_after_own_return']
                                 for a, b in zip(original, reference, strict=True)],
            }
            contrasts[other] = {}
            for metric, values in differences.items():
                delta = np.array(values)
                contrasts[other][metric] = dict(mean=float(delta.mean()),
                                                ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
    return dict(totals=totals, release_minus=contrasts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--test', choices=['arrival','release'])
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    memory, _ = pretrain_motor(147884320, gear=SOURCE_GEAR)
    modes = ['baseline', 'arrival', 'release'] if args.test is None else ['baseline', args.test, args.test+'_fresh']
    schedule = cases(15244117 if args.test is None else 33576137, 8 if args.test is None else 24)
    limit = 20 if args.test is None else 90
    jobs = [(case, mode, args.checkpoint, memory, limit, 3) for case in schedule for mode in modes]
    results = {mode:[] for mode in modes}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for mode, result in pool.map(match, jobs):
            results[mode].append(result)
    stats = summarize(results)
    receipt = dict(cases=[asdict(case) for case in schedule], limit_seconds=limit,
                   first_to=3, opponent_speed=.35, gear=SOURCE_GEAR, summary=stats, outcomes=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, separators=(',', ':'))+'\n')
    print(json.dumps(stats, indent=2))





if __name__ == '__main__':
    main()
