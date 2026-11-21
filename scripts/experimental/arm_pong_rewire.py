"""Assess continued learning of two independent button effects during native Pong."""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from arm_plays_pong import ArmPongController, MatchCase, NATIVE_HZ, run_match
from button_pong_skill import DesiredPongPolicy
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor


SWITCH = 7 * NATIVE_HZ
DURATION = 22 * NATIVE_HZ
PRACTICE_SEED = 8153017


class IndependentEffects:
    """Estimate each button's own signed response from its observed occupation.

    A two-stage leaky trace turns recent image overlap into expected motion
    magnitude. Only assign a paddle displacement when one trace dominates.
    A zero estimate represents an untested button. Both need signed evidence.
    """

    def __init__(self, guess, *, lock_after_acquisition):
        self.guess = guess
        self.order = (guess, 1 - guess)
        self.lock = lock_after_acquisition
        self.learn = True
        self.estimate = [0., 0.]
        self.signs = [0, 0]
        self.events = []
        self.samples = 0
        self.reset_trace()

    def reset_trace(self):
        self.occupation = np.zeros(2)
        self.speed = np.zeros(2)
        self.distance = np.zeros(2)
        self.history = deque(maxlen=5)

    @property
    def down_button(self):
        return self.signs.index(1) if sorted(self.signs) == [-1, 1] else None

    def advance(self, visible):
        self.occupation *= .90
        if visible is not None:
            self.occupation[visible] += .10
        self.speed += .035 * (.74 * self.occupation - self.speed)
        self.distance += self.speed / NATIVE_HZ

    def observe(self, paddle, tick):
        self.history.append((paddle, self.distance.copy()))
        if not self.learn or len(self.history) < 3:
            return
        old_y, old_distance = self.history[0]
        delta = self.distance - old_distance
        button = int(np.argmax(delta))
        if delta[button] < .005 or delta[1 - button] > .30 * delta[button]:
            return
        motion = paddle - old_y
        if abs(motion) < .003:
            return
        response = np.clip(motion / delta[button], -1., 1.)
        self.estimate[button] += .42 * (response - self.estimate[button])
        self.samples += 1
        if abs(self.estimate[button]) < .53:
            return
        sign = int(np.sign(self.estimate[button]))
        if sign != self.signs[button]:
            self.signs[button] = sign
            self.events.append(dict(tick=tick, button=button, sign=sign,
                                    signs=self.signs.copy(), down_button=self.down_button))
            if self.lock and self.down_button is not None:
                self.learn = False

    def resolve(self, direction):
        if self.down_button is None:
            for index in self.order:
                if self.signs[index] == 0:
                    return index
            # Two estimates now disagree with the known two-action task.
            # Re-test the older one; its effect is still learned independently.
            return 1 - self.events[-1]['button']
        return None if direction == 0 else self.signs.index(direction)


class TwoButtonController(ArmPongController):
    def __init__(self, policy, motor, seed, *, lock):
        super().__init__(policy, motor, seed, initial_probe=False)
        self.binding = IndependentEffects(self.binding.guess, lock_after_acquisition=lock)

    def command(self, arm_frame, screen, tick):
        decision = super().command(arm_frame, screen, tick)
        exploring = self.binding.down_button is None
        self._probe_ticks += int(exploring)
        return replace(decision, probing=exploring)


def schedule():
    rng = np.random.default_rng(416027)
    wires = rng.permutation([0, 3] * 8)
    return [MatchCase(int(rng.integers(1_800_000_000, 2_100_000_000)),
                      int(rng.integers(2**30)), int(wire), int(rng.integers(2**30)))
            for wire in wires]


def run_case(case, checkpoint, lock=False, *, callback=None, return_trace=False, contact_enabled=True):
    memory, _ = pretrain_motor(PRACTICE_SEED, gear=SOURCE_GEAR)
    motor = VisualButtonMotor(case.controller_seed, memory=memory)
    policy = DesiredPongPolicy.load(checkpoint, learn=False, exploration=0.)
    controller = TwoButtonController(policy, motor, case.controller_seed, lock=lock)
    result = run_match(case, policy, motor, first_to=100, max_ticks=DURATION,
                       opponent_speed=.45, panel_kwargs=dict(gear=SOURCE_GEAR),
                       rewire_ticks=(SWITCH,), controller=controller, record=True, callback=callback,
                       contact_enabled=contact_enabled)
    trajectory = result.pop('trajectory')
    phases = {}
    for start, stop, name, expected in ((0, SWITCH, 'before', result['initial_down_button']),
                                         (SWITCH, DURATION, 'after', 1 - result['initial_down_button'])):
        rows = trajectory[start:stop]
        incorrect = [row['tick'] for row in rows if row['inferred_down_button'] != expected]
        # Require correctness through the remaining phase and at least 1 s of verification.
        stable = max(incorrect, default=start - 1) + 1
        phases[name] = dict(returns=sum(row['hit'] for row in rows),
                            incoming=sum(row['opportunity'] for row in rows),
                            won=sum(row['score'] == 1 for row in rows),
                            lost=sum(row['score'] == -1 for row in rows),
                            sustained_both_s=(stable - start) / NATIVE_HZ if stable <= stop - NATIVE_HZ else None)
    result['phases'] = phases
    if return_trace:
        result['trajectory'] = trajectory
    return result


def summarize(rows):
    summary = {}
    for variant in ('adaptive', 'locked'):
        summary[variant] = {}
        for phase in ('before', 'after'):
            phase_rows = [row[variant]['phases'][phase] for row in rows]
            metrics = {key: sum(row[key] for row in phase_rows) for key in ('returns', 'incoming', 'won', 'lost')}
            times = [row['sustained_both_s'] for row in phase_rows if row['sustained_both_s'] is not None]
            metrics.update(sustained=len(times), median_when_sustained_s=float(np.median(times)) if times else None,
                           max_when_sustained_s=max(times, default=None))
            summary[variant][phase] = metrics
    rng = np.random.default_rng(416028)
    draw = rng.integers(len(rows), size=(8000, len(rows)))
    rates = []
    for variant in ('adaptive', 'locked'):
        counts = np.array([[r[variant]['phases']['after'][k] for k in ('returns', 'incoming')] for r in rows])
        hits, opportunities = counts[draw].sum(axis=1).T
        rates.append(hits / np.maximum(1, opportunities))
    summary['paired_after_return_rate_difference_ci95'] = np.quantile(rates[0] - rates[1], [.025, .975]).tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for case in schedule():
        rows.append({name: run_case(case, args.checkpoint, lock=name == 'locked')
                     for name in ('adaptive', 'locked')})
    disconnected = run_case(schedule()[0], args.checkpoint, contact_enabled=False)
    if disconnected['binding_events'] or disconnected['physical_motion_ticks']:
        raise ValueError('a disconnected button created movement evidence')
    args.out.parent.mkdir(exist_ok=True, parents=True)
    receipt = dict(switch_tick=SWITCH, duration_ticks=DURATION, source_gear=SOURCE_GEAR,
                   opponent_speed=.45, summary=summarize(rows), matches=rows,
                   disconnected_first_case={key: disconnected[key] for key in
                                            ('case', 'ticks', 'binding_events', 'physical_motion_ticks',
                                             'own_returns', 'own_opportunities', 'points_won', 'points_lost')})
    args.out.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
    print(json.dumps(receipt['summary'], indent=2))


if __name__ == '__main__':
    main()
