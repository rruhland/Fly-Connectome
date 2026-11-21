"""Compare two visual-control cadences at the arm's original joint speeds."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from arm_plays_pong import MatchCase, NATIVE_HZ, SCREEN_TICKS, run_match
from arm_pong_arrival import ArrivalController
from button_pong_skill import DesiredPongPolicy
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor


CONDITIONS = {'trained_120hz': (True, False), 'trained_30hz': (True, True), 'fresh_30hz': (False, True)}
CASE_SEED = 50747035
ARM_PRACTICE_SEED = 147884320
ARM_PRACTICE_TICKS = 80
MATCHES = 12
FIRST_TO = 3
MAX_TICKS = 55 * NATIVE_HZ
OPPONENT_SPEED = .35


class CameraRateController(ArrivalController):
    """At the slower rate, consume arm and game observations together."""

    def __init__(self, policy, motor, seed, *, hold):
        super().__init__(policy, motor, seed, release=True)
        self.hold = hold
        self.arm_camera_reads = 0
        self.held_command_ticks = 0
        self.initial_probe_native_ticks = 0
        self._last_decision = None

    def command(self, arm_frame, screen, tick):
        if self.hold and screen is None:
            decision = self._last_decision
            # No frame or decision is sampled. Carry the previous optical
            # occupation forward to preserve the trace filters' native clock.
            button, down = decision.visual_button, decision.inferred_down_button
            sign = 0 if button is None or down is None else (1 if button == down else -1)
            self.pressed += .10 * (sign - self.pressed)
            self.binding.advance(button)
            self.held_command_ticks += 1
        else:
            decision = super().command(arm_frame, screen, tick)
            self.arm_camera_reads += 1
            self._last_decision = decision
        self.initial_probe_native_ticks += int(decision.probing)
        return decision


def cases():
    rng = np.random.default_rng(CASE_SEED)
    games = rng.choice(90_000_000, MATCHES, replace=False) + 1_800_000_000
    panel = rng.integers(2**32, size=MATCHES)
    learner = rng.integers(2**32, size=MATCHES)
    wiring = rng.permutation(np.resize((0, 3), MATCHES))
    return [MatchCase(*map(int, row)) for row in zip(games, panel, wiring, learner, strict=True)]


def _one_match(job):
    index, condition, case, checkpoint, memory = job
    trained, hold = CONDITIONS[condition]
    policy = DesiredPongPolicy.load(checkpoint) if trained else DesiredPongPolicy()
    motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=memory)
    controller = CameraRateController(policy, motor, case.controller_seed, hold=hold)
    result = run_match(case, policy, motor, first_to=FIRST_TO, max_ticks=MAX_TICKS,
                       opponent_speed=OPPONENT_SPEED, panel_kwargs={'gear': SOURCE_GEAR}, controller=controller)
    camera_reads = result['screen_samples'] if hold else result['ticks']
    if controller.arm_camera_reads != camera_reads or controller.held_command_ticks != result['ticks'] - camera_reads:
        raise AssertionError('arm commands did not follow the requested camera cadence')
    if result['screen_samples'] != sum((point['physics_ticks'] + SCREEN_TICKS - 1) // SCREEN_TICKS
                                       for point in result['points']):
        raise AssertionError('game camera did not sample once per screen interval and point start')
    result.update(arm_camera_reads=controller.arm_camera_reads, held_command_ticks=controller.held_command_ticks,
                  initial_probe_native_ticks=controller.initial_probe_native_ticks)
    return index, condition, result


def _summarize(results):
    summary = {}
    for condition, rows in results.items():
        quantities = ('points_won', 'points_lost', 'scored_after_own_return', 'own_returns', 'own_opportunities',
                      'seconds', 'ticks', 'screen_samples', 'arm_camera_reads', 'held_command_ticks')
        totals = {key: sum(row[key] for row in rows) for key in quantities}
        totals.update(own_match_wins=sum(row['winner'] == 'own' for row in rows),
                      opposing_match_wins=sum(row['winner'] == 'opponent' for row in rows),
                      capped_matches=sum(row['winner'] is None for row in rows),
                      first_returns=sum(bool(row['first_incoming'] and row['first_incoming']['success']) for row in rows),
                      first_opportunities=sum(row['first_incoming'] is not None for row in rows))
        summary[condition] = totals
    paired = {}
    draws = np.random.default_rng(702193).integers(MATCHES, size=(5000, MATCHES))
    for active, control in (('trained_30hz', 'trained_120hz'), ('trained_30hz', 'fresh_30hz')):
        delta = np.array([a['scored_after_own_return'] - b['scored_after_own_return']
                          for a, b in zip(results[active], results[control], strict=True)])
        paired[f'{active}_minus_{control}'] = dict(goals_after_return_per_match=float(delta.mean()),
                paired_ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
    return summary, paired


def evaluate(checkpoint, workers):
    source = Path(__file__).resolve()
    dependencies = ('arm_plays_pong.py', 'arm_pong_arrival.py', 'button_pong_skill.py', 'pong_button_panel.py')
    inputs = (source, *(source.with_name(name) for name in dependencies), checkpoint)
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}
    memory, practice = pretrain_motor(ARM_PRACTICE_SEED, ticks=ARM_PRACTICE_TICKS, gear=SOURCE_GEAR)
    schedule = cases()
    results = {key: [None] * MATCHES for key in CONDITIONS}
    with ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        futures = [pool.submit(_one_match, (i, condition, case, checkpoint, memory))
                   for i, case in enumerate(schedule) for condition in CONDITIONS]
        for count, future in enumerate(as_completed(futures), 1):
            index, condition, result = future.result()
            results[condition][index] = result
            if count % 6 == 0:
                print(f'Camera-rate matches {count}/{len(futures)}', flush=True)
    for matched in zip(*results.values(), strict=True):
        signatures = [(row['case'], row['initial_down_button'], row['points'][0]['serve'],
                       {key: value for key, value in (row['first_incoming'] or {}).items() if key != 'success'})
                      for row in matched]
        if signatures.count(signatures[0]) != len(signatures):
            raise AssertionError('paired cases differ before the first left crossing')
    if hashes != {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}:
        raise AssertionError('experiment source or checkpoint changed while running')
    summary, paired = _summarize(results)
    return dict(case_seed=CASE_SEED, schedule=[asdict(case) for case in schedule], first_to=FIRST_TO,
                max_ticks=MAX_TICKS, opponent_speed=OPPONENT_SPEED, panel_gear=SOURCE_GEAR, native_hz=NATIVE_HZ,
                paired_bootstrap=dict(draws=5000, seed=702193, unit='matched independent match'),
                arm_practice_seed=ARM_PRACTICE_SEED, arm_practice=practice, arm_practice_camera_hz=NATIVE_HZ,
                arm_practice_memory=memory, source_sha256=hashes, summary=summary, paired=paired, matches=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    arguments = parser.parse_args()
    report = evaluate(arguments.checkpoint, arguments.workers)
    with arguments.output.open('x') as output:
        json.dump(report, output, allow_nan=False, separators=(',', ':'))
        output.write('\n')
    print(json.dumps(dict(summary=report['summary'], paired=report['paired']), indent=2))
