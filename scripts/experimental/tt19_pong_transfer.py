"""Test two numeric gains learned on native Pong in a fixed SO-101 receiver."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt8_service_trial import PointInterface
from .tt19_stroke import PREP_TICKS, RallyStrokePoint, StrokeSearch, _local_vector
from .tt19_stroke_trial import IdentityWiring, _arena, verify_physics
from .tt19_view import ViewTransport


TT19 = ROOT / 'artifacts/tt19'
NATIVE_TRAINING = ROOT / 'artifacts/tt3/numeric-pong-source-refinement.json.gz'
NATIVE_PARAMETERS = ROOT / 'artifacts/tt3/numeric-pong-source-refined.json'
RECEIVER = TT19 / 'stroke-strict-rearm-player.json.gz'
RECEIVER_FIXTURE = TT19 / 'stroke-strict-rearm.json.gz'
RECEIVER_PRACTICE = TT19 / 'stroke-receive-practice-strict.json.gz'
SEED = 582617439
POINTS = 36


def native_coefficients():
    """Select the best proposal by native reward per scored start."""
    source = _read(NATIVE_PARAMETERS)
    receipt = _read(NATIVE_TRAINING)
    if receipt['portable_memory'] != source:
        raise ValueError('the native Pong parameters differ from their scored source')
    raw = np.asarray(source['coefficients'])
    totals, counts = np.array(source['totals']), np.array(source['counts'])
    selected = int(np.argmax(totals / counts))
    gains = dict(identity=(1., 1.), native_reward=tuple(raw[selected, 1:]),
                 score_blind=tuple(np.mean(raw[:, 1:], axis=0)))
    return gains, dict(parameters=NATIVE_PARAMETERS.name, parameters_sha256=_hash(NATIVE_PARAMETERS),
                       training=NATIVE_TRAINING.name, training_sha256=_hash(NATIVE_TRAINING),
                       candidate_coefficients=raw.tolist(), native_rewards=totals.tolist(),
                       native_starts=counts.tolist(), selected_candidate=selected,
                       native_screen_starts=len(receipt['screen_cases']) * len(receipt['candidates'])
                       * receipt['points_per_match'][0],
                       native_validation_starts=len(receipt['validation_cases']) * len(raw)
                       * receipt['points_per_match'][1])


class LateralPongStroke(RallyStrokePoint):
    """Change only the lateral target terms of the existing optical aim."""

    def __init__(self, *args, gains, **kwargs):
        super().__init__(*args, **kwargs)
        self.displacement_gain, self.motion_gain = gains
        self.ready_x = None
        self.lateral_aim = []

    def observe(self, frames):
        super().observe(frames)
        blade = self.observations[-1][1]
        if self.ready_x is None and blade is not None:
            self.ready_x = blade[0]

    def _aim(self, tick, command):
        if self.pan_gain is None or self.server != -1 or self.swing_tick is not None or tick < PREP_TICKS:
            return command
        recent = [(index, self.observations[index]) for index in range(tick, max(-1, tick - 2), -1)
                  if all(value is not None for value in self.observations[index][:2])]
        if not recent:
            return command
        current, (ball, blade, _) = recent[0]
        speed = _local_vector(self.observations, current, 0, (-4, -3, -2, -1, 0), self.dt)
        if speed is None or not -7. < speed[1] < -.6 or abs(speed[0]) > 3.5:
            return command
        horizon = (blade[1] - ball[1]) / speed[1]
        if not .09 < horizon < .76 or not -.02 < ball[2] < .7:
            return command
        drift = np.clip(horizon * speed[0], -.11, .11)
        baseline_x = ball[0] + drift
        target = (baseline_x + (self.displacement_gain - 1.) * (ball[0] - self.ready_x)
                  + (self.motion_gain - 1.) * drift)
        if abs(target) > .30:
            return command
        change = float(np.clip(.34 * (target - blade[0]) / self.pan_gain, -.58, .58))
        total = float(np.clip(self.pan_offset + change, -4., 4.))
        correction = total - self.pan_offset
        self.pan_offset = total
        self.lateral_aim.append([tick, float(baseline_x), float(target), correction])
        return (command[0] + correction, *command[1:])


def point(arena, tracker, seed, stride, stroke, pan_gain, gains):
    interface = PointInterface(IdentityWiring(arena), server=-1, seed=SEED)
    player = LateralPongStroke(stroke, stroke, tracker, -1, arena.control_dt,
                              pan_gain=pan_gain, gains=gains)
    player.observe(interface.reset(seed=seed))
    actions = []
    while True:
        action = player.command()
        actions.append(list(action))
        frame, reward, terminated, truncated, _ = interface.step(action, render=len(actions) % stride == 0)
        player.observe(frame)
        if terminated or truncated:
            break
    return dict(seed=seed, game_seed=SEED, server=-1, ticks=len(actions), actions=actions,
                action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest(),
                applied=interface.applied, rgb_sha256=tracker.digest.hexdigest(),
                public_reward=reward, evaluator=arena.point_result(), events=arena.events.copy(),
                swing_ticks=player.swing_ticks, lateral_aim=player.lateral_aim)


def outcomes(rows):
    return dict(points=len(rows), frames=sum(row['ticks'] for row in rows),
                first_legal_returns=sum(row['evaluator']['legal_returns'] >= 1 for row in rows),
                total_legal_returns=sum(row['evaluator']['legal_returns'] for row in rows),
                multiple_returns=sum(row['evaluator']['legal_returns'] >= 2 for row in rows),
                player_contacts=sum(row['evaluator']['contacts'] for row in rows),
                scored_points=sum(row['public_reward'] > 0 for row in rows),
                aim_adjustment_frames=sum(len(row['lateral_aim']) for row in rows))


def compare(before, after):
    if [row['seed'] for row in before] != [row['seed'] for row in after]:
        raise ValueError('receiver candidates did not see identical opening starts')
    pairs = list(zip(before, after, strict=True))
    if any(a['actions'][:PREP_TICKS] != b['actions'][:PREP_TICKS] for a, b in pairs):
        raise ValueError('an intervention changed an action before lateral aim starts')
    returns = lambda rows: np.array([row['evaluator']['legal_returns'] > 0 for row in rows], dtype=int)
    change = returns(after) - returns(before)
    better = int(np.sum(change > 0))
    worse = int(np.sum(change < 0))
    random = np.random.default_rng(832954316)
    draws = random.integers(len(change), size=(8192, len(change)))
    return dict(same_preparation_tapes=len(pairs), changed_action_tapes=sum(
        a['action_sha256'] != b['action_sha256'] for a, b in pairs),
                changed_referee_outcomes=sum(a['evaluator'] != b['evaluator'] for a, b in pairs),
                first_return_gains=better, first_return_losses=worse,
                first_return_fraction_delta=float(change.mean()),
                paired_interval95=np.quantile(change[draws].mean(axis=1), [.025, .975]).tolist(),
                exact_two_sided_p=1. if better + worse == 0 else float(binomtest(better, better + worse).pvalue))


def run(output):
    gains, source = native_coefficients()
    fixture = _read(RECEIVER_FIXTURE)
    stored = _read(RECEIVER)['tt19_stroke']
    if fixture['checkpoint_sha256'] != _hash(RECEIVER) or not fixture['rearm'] or not fixture['arena']['strict_endline']:
        raise ValueError('the frozen receiver differs from its qualified strict game')
    if fixture['pan_calibration']['practice_sha256'] != _hash(RECEIVER_PRACTICE):
        raise ValueError('the receiver pan calibration differs from its source')
    pan_gain = stored['pan_calibration']['gain_m_per_command']
    search = StrokeSearch.restore(stored['policy'])
    stroke = search.choose(-1, train=False)
    starts = _seeds(SEED, 28211, 0, POINTS)['tests']
    used = {row['seed'] for report in (_read(RECEIVER_PRACTICE), fixture)
            for split in ('practice', 'test') for row in report[split]}
    if used.intersection(starts) or len(set(starts)) != POINTS:
        raise ValueError('the held receive starts overlap an earlier SO-101 course')
    settings, cameras, stride = fixture['arena'], fixture['cameras'], fixture['capture_stride']
    arena = _arena(settings, cameras, fixture['custom_table'])
    arms = {name: [] for name in gains}
    order = tuple(gains)
    try:
        for index, start in enumerate(starts):
            for name in order[index % len(order):] + order[:index % len(order)]:
                view = ViewTransport(cameras, state=stored['rgb_calibration'])
                row = point(arena, view.tracker(), start, stride, stroke, pan_gain, gains[name])
                arms[name].append(row)
            if index % 6 == 5:
                print(index + 1, {name: outcomes(rows)['first_legal_returns'] for name, rows in arms.items()}, flush=True)
    finally:
        arena.close()
    all_points = [row for rows in arms.values() for row in rows]
    receipt = dict(schema='tt19-native-pong-forward-v1', seed=SEED, starts=starts, arena=settings,
                   cameras=cameras, capture_stride=stride, custom_table=fixture['custom_table'],
                   target_role_supplied='receive', source=source, gains=gains,
                   receiver=dict(checkpoint=RECEIVER.name, checkpoint_sha256=_hash(RECEIVER),
                                 qualified_game_sha256=_hash(RECEIVER_FIXTURE), stroke=asdict(stroke),
                                 pan_calibration=stored['pan_calibration'], new_practice_points=0,
                                 new_calibration_commands=0),
                   arms={name: dict(points=rows, metrics=outcomes(rows)) for name, rows in arms.items()},
                   paired=dict(native_vs_identity=compare(arms['identity'], arms['native_reward']),
                               native_vs_score_blind=compare(arms['score_blind'], arms['native_reward'])))
    replay = dict(arena=settings, cameras=cameras, custom_table=fixture['custom_table'], practice=[], test=all_points)
    receipt['replayed_points'] = verify_physics(json.loads(json.dumps(replay)))
    _write(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TT19 / 'pong-forward-receive.json.gz')
    args = parser.parse_args()
    result = run(args.output)
    print(result['paired'])
