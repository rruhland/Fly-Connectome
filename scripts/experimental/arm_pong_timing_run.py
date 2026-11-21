"""Compare visual release timing with matched controls at the arm's original speed."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np

from arm_plays_pong import ArmPongController, MatchCase, NATIVE_HZ, run_match
from arm_pong_timing import ReleaseTiming
from button_pong_skill import DesiredPongPolicy
from planar_position_arm import MAX_JOINT_SPEED
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor


CONDITIONS = ('adaptive_release', 'fixed_release', 'ball_masked')
TEST_SEED = 87584152
COURSE_COUNT = 12
FIRST_TO = 3
MAX_TICKS = 18 * NATIVE_HZ
OPPONENT_SPEED = .45
PRACTICE_TICKS = 80
BASE_STOP_SECONDS = .30
AUDIT_MAX_TICK_DIFFERENCE = 6


class ReleaseController(ArmPongController):
    """Leave target choice and direction binding in the common controller."""

    def __init__(self, policy, motor, seed, *, learn_delay, show_ball):
        super().__init__(policy, motor, seed)
        self.timing = ReleaseTiming(1 / NATIVE_HZ, learn=learn_delay)
        self.show_ball = show_ball
        self.first_anticipation_tick = None
        self.anticipation_ticks = 0
        self.changed_intents = 0

    def command(self, arm_frame, screen, tick):
        self._timing_tick = tick
        decision = super().command(arm_frame, screen, tick)
        if not decision.probing:
            self.timing.request(tick, decision.target)
        return decision

    def occupied_button(self, panel):
        held = super().occupied_button(panel)
        self.timing.observe(self._timing_tick, held)
        return held

    def aim(self, screen, tick):
        if not self.show_ball:
            screen = replace(screen, ball=None)
        ordinary = super().aim(screen, tick)
        extra = self.timing.stop_seconds
        if not extra:
            return ordinary
        if self.first_anticipation_tick is None:
            self.first_anticipation_tick = tick
        self.anticipation_ticks += 1
        revised = self.policy.action(screen, tick, stop_seconds=BASE_STOP_SECONDS + extra)
        self.changed_intents += int(revised != ordinary)
        return revised


def cases(seed=TEST_SEED, count=COURSE_COUNT):
    """Reserve game seeds outside direct practice and balance both wirings."""
    if count < 2 or count % 2:
        raise ValueError('the paired course count must be positive and even')
    rng = np.random.default_rng(seed)
    games = rng.choice(500_000_000, size=count, replace=False) + 700_000_000
    schedule = []
    for index, game in enumerate(games):
        panel_seed, controller_seed, wiring_seed = (int(value) for value in rng.integers(2**32, size=3))
        while np.random.default_rng(wiring_seed).permutation((-1, 1))[index % 2] != 1:
            wiring_seed = int(rng.integers(2**32))
        schedule.append(MatchCase(int(game), panel_seed, wiring_seed, controller_seed))
    return schedule


def _release_audit(visual_releases, contacts):
    physical = []
    previous = None
    for event in contacts:
        if previous is not None and event['button'] != previous:
            physical.append((previous, event['tick']))
        previous = event['button']

    offsets = []
    for event in visual_releases:
        eligible = [(abs(event.seen_tick - tick), index) for index, (button, tick) in enumerate(physical)
                    if button == event.button]
        if not eligible:
            continue
        difference, index = min(eligible)
        if difference <= AUDIT_MAX_TICK_DIFFERENCE:
            offsets.append(event.seen_tick - physical.pop(index)[1])
    return offsets


def _campaign(case, checkpoint):
    memory, practice = pretrain_motor(case.controller_seed, ticks=PRACTICE_TICKS, gear=SOURCE_GEAR)
    matches = {}
    for condition in CONDITIONS:
        policy = DesiredPongPolicy.load(checkpoint, learn=False, exploration=0.)
        motor = VisualButtonMotor(case.controller_seed, memory=memory)
        controller = ReleaseController(policy, motor, case.controller_seed,
                                       learn_delay=condition == 'adaptive_release', show_ball=condition != 'ball_masked')
        result = run_match(case, policy, motor, first_to=FIRST_TO, max_ticks=MAX_TICKS,
                           opponent_speed=OPPONENT_SPEED, panel_kwargs={'gear': SOURCE_GEAR}, controller=controller)
        releases = controller.timing.releases
        offsets = _release_audit(releases, result['contacts'])
        result['release_timing'] = dict(
            samples=[asdict(release) for release in releases], physical_audit_offsets=offsets,
            first_anticipation_tick=controller.first_anticipation_tick,
            anticipation_frames=controller.anticipation_ticks, changed_intents=controller.changed_intents,
        )
        matches[condition] = result

    signatures = [None if row['first_incoming'] is None else
                  {key: value for key, value in row['first_incoming'].items() if key != 'success'}
                  for row in matches.values()]
    if signatures.count(signatures[0]) != len(signatures):
        raise AssertionError('matched conditions have different incoming trajectories before first contact')
    return dict(case=asdict(case), arm_practice=practice, matches=matches)


def _summarize(rows):
    quantities = ('own_returns', 'own_opportunities', 'points_won', 'points_lost',
                  'scored_after_own_return', 'last_touch_scores', 'seconds', 'physical_motion_ticks')
    reports = {}
    for condition in CONDITIONS:
        matches = [row['matches'][condition] for row in rows]
        report = {key: sum(match[key] for match in matches) for key in quantities}
        report.update(first_returns=sum(bool(match['first_incoming'] and match['first_incoming']['success'])
                                        for match in matches),
                      first_opportunities=sum(match['first_incoming'] is not None for match in matches),
                      own_match_wins=sum(match['winner'] == 'own' for match in matches),
                      opponent_match_wins=sum(match['winner'] == 'opponent' for match in matches),
                      capped_matches=sum(match['winner'] is None for match in matches),
                      changed_intents=sum(match['release_timing']['changed_intents'] for match in matches),
                      anticipatory_frames=sum(match['release_timing']['anticipation_frames'] for match in matches))
        release_ticks = [event['seen_tick'] - event['request_tick'] for match in matches
                         for event in match['release_timing']['samples']]
        offsets = [offset for match in matches for offset in match['release_timing']['physical_audit_offsets']]
        report.update(camera_releases=len(release_ticks), confirmed_camera_releases=len(offsets),
                      median_release_seconds=None if not release_ticks else median(release_ticks) / NATIVE_HZ,
                      median_absolute_contact_audit_seconds=None if not offsets else median(map(abs, offsets)) / NATIVE_HZ)
        reports[condition] = report
    return reports


def _paired_effects(rows):
    rng = np.random.default_rng(199728538)
    draws = rng.integers(len(rows), size=(5000, len(rows)))

    def counts(condition):
        return np.array([(match['own_returns'], match['own_opportunities'], match['scored_after_own_return'])
                         for match in (row['matches'][condition] for row in rows)])

    def comparison(active, reference):
        a, b = counts(active), counts(reference)
        a_sums, b_sums = a[draws].sum(axis=1), b[draws].sum(axis=1)
        supported = (a_sums[:, 1] > 0) & (b_sums[:, 1] > 0)
        hit_differences = (a_sums[supported, 0] / a_sums[supported, 1]
                           - b_sums[supported, 0] / b_sums[supported, 1])
        goals_difference = a_sums[:, 2] / len(rows) - b_sums[:, 2] / len(rows)
        a_returns, a_opportunities, _ = a.sum(axis=0)
        b_returns, b_opportunities, _ = b.sum(axis=0)
        return dict(return_rate_difference=None if not (a_opportunities and b_opportunities)
                    else a_returns / a_opportunities - b_returns / b_opportunities,
                    paired_return_rate_ci95=None if not len(hit_differences)
                    else np.quantile(hit_differences, [.025, .975]).tolist(),
                    paired_return_rate_draws=int(supported.sum()),
                    post_return_goal_difference_per_match=float(np.mean(a[:, 2] - b[:, 2])),
                    paired_post_return_goal_ci95=np.quantile(goals_difference, [.025, .975]).tolist())

    return dict(adaptive_versus_fixed=comparison('adaptive_release', 'fixed_release'),
                fixed_versus_ball_masked=comparison('fixed_release', 'ball_masked'))


def evaluate(checkpoint, *, seed=TEST_SEED, count=COURSE_COUNT):
    checkpoint = Path(checkpoint)
    source = Path(__file__).resolve()
    experimental_files = ('arm_plays_pong.py', 'arm_pong_timing.py', 'pong_button_panel.py',
                          'button_pong_skill.py', 'planar_position_arm.py', 'camera_motor_learner.py',
                          'camera_reaching_arena.py', 'pong_score_placement.py', 'return_metrics.py')
    files = (source, *(source.with_name(name) for name in experimental_files),
             source.parents[2] / 'src/fly_connectome/pong.py', checkpoint)
    hashes = {file.name: sha256(file.read_bytes()).hexdigest() for file in files}
    schedule = cases(seed, count)
    rows = []
    for index, case in enumerate(schedule, 1):
        rows.append(_campaign(case, checkpoint))
        print(f'Original gearing {index}/{len(schedule)}', flush=True)
    if [row['matches']['fixed_release']['initial_down_button'] for row in rows] != [i % 2 for i in range(count)]:
        raise AssertionError('wired state did not match the balanced schedule')
    if hashes != {file.name: sha256(file.read_bytes()).hexdigest() for file in files}:
        raise AssertionError('experiment code or checkpoint changed during execution')
    return dict(seed=seed, matches_per_condition=count, first_to=FIRST_TO, max_ticks=MAX_TICKS,
                opponent_speed=OPPONENT_SPEED, bootstrap_draws=5000, bootstrap_unit='matched match',
                gearing=SOURCE_GEAR, joint_speed_radians_per_second=MAX_JOINT_SPEED,
                arm_practice_ticks=PRACTICE_TICKS, checkpoint=checkpoint.name,
                source_sha256=hashes, summary=_summarize(rows), paired=_paired_effects(rows), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=TEST_SEED)
    parser.add_argument('--count', type=int, default=COURSE_COUNT)
    arguments = parser.parse_args()
    result = evaluate(arguments.checkpoint, seed=arguments.seed, count=arguments.count)
    with arguments.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(summary=result['summary'], paired=result['paired']), indent=2, allow_nan=False))
