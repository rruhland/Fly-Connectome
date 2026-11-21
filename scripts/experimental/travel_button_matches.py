"""Match an axial-button Pong arm at 30 Hz with and without camera latency."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from arm_plays_pong import MatchCase, NATIVE_HZ, SCREEN_TICKS, _point_result, screen_frame
from button_pong_skill import DesiredPongPolicy
from fly_connectome.pong import Physics, Pong
from planar_position_arm import MAX_JOINT_SPEED
from return_metrics import ReturnCounter
from travel_button_control import CameraPipe, PRACTICE_SEED, TravelController, ZERO_COMMAND, practice_motor
from travel_button_panel import TravelButtonPanel


CASE_SEED = 15_483_821
PILOT_SEED = 72_035_261
BOOTSTRAP_SEED = 501_845
LATENCY = 6
FIRST_TO = 3
MAX_TICKS = 55 * NATIVE_HZ
OPPONENT = .35
MATCHES = 16


@dataclass(frozen=True)
class Condition:
    trained: bool
    latency_ticks: int
    compensate: bool
    connected: bool = True


CONDITIONS = {
    'trained_no_delay': Condition(True, 0, True),
    'fresh_no_delay': Condition(False, 0, True),
    'trained_compensated': Condition(True, LATENCY, True),
    'fresh_compensated': Condition(False, LATENCY, True),
    'trained_uncompensated': Condition(True, LATENCY, False),
    'trained_disconnected': Condition(True, LATENCY, True, False),
}


def schedule(count, seed):
    rng = np.random.default_rng(seed)
    games = rng.choice(60_000_000, count, replace=False) + 2_020_000_000
    scenes = rng.integers(2**32, size=count)
    learners = rng.integers(2**32, size=count)
    wiring = rng.permutation(np.resize((0, 3), count))
    return [MatchCase(*map(int, row)) for row in zip(games, scenes, wiring, learners, strict=True)]


def run_match(case, condition, checkpoint, practice, limit, *, replay=None, keep_tape=False):
    game = Pong([case.pong_seed], Physics(opponent_speed=OPPONENT))
    panel = TravelButtonPanel(case.panel_seed, wiring_seed=case.wiring_seed)
    if replay is None:
        policy = DesiredPongPolicy.load(checkpoint) if condition.trained else DesiredPongPolicy()
        controller = TravelController(policy, practice, case.controller_seed, compensate=condition.compensate)
        pipe = CameraPipe(condition.latency_ticks)
        controller.begin_point()
    else:
        next_change = iter(replay)
        change = next(next_change, None)
    actions_hash, dynamics_hash = sha256(), sha256()
    commands = {drive: torch.tensor([-float(drive)]) for drive in (-1, 0, 1)}
    reset = torch.ones(1, dtype=torch.bool)
    tape = []
    points = []
    counts = ReturnCounter()
    wins = losses = point_start = drive_ticks = moving_ticks = 0
    initial = tuple(game.ball[0].tolist() + game.ball_velocity[0].tolist())
    last_touch = None
    action = ZERO_COMMAND
    for tick in range(limit):
        old_action = action
        if replay is None:
            delivered = pipe.sample(tick, len(points), panel.frame, lambda: screen_frame(game))
            if delivered is not None:
                capture, image_point, arm_image, game_image = delivered
                action = controller.command(capture, tick, arm_image, game_image if image_point == len(points) else None)
        elif change is not None and tick == change[0]:
            action = tuple(change[1])
            change = next(next_change, None)
        if keep_tape and action != old_action:
            tape.append((tick, action))
        panel_drive = panel.step(action)
        drive = panel_drive if condition.connected else 0
        actions_hash.update(np.asarray(action, dtype='<f8').tobytes())
        dynamics_hash.update(np.asarray((*panel.tip, panel.height, *panel.depth), dtype='<f8').tobytes())
        drive_ticks += int(drive != 0)
        moving_ticks += int(any(action))
        old_right = counts.opponent_returns
        outcome = counts.step(game, game.step, commands[drive], auto_reset=False)
        hit, score = int(outcome.hits[0]), int(outcome.scores[0])
        dynamics_hash.update(bytes((drive + 1, hit, score + 1)))
        if hit:
            last_touch = 'own'
        elif counts.opponent_returns > old_right:
            last_touch = 'opponent'
        if score or tick + 1 == limit:
            points.append(_point_result(counts, point_start, tick + 1, score, last_touch, initial))
            if replay is None:
                controller.finish_point(score)
            wins += int(score == 1)
            losses += int(score == -1)
            if max(wins, losses) >= FIRST_TO or tick + 1 == limit:
                break
            game.reset(reset)
            initial = tuple(game.ball[0].tolist() + game.ball_velocity[0].tolist())
            last_touch = None
            counts = ReturnCounter()
            point_start = tick + 1
            if replay is None:
                controller.begin_point()
    if replay is not None and change is not None:
        raise ValueError('replay contains actions beyond the end of the game')
    presses = [x for x in panel.transitions if x['closed']]
    releases = [x for x in panel.transitions if not x['closed']]
    if (panel.steps != tick + 1 or
        any(not x['overlap'] or x['depth_m'] < panel.switches[x['button']].closes_at for x in presses) or
        any(x['depth_m'] > panel.switches[x['button']].opens_at for x in releases)):
        raise AssertionError('a switch transitioned without the required physical depth')
    incoming = next((point['incoming'] for point in points if point['incoming'] is not None), None)
    result = dict(case=asdict(case), ticks=tick + 1, seconds=(tick + 1)/NATIVE_HZ,
                  winner='own' if wins == FIRST_TO else 'opponent' if losses == FIRST_TO else None,
                  points_won=wins, points_lost=losses, points=points, first_incoming=incoming,
                  own_returns=sum(p['own_returns'] for p in points),
                  own_opportunities=sum(p['own_opportunities'] for p in points),
                  earned_goals=sum(p['scored_after_own_return'] for p in points),
                  wiring_down=panel.wiring.index(1), driven_ticks=drive_ticks, moving_ticks=moving_ticks,
                  presses=len(presses), releases=len(releases), depth_only_releases=panel.depth_only_releases,
                  shallow_overlap_ticks=panel.shallow_overlap_ticks, maximum_depth_m=panel.max_depth,
                  switch_specs=[asdict(x) for x in panel.switches],
                  first_press=next(iter(presses), None), first_release=next(iter(releases), None),
                  actions_sha256=actions_hash.hexdigest(), physics_sha256=dynamics_hash.hexdigest())
    if replay is None:
        if not (controller.arm_reads == pipe.delivered == controller.game_reads + controller.discarded_game_reads
                and pipe.captured == (tick // SCREEN_TICKS) + 1):
            raise AssertionError('camera cadence, delivery, and game observation counts disagree')
        events = [dict(event, observed_at_tick=event['tick'] + condition.latency_ticks)
                  for event in controller.binding.events]
        result.update(camera_captures=pipe.captured, camera_deliveries=pipe.delivered,
                      game_reads=controller.game_reads, discarded_game_reads=controller.discarded_game_reads,
                      received_frame_age_ticks=pipe.age_sum, motor_samples=controller.motor_samples,
                      binding_events=events, final_binding=controller.binding.down_button)
        if not condition.connected and (events or drive_ticks):
            raise AssertionError('disconnected switches supplied direction evidence or game input')
    if keep_tape:
        result.update(tape=tape, transitions=panel.transitions)
    return result


def _worker(job):
    i, name, case, condition, checkpoint, practice, limit = job
    row = run_match(case, condition, checkpoint, practice, limit, keep_tape=i == 0)
    if i == 0:
        tape = row['tape']
        verified = run_match(case, condition, checkpoint, practice, limit, replay=tape, keep_tape=True)
        if any(row[key] != value for key, value in verified.items()):
            raise AssertionError('physical action replay is different from the recorded match')
        row['replay_verified'] = True
    return i, name, row


def summarize(results):
    totals = {}
    for name, rows in results.items():
        quantities = ('points_won', 'points_lost', 'earned_goals', 'own_returns', 'own_opportunities',
                      'driven_ticks', 'presses', 'releases', 'depth_only_releases', 'shallow_overlap_ticks',
                      'camera_captures', 'camera_deliveries', 'game_reads', 'discarded_game_reads')
        totals[name] = {key: sum(row[key] for row in rows) for key in quantities}
        totals[name].update(wins=sum(row['winner'] == 'own' for row in rows),
                            losses=sum(row['winner'] == 'opponent' for row in rows),
                            capped=sum(row['winner'] is None for row in rows),
                            median_seconds=float(np.median([row['seconds'] for row in rows])),
                            correct_final_bindings=sum(row['final_binding'] == row['wiring_down'] for row in rows))
    count = len(next(iter(results.values())))
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(count, size=(8000, count))
    paired = {}
    for a, b in (('trained_compensated', 'fresh_compensated'), ('trained_no_delay', 'fresh_no_delay'),
                 ('trained_compensated', 'trained_no_delay'), ('trained_compensated', 'trained_uncompensated'),
                 ('trained_compensated', 'trained_disconnected')):
        if a not in results or b not in results:
            continue
        differences = {}
        for key, get in (('earned_goals', lambda r: r['earned_goals']),
                         ('win_fraction', lambda r: int(r['winner'] == 'own')),
                         ('score_margin', lambda r: r['points_won'] - r['points_lost'])):
            values = np.array([get(left) - get(right) for left, right in zip(results[a], results[b], strict=True)])
            differences[key] = dict(mean=float(values.mean()),
                                    ci95=np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist())
        paired[f'{a}_minus_{b}'] = differences
    return dict(totals=totals, paired=paired)


def evaluate(checkpoint, workers, *, pilot=False):
    seed, count, limit = (PILOT_SEED, 4, NATIVE_HZ * 22) if pilot else (CASE_SEED, MATCHES, MAX_TICKS)
    cases = schedule(count, seed)
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    inputs = [checkpoint, *(here / file for file in ('arm_plays_pong.py', 'arm_pong_rewire.py', 'button_pong_skill.py',
              'body_reaching_arena.py', 'camera_motor_learner.py', 'camera_reaching_arena.py', 'planar_position_arm.py', 'pong_button_panel.py',
              'pong_score_placement.py', 'return_metrics.py', 'travel_button_control.py', 'travel_button_matches.py',
              'travel_button_panel.py')), root / 'src/fly_connectome/pong.py']
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}
    practice = practice_motor()
    results = {key: [None] * count for key in CONDITIONS}
    with ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        futures = [pool.submit(_worker, (i, name, case, condition, checkpoint, practice, limit))
                   for i, case in enumerate(cases) for name, condition in CONDITIONS.items()]
        for n, future in enumerate(as_completed(futures), 1):
            i, name, result = future.result()
            results[name][i] = result
            if n % len(CONDITIONS) == 0:
                print(f'Completed {n} of {len(futures)} matches', flush=True)
    for i, rows in enumerate(zip(*results.values(), strict=True)):
        # Only the first point has identical play before any receiving-arm contact.
        signatures = [(row['case'], row['wiring_down'], row['switch_specs'], row['points'][0]['serve'],
                       {k: v for k, v in (row['points'][0]['incoming'] or {}).items() if k != 'success'}) for row in rows]
        if signatures.count(signatures[0]) != len(signatures) or rows[0]['case'] != asdict(cases[i]):
            raise AssertionError('matched games differ in their starting physical conditions')
    if hashes != {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}:
        raise AssertionError('experiment input changed during qualification')
    return dict(case_seed=seed, schedule=[asdict(case) for case in cases], first_to=FIRST_TO, max_ticks=limit,
                game_and_servo_hz=NATIVE_HZ, camera_hz=NATIVE_HZ//SCREEN_TICKS, opponent_speed=OPPONENT,
                joint_radians_per_second=MAX_JOINT_SPEED, conditions={k: asdict(v) for k, v in CONDITIONS.items()},
                arm_practice=dict(seed=PRACTICE_SEED, camera_samples=21, motion_observations=20,
                                  gram=practice.gram.tolist(), cross=practice.cross.tolist()),
                paired_bootstrap=dict(seed=BOOTSTRAP_SEED, draws=8000, unit='whole matched match'),
                inputs_sha256=hashes, summary=summarize(results), matches=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.workers, pilot=args.pilot)
    payload = (json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if args.output.suffix == '.gz':
        payload = gzip.compress(payload, mtime=0)
    with args.output.open('xb') as output:
        output.write(payload)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
