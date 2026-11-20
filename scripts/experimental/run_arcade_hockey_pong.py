"""Run the paired hockey-practice → changed-Pong experiment and save per-life receipts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import torch

from air_hockey import AirHockeyCamera
from air_hockey_learning import GoalMemory, HockeyPlayer
from arcade_hockey_pong import AdaptiveWall, COEFFICIENTS, PongFlightActor
from arcade_transfer_session import CHECKPOINT
from arcade_wall_learn import WallMemory
from fly_connectome.control.environment import PongCamera
from fly_connectome.pong import Pong
from return_metrics import ReturnCounter


CONDITIONS = {
    'pong_only': ('none', 'live'),
    'extra_pong': ('pong', 'live'),
    'varied_pong': ('varied_pong', 'live'),
    'after_hockey': ('hockey', 'live'),
    'hockey_fixed_after_surprise': ('hockey', 'fixed_on_surprise'),
    'frozen_pong': ('none', 'frozen'),
}
SOURCE_COEFFICIENTS = (.46, .59, .72, .83)
TARGET_RANGES = ((.32, .43), (.50, .57), (.63, .70), (.76, .80))
EARLY_FRAMES = (100, 200, 400, 800, 1600)
FORECAST_FRAME = 15
ROOT = Path(__file__).resolve().parents[2]


def save(path, contents):
    path.write_text(json.dumps(contents, indent=2, allow_nan=False) + '\n')


def source_fingerprints():
    files = ('scripts/experimental/run_arcade_hockey_pong.py', 'scripts/experimental/arcade_hockey_pong.py',
             'scripts/experimental/arcade_wall_learn.py', 'scripts/experimental/air_hockey.py',
             'scripts/experimental/air_hockey_learning.py', 'src/fly_connectome/pong.py')
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def registered_plan(seed, streams, source_budget, lives):
    """Generate independent source streams and paired target courses without outcomes."""
    rng = np.random.default_rng(seed)
    source = [dict(seed=int(rng.integers(2**31 - 1)), anchor_seed=int(rng.integers(2**31 - 1)),
                   budget=source_budget, restitution=value)
              for value in SOURCE_COEFFICIENTS]
    target = []
    for index in range(streams):
        value = float(rng.uniform(*TARGET_RANGES[index % len(TARGET_RANGES)]))
        target.append(dict(stream=index, restitution=value, cases=[
            dict(seed=int(rng.integers(2**31 - 1)), bank=life < lives - 2) for life in range(lives)
        ]))
    return dict(seed=seed, source=source, target=target,
                conditions=CONDITIONS, error_threshold=.07, checkpoints_frames=EARLY_FRAMES,
                forecast_frame=FORECAST_FRAME, initial_checkpoint=str(CHECKPOINT.name),
                checkpoint_sha256=sha256(CHECKPOINT.read_bytes()).hexdigest(), sources=source_fingerprints())


def native_camera(seed, restitution, bank, max_frames, *, shallow=False):
    """Create evaluator inputs, including an optional banked opponent return."""
    camera = PongCamera(seed, opponent_speed=.55, max_episode_ticks=round(max_frames * 2.4))
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(.74, .99) if shallow else rng.uniform(.86, 1.10)) if bank else .55
    camera.game = Pong([seed], replace(camera.game.config, wall_restitution=restitution, ball_speed=speed))
    if bank:
        x = float(rng.uniform(.72, .88))
        angle = float(rng.uniform(.33, .57) if shallow else rng.uniform(.69, .99))
        if shallow:
            y = .0125 + (x - 4 / 64) * math.tan(angle) * float(rng.uniform(.38, .60))
        else:
            y = float(rng.uniform(.27, .38))
        vx, vy = -speed * math.cos(angle), -speed * math.sin(angle)
        if rng.integers(2):
            y, vy = 1 - y, -vy
        camera.game.ball[0] = torch.tensor([x, y])
        camera.game.ball_velocity[0] = torch.tensor([vx, vy])
    return camera


def _play_pong(timing, memory, seed, *, restitution, bank, limit, shallow=False):
    camera = native_camera(seed, restitution, bank, limit, shallow=shallow)
    game = camera.game
    counter = ReturnCounter()
    step = game.step
    game.step = lambda drive, **kwargs: counter.step(game, step, drive, **kwargs)
    frame = game.render(64, 64)[0]
    opening_pixels = sha256(frame.numpy().tobytes()).hexdigest()
    actor = PongFlightActor(timing, memory)
    actor.start(frame)
    commands = bytearray()
    bounces = []
    hints = []
    forecast = None
    initial = memory.coefficient
    elapsed = 0.
    for index in range(limit):
        command = actor.choose()
        if index == FORECAST_FRAME and bank:
            forecast = actor.target
        commands.append(command)
        transition = camera.step(command)
        bounce = actor.observe(transition.frame, transition.elapsed_seconds)
        elapsed += transition.elapsed_seconds
        if bounce is not None:
            bounces.append(dict(asdict(bounce), observed_at=index + 1, estimate=memory.coefficient))
        hint = actor.flight.last_hint
        if hint is not None:
            hints.append(dict(asdict(hint), observed_at=index + 1, estimate=memory.coefficient,
                              change_probability=memory.change_probability))
        if transition.terminated or transition.truncated:
            break
    annotation = counter.summary(game, truncated=transition.truncated, terminated=transition.terminated,
                                 reward=transition.reward, seconds=elapsed)
    incoming = annotation['first_incoming']
    prediction_error = abs(forecast - incoming['pre_ball'][1]) if forecast is not None and incoming else None
    return dict(seed=seed, restitution=restitution, bank=bank, frames=len(commands),
                first_frame_sha256=opening_pixels, actions_sha256=sha256(commands).hexdigest(),
                initial_requests_sha256=sha256(commands[:FORECAST_FRAME + 1]).hexdigest(),
                initial_estimate=initial, final_estimate=memory.coefficient, bounces=bounces, hints=hints,
                forecast=forecast, forecast_error=prediction_error, **annotation)


def _practice_pong(timing, seed, budget, *, bank=True, restitution=1.):
    rng = np.random.default_rng(seed)
    wall = WallMemory()
    rows = []
    consumed = 0
    while consumed < budget:
        row = _play_pong(timing, wall, int(rng.integers(2**31 - 1)), restitution=restitution, bank=bank,
                         limit=min(300, budget - consumed))
        rows.append(row)
        consumed += row['frames']
    return wall, dict(wall=wall.to_state(), frames=consumed, lives=rows)


def _practice_hockey(seed, budget, restitution):
    rng = np.random.default_rng(seed)
    wall = WallMemory()
    goals = GoalMemory()
    rows = []
    consumed = 0
    while consumed < budget:
        case = int(rng.integers(2**31 - 1))
        arena = AirHockeyCamera(case, max_frames=min(300, budget - consumed), wall_restitution=restitution,
                                serve_style='bank', opponent='guard')
        actor = HockeyPlayer(wall, goals, seed=case, learn_wall=True, learn_goals=True, explore=.25)
        frame = arena.frame()
        opening_pixels = sha256(frame.tobytes()).hexdigest()
        actor.observe(frame)
        commands = bytearray()
        while True:
            command = actor.action()
            commands.append(command)
            frame, reward, terminated, truncated = arena.step(command)
            actor.observe(frame, reward, terminated, truncated)
            if terminated or truncated:
                break
        consumed += len(commands)
        rows.append(dict(seed=case, frames=len(commands), reward=reward, terminated=terminated, truncated=truncated,
                         first_frame_sha256=opening_pixels, actions_sha256=sha256(commands).hexdigest(),
                         bounces=actor.bounces, player_contacts=sum(e['kind'] == 'contact' and e['side'] == 'player'
                                                                  for e in arena.events)))
    return wall, dict(wall=wall.to_state(), goals=goals.to_state(), frames=consumed, lives=rows)


def _target_course(timing, anchor, practice, mode, course, threshold):
    model = AdaptiveWall(anchor, practice, mode)
    coefficient = course['restitution']
    rows = [_play_pong(timing, model, **case, restitution=coefficient, limit=240, shallow=True)
            for case in course['cases']]
    events = []
    frames = 0
    for row in rows:
        visible = [dict(event, kind=kind) for key, kind in (('bounces', 'confirmed'), ('hints', 'hint'))
                   for event in row[key]]
        events.extend(dict(event, course_frame=frames + event['observed_at'])
                      for event in sorted(visible, key=lambda event: event['observed_at']))
        frames += row['frames']
    formal = []
    for index, event in enumerate(events):
        if all(abs(item['estimate'] - coefficient) <= threshold for item in events[index:]):
            count = sum(e['kind'] == 'confirmed' for e in events[:index + 1])
            formal.append(dict(confirmed_bounces=count, cameras=event['course_frame'], kind=event['kind']))
            break
    checkpoints = []
    for budget in EARLY_FRAMES:
        if budget <= frames:
            prior = [e['estimate'] for e in events if e['course_frame'] <= budget]
            estimate = prior[-1] if prior else model.anchor
            checkpoints.append(dict(cameras=budget, estimate=estimate, error=abs(estimate - coefficient)))
    return dict(lives=rows, events=events, frames=frames, recovered=formal[0] if formal else None,
                first_confirmed_frame=next((e['course_frame'] for e in events if e['kind'] == 'confirmed'), None),
                error_after_bounces=[abs(e['estimate'] - coefficient) for e in events if e['kind'] == 'confirmed'],
                error_after_first_hint=next((abs(e['estimate'] - coefficient) for e in events if e['kind'] == 'hint'), None),
                final_error=abs(model.coefficient - coefficient), checkpoints=checkpoints,
                initial_error=abs(model.anchor - coefficient), alternative_prior=model.prior.tolist(),
                change_rate=model.change_rate)


def _verified_opening_signature(row):
    incoming = row['first_incoming']
    return (row['first_frame_sha256'], None if incoming is None else
            {field: value for field, value in incoming.items() if field != 'success'})


def _first_k_mean(target, count):
    items = target['error_after_bounces'][:count]
    return float(np.mean(items)) if len(items) == count else None


def _metrics(course):
    recovery = course['recovered']
    stages = (course['lives'][:3], course['lives'][3:-2], course['lives'][-2:])
    metrics = dict(initial_error=course['initial_error'], final_error=course['final_error'],
                   first_hint_error=course['error_after_first_hint'],
                   first_bounce_error=_first_k_mean(course, 1), first_three_bounce_mae=_first_k_mean(course, 3),
                   first_six_bounce_mae=_first_k_mean(course, 6), recovered=bool(recovery),
                   recovered_before_confirmed=bool(recovery and recovery['kind'] == 'hint'
                       and course['first_confirmed_frame'] is not None
                       and recovery['cameras'] < course['first_confirmed_frame']),
                   recovery_cameras=800 if recovery is None else min(800, recovery['cameras']))
    for name, stage in zip(('early_bank', 'later_bank', 'natural'), stages, strict=True):
        errors = [r['forecast_error'] for r in stage if r['forecast_error'] is not None]
        metrics[name] = dict(returns=sum(r['own_returns'] for r in stage),
                             opportunities=sum(r['own_opportunities'] for r in stage),
                             scores=sum(r['reward'] for r in stage),
                             first_returns=sum(bool(r['first_incoming'] and r['first_incoming']['success']) for r in stage),
                             first_opportunities=sum(r['first_incoming'] is not None for r in stage),
                             forecast_mae=float(np.mean(errors)) if errors else None)
    return metrics


def _evaluate_course(payload):
    timing, anchor, practice, plan, course = payload
    arms = {name: _target_course(timing, anchor, practice[learned], mode, course, plan['error_threshold'])
            for name, (learned, mode) in plan['conditions'].items()}
    reference = [_verified_opening_signature(row) for row in arms['pong_only']['lives']]
    if any([_verified_opening_signature(row) for row in trial['lives']] != reference for trial in arms.values()):
        raise AssertionError('paired arms had different first incoming geometry before an own return')
    if len({trial['lives'][0]['initial_requests_sha256'] for trial in arms.values()}) != 1:
        raise AssertionError('the pre-bounce initial commands differ')
    return dict(stream=course['stream'], restitution=course['restitution'], arms=arms,
                metrics={name: _metrics(trial) for name, trial in arms.items()})


def run(plan, destination, workers):
    torch.set_num_threads(1)
    if sha256(CHECKPOINT.read_bytes()).hexdigest() != plan['checkpoint_sha256']:
        raise ValueError('the expected trained Pong checkpoint has changed')
    if plan.get('sources') != source_fingerprints():
        raise ValueError('experiment sources differ from the registration')
    timing = json.loads(CHECKPOINT.read_text())['timing']
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / 'plan.json', plan)
    started = time.monotonic()
    anchor = WallMemory()
    practice = dict(none=[], pong=[], varied_pong=[], hockey=[])
    receipts = dict(anchor=[], pong=[], varied_pong=[], hockey=[])
    for row in plan['source']:
        pong = _practice_pong(timing, row['seed'], row['budget'])
        varied = _practice_pong(timing, row['seed'], row['budget'], restitution=row['restitution'])
        hockey = _practice_hockey(row['seed'], row['budget'], row['restitution'])
        baseline, report = _practice_pong(timing, row['anchor_seed'], row['budget'])
        anchor.count += baseline.count
        anchor.precision += baseline.precision
        anchor.weighted_total += baseline.weighted_total
        receipts['anchor'].append(report)
        for name in practice:
            practice[name].append(baseline)
        for name, (memory, summary) in (('pong', pong), ('varied_pong', varied), ('hockey', hockey)):
            practice[name].append(memory)
            receipts[name].append(summary)
    receipts['common_pong_memory'] = anchor.to_state()
    receipts['pretrained_pong_timing'] = timing
    save(destination / 'practice.json', receipts)
    print('source wall response estimates', anchor.coefficient,
          {name: [(round(m.coefficient, 3), m.count) for m in memories] for name, memories in practice.items()},
          flush=True)
    all_courses = []
    work = ((timing, anchor, practice, plan, course) for course in plan['target'])
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for result in pool.map(_evaluate_course, work):
            all_courses.append(result)
            save(destination / 'streams.json', all_courses)
            print(result['stream'], round(result['restitution'], 3), {name: (row['recovered'], round(row['final_error'], 3))
                for name, row in result['arms'].items()}, flush=True)
    write_summary(destination, all_courses, started)


def write_summary(destination, courses, started):
    metrics = {arm: [row['metrics'][arm] for row in courses] for arm in CONDITIONS}
    summary = {}
    for name, rows in metrics.items():
        values = {}
        for key in ('initial_error', 'first_hint_error', 'first_bounce_error', 'first_three_bounce_mae',
                    'first_six_bounce_mae', 'final_error', 'recovered', 'recovered_before_confirmed', 'recovery_cameras'):
            observed = [row[key] for row in rows if row[key] is not None]
            values[key] = float(np.mean(observed)) if observed else None
        for stage in ('early_bank', 'later_bank', 'natural'):
            keys = ('returns', 'opportunities', 'first_returns', 'first_opportunities', 'scores')
            values[stage] = {key: sum(row[stage][key] for row in rows) for key in keys}
            mae = [row[stage]['forecast_mae'] for row in rows if row[stage]['forecast_mae'] is not None]
            values[stage]['forecast_mae'] = float(np.mean(mae)) if mae else None
        timing = {budget: [item['error'] for row in courses for item in row['arms'][name]['checkpoints']
                           if item['cameras'] == budget] for budget in EARLY_FRAMES}
        values['camera_error'] = {key: dict(streams=len(errors), mean=float(np.mean(errors)))
                                  for key, errors in timing.items() if errors}
        summary[name] = values

    rng = np.random.default_rng(6419023)

    def compare(left, right):
        differences = np.asarray([a - b for a, b in zip(left, right, strict=True) if a is not None and b is not None])
        if not len(differences):
            return None
        draws = rng.integers(len(differences), size=(10000, len(differences)))
        interval = np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()
        return dict(eligible=len(differences), difference=float(differences.mean()), ci95=interval,
                    lower=int((differences < 0).sum()), tied=int((differences == 0).sum()))

    paired = {}
    for control in ('pong_only', 'extra_pong', 'varied_pong', 'hockey_fixed_after_surprise', 'frozen_pong'):
        paired[control] = {}
        for key in ('first_hint_error', 'first_bounce_error', 'first_three_bounce_mae', 'first_six_bounce_mae',
                    'final_error', 'recovery_cameras'):
            paired[control][key] = compare([row[key] for row in metrics['after_hockey']],
                                           [row[key] for row in metrics[control]])
        for stage in ('early_bank', 'later_bank', 'natural'):
            paired[control][stage] = {
                key: compare([row[stage][key] for row in metrics['after_hockey']],
                             [row[stage][key] for row in metrics[control]])
                for key in ('forecast_mae', 'returns', 'first_returns', 'scores')
            }
    early = [row['early_bank'] for row in metrics['after_hockey']]
    save(destination / 'summary.json', dict(paired_differences_hockey_minus_control=paired, arms=summary,
         streams=len(courses), mean_after_hockey_early_bank_opportunities=float(np.mean([r['opportunities'] for r in early])),
         coefficient_grid=COEFFICIENTS.tolist(), elapsed_wall_seconds=time.monotonic() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('register', 'run'))
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--streams', type=int, default=12)
    parser.add_argument('--source-budget', type=int, default=650)
    parser.add_argument('--lives', type=int, default=8)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.operation == 'register':
        if args.plan.exists() or args.seed is None or args.streams < 1 or args.lives < 6 or args.source_budget < 20:
            parser.error('a new plan path, seed and positive source and target budgets are required')
        save(args.plan, registered_plan(args.seed, args.streams, args.source_budget, args.lives))
    elif args.output is not None and args.workers > 0:
        run(json.loads(args.plan.read_text()), args.output, args.workers)
    else:
        parser.error('--output is required for run')
