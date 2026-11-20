"""Carry a visually acquired button polarity from played Pong to camera air hockey."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from air_hockey import ACTIONS, HZ, PIXELS_PER_UNIT, AirHockeyCamera
from air_hockey_experiment import sha, write
from air_hockey_learning import HockeyPlayer
from arcade_hockey_pong import PONG_CONTACT, PONG_WALLS, PongImage
from arcade_wall_learn import WallFlight, WallMemory, project_crossing


ROOT = Path(__file__).resolve().parents[2]
PONG_TICKS = 960
HOCKEY_FRAMES = 540
STYLES = ('fast-bank', 'direct', 'bank', 'mixed')
ARMS = ('carried', 'canonical_online', 'probe_online', 'crossed_online', 'crossed_frozen')
ENDPOINTS = ('first_contact', 'opening_return', 'goal_after_contact', 'reward')
FILES = (Path(__file__), Path(__file__).with_name('air_hockey.py'),
         Path(__file__).with_name('air_hockey_learning.py'),
         Path(__file__).with_name('arcade_hockey_pong.py'),
         Path(__file__).with_name('arcade_wall_learn.py'), ROOT / 'src/fly_connectome/pong.py',
         ROOT / 'src/fly_connectome/control/environment.py')


@dataclass
class VisualPolarity:
    """Associate a signed request with the sign of its measured displacement."""

    polarity: int | None = None
    observations: int = 0
    contradictions: int = 0

    def request(self, intent):
        return intent * (1 if self.polarity is None else self.polarity)

    def observe(self, request, before, after, resolution):
        displacement = after - before
        if not request or abs(displacement) < resolution:
            return
        inferred = request if displacement > 0 else -request
        self.observations += 1
        self.contradictions += self.polarity is not None and inferred != self.polarity
        self.polarity = inferred


def pong_actor(frame, polarity, *, initial_probe):
    image = PongImage.read(frame.numpy())
    flight = WallFlight(WallMemory(default=1.), PONG_WALLS, position_noise=.0048)
    flight.observe(image.ball, .02, learn=False)
    anchor = image.paddle
    position_history = deque([(0., anchor)], maxlen=6)
    elapsed = 0.
    hits = probes = 0
    requests = bytearray()
    actions_seen = None
    while True:
        acquiring = polarity.polarity is None
        if acquiring:
            requested = initial_probe
            probes += 1
        else:
            target = .5
            if flight.position is not None and flight.velocity is not None and flight.velocity[0] < -.10:
                crossing = project_crossing(flight.position, flight.velocity, PONG_CONTACT, PONG_WALLS, 1.)
                if crossing is not None:
                    target = min(.90, max(.10, crossing))
            start, start_y = position_history[0]
            drift = .18 * (image.paddle - start_y) / max(.02, elapsed - start)
            error = target - image.paddle - min(.09, max(-.09, drift))
            intent = int(error > .015) - int(error < -.015)
            requested = polarity.request(intent)
        requests.append(requested + 1)
        transition = yield requested
        image = PongImage.read(transition.frame.numpy())
        elapsed += transition.elapsed_seconds
        hits += transition.hits
        position_history.append((elapsed, image.paddle))
        flight.observe(image.ball, transition.elapsed_seconds, learn=False)
        if acquiring:
            polarity.observe(requested, anchor, image.paddle, 1.5 / 64)
            if polarity.polarity is not None:
                actions_seen = dict(anchor=anchor, measured=image.paddle, request=requested)
        if transition.terminated or transition.truncated:
            return dict(hits=hits, frames=len(requests), seconds=elapsed, probes=probes,
                        score=int(transition.reward), measurement=actions_seen,
                        request_sha256=hashlib.sha256(requests).hexdigest())


def play_pong(case, mapping, polarity, probe):
    from fly_connectome.control.environment import PongCamera

    camera = PongCamera(case, max_episode_ticks=PONG_TICKS, opponent_speed=.45)
    agent = pong_actor(camera.game.render(64, 64)[0], polarity, initial_probe=probe)
    requested = next(agent)
    while True:
        # The signed command is indexed independently of the actual actuator direction.
        physical = 1 - requested * mapping
        try:
            requested = agent.send(camera.step(physical))
        except StopIteration as result:
            return dict(seed=case, **result.value)


def play_hockey(case, device, motor, *, probe, frozen, probe_direction):
    camera = AirHockeyCamera(case['seed'], max_frames=HOCKEY_FRAMES,
                            serve_style=case['style'], opponent='guard')
    player = HockeyPlayer(WallMemory(default=1.), fixed_center=True)
    frame = camera.frame()
    initial_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    player.observe(frame)
    started = [asdict(axis) for axis in motor]
    first_correct = [0 if axis.polarity == physical else None
                     for axis, physical in zip(motor, device, strict=True)]
    first_request = [None, None]
    first_wrong = [None, None]
    opposite = [0, 0]
    exploratory = 0
    requests = bytearray()
    while not (camera.terminated or camera.truncated):
        intent = ACTIONS[player.action()]
        requested = tuple(axis.request(value) for axis, value in zip(motor, intent, strict=True))
        if probe and any(axis.polarity is None for axis in motor):
            requested = tuple(direction if axis.polarity is None else request
                              for direction, axis, request in zip(probe_direction, motor, requested, strict=True))
            exploratory += 1
        requests.append(ACTIONS.index(requested))
        executed = tuple(button * polarity for button, polarity in zip(requested, device, strict=True))
        for axis, (desired, actual) in enumerate(zip(intent, executed, strict=True)):
            if requested[axis] and first_request[axis] is None:
                first_request[axis] = camera.frames + 1
            if desired * actual < 0:
                if first_wrong[axis] is None:
                    first_wrong[axis] = camera.frames + 1
                opposite[axis] += 1
        before = player.image.player
        transition = camera.step(ACTIONS.index(executed))
        player.observe(*transition)
        after = player.image.player
        for index, axis in enumerate(motor):
            if index == 1 and frozen:
                continue
            axis.observe(requested[index], before[index], after[index], .55 / PIXELS_PER_UNIT)
            if first_correct[index] is None and axis.polarity == device[index]:
                first_correct[index] = camera.frames
    contacts = [event for event in camera.events if event['kind'] == 'contact']
    first = next((event for event in camera.events if event['kind'] in ('contact', 'goal', 'end_rail')), None)
    own = [event for event in contacts if event['side'] == 'player']
    first_contact = bool(first and first.get('side') == 'player')
    return dict(**case, initial_sha256=initial_hash, reward=camera.reward, frames=camera.frames,
                seconds=camera.frames / HZ, probes=exploratory, first_contact=first_contact,
                opening_return=bool(first_contact and first['incoming'][0] < -.10 and first['velocity'][0] > .10),
                contacts=len(own), goal_after_contact=bool(camera.reward > 0 and own),
                last_touch_goal=bool(camera.reward > 0 and contacts and contacts[-1]['side'] == 'player'),
                first_correct=first_correct, first_request=first_request, first_wrong=first_wrong,
                opposite=opposite, before=started, after=[asdict(axis) for axis in motor],
                request_sha256=hashlib.sha256(requests).hexdigest(), events=camera.events)


def schedule(seed, pairs, donor_points, target_points):
    if min(pairs, donor_points, target_points) < 1:
        raise ValueError('positive donor and target sample sizes required')
    rng = random.Random(seed)
    seeds = iter(rng.sample(range(1, 2**31 - 1), 2 * pairs * (donor_points + target_points)))
    blocks = []
    for i in range(pairs):
        direction = rng.choice((-1, 1))
        cases = []
        for side in range(2):
            source_motor = direction if side == 0 else -direction
            cases.append(dict(stream=2 * i + side, source_motor=source_motor,
                              horizontal_motor=rng.choice((-1, 1)),
                              pong_probe=rng.choice((-1, 1)),
                              hockey_probe=(rng.choice((-1, 1)), rng.choice((-1, 1))),
                              pong=[next(seeds) for _ in range(donor_points)],
                              hockey=[dict(seed=next(seeds), style=STYLES[(i + side + point) % len(STYLES)])
                                      for point in range(target_points)]))
        blocks.append(cases)
    return blocks


def play_pair(cases):
    import torch

    torch.set_num_threads(1)
    donors = []
    for case in cases:
        axis = VisualPolarity()
        rows = [play_pong(seed, case['source_motor'], axis, case['pong_probe']) for seed in case['pong']]
        donors.append(dict(learned=axis.polarity, pong=rows))
    result = []
    for case, donor, other in zip(cases, donors, reversed(donors), strict=True):
        rows = {}
        for arm in ARMS:
            prior = donor['learned'] if arm == 'carried' else other['learned'] if arm.startswith('crossed') else None
            motor = (VisualPolarity(), VisualPolarity(prior))
            device = (case['horizontal_motor'], case['source_motor'])
            rows[arm] = [play_hockey(game, device, motor, probe=arm == 'probe_online',
                                     frozen=arm == 'crossed_frozen', probe_direction=case['hockey_probe'])
                         for game in case['hockey']]
        if any([r['initial_sha256'] for r in rows[arm]] != [r['initial_sha256'] for r in rows['carried']]
               for arm in ARMS):
            raise RuntimeError('paired camera starts differ')
        result.append(dict(case=case, **donor, hockey=rows))
    return result


def totals(rows):
    return dict(points=len(rows), contacts=sum(r['contacts'] for r in rows),
                points_with_contact=sum(r['contacts'] > 0 for r in rows),
                first_contacts=sum(r['first_contact'] for r in rows), opening_returns=sum(r['opening_return'] for r in rows),
                goals=sum(r['reward'] > 0 for r in rows), goals_after_contact=sum(r['goal_after_contact'] for r in rows),
                last_touch_goals=sum(r['last_touch_goal'] for r in rows), conceded=sum(r['reward'] < 0 for r in rows),
                timeouts=sum(r['reward'] == 0 for r in rows), reward=sum(r['reward'] for r in rows),
                frames=sum(r['frames'] for r in rows), seconds=sum(r['frames'] for r in rows) / HZ,
                probes=sum(r['probes'] for r in rows),
                opposite_x=sum(r['opposite'][0] for r in rows), opposite_y=sum(r['opposite'][1] for r in rows))


def paired(blocks, arm, metric, subset):
    deltas = np.asarray([[np.mean([float(a[metric]) - float(b[metric]) for a, b in zip(
        subset(row['hockey']['carried']), subset(row['hockey'][arm]), strict=True)])
        for row in block] for block in blocks])
    by_block = deltas.mean(axis=1)
    rng = np.random.default_rng(684153771)
    trials = by_block[rng.integers(len(blocks), size=(10000, len(blocks)))].mean(axis=1)
    return dict(diff=float(deltas.mean()), interval_95=np.quantile(trials, [.025, .975]).tolist(),
                better=int((deltas > 0).sum()), worse=int((deltas < 0).sum()))


def summarize(blocks):
    streams = [row for block in blocks for row in block]
    pong = [r for row in streams for r in row['pong']]
    calibration = {}
    for arm in ARMS:
        openings = [row['hockey'][arm][0] for row in streams]
        calibration[arm] = dict(
            correct_from_start_x=sum(r['first_correct'][0] == 0 for r in openings),
            correct_from_start_y=sum(r['first_correct'][1] == 0 for r in openings),
            y_inferred_by_first_request=sum(r['first_correct'][1] is not None and r['first_request'][1] is not None
                                             and r['first_correct'][1] <= r['first_request'][1] for r in openings),
            x_inferred_by_first_request=sum(r['first_correct'][0] is not None and r['first_request'][0] is not None
                                             and r['first_correct'][0] <= r['first_request'][0] for r in openings),
            y_unresolved=sum(r['first_correct'][1] is None for r in openings),
        )
    return dict(source=dict(streams=len(streams), correct=sum(row['learned'] == row['case']['source_motor'] for row in streams),
                           points=len(pong), hits=sum(row['hits'] for row in pong),
                           wins=sum(row['score'] > 0 for row in pong), losses=sum(row['score'] < 0 for row in pong),
                           timeouts=sum(row['score'] == 0 for row in pong), frames=sum(row['frames'] for row in pong),
                           seconds=sum(row['seconds'] for row in pong), probes=sum(row['probes'] for row in pong)),
                calibration=calibration,
                **{name: dict(arms={arm: totals([r for stream in streams for r in subset(stream['hockey'][arm])])
                                     for arm in ARMS}, paired={arm: {endpoint: paired(blocks, arm, endpoint, subset)
                                                                      for endpoint in ENDPOINTS} for arm in ARMS[1:]})
                   for name, subset in (('opening_point', lambda rows: rows[:1]), ('all_points', lambda rows: rows))})


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    registration = dict(schema='pong-hockey-camera-motor-polarity-v1', seed=args.seed,
                        schedule=schedule(args.seed, args.pairs, args.donor_points, args.target_points),
                        files={str(path.relative_to(ROOT)): sha(path) for path in FILES},
                        horizons=dict(pong_physics_ticks=PONG_TICKS, hockey_camera_frames=HOCKEY_FRAMES),
                        primary='opening point first return vs both online cold controls, then goals after contact and net reward',
                        controls=list(ARMS[1:]), pairing='bootstrap complete two-stream donor swap blocks',
                        passed_state='one learned signed button polarity; horizontal is new in every arm',
                        source_supervision='requested button and rendered paddle displacement during live Pong',
                        target_supervision='requested button and rendered mallet displacement during live hockey',
                        scaffolding='camera marker decoder, geometry and analytic fixed-center pursuit in every arm',
                        source_probe='opening hold of a randomly chosen nonzero button until visible paddle displacement',
                        target_controls='preserve conventional defaults while learning on ordinary play; a blind simultaneous '
                                        'x/y probe; a swapped donor with and without vertical updates')
    write(args.output / 'registration.json', registration)


def run(args):
    registration = json.loads((args.output / 'registration.json').read_bytes())
    if any(sha(ROOT / path) != old for path, old in registration['files'].items()):
        raise RuntimeError('registered implementation has changed')
    blocks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, rows in enumerate(pool.map(play_pair, registration['schedule']), 1):
            blocks.append(rows)
            print(f'Completed {index}/{len(registration["schedule"])} independent donor pairs', flush=True)
    write(args.output / 'rounds.json', blocks)
    summary = summarize(blocks)
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('output', type=Path)
    plan.add_argument('--seed', type=int, required=True)
    plan.add_argument('--pairs', type=int, default=16)
    plan.add_argument('--donor-points', type=int, default=2)
    plan.add_argument('--target-points', type=int, default=4)
    plan.set_defaults(action=prepare)
    execute = sub.add_parser('run')
    execute.add_argument('output', type=Path)
    execute.add_argument('--workers', type=int, default=4)
    execute.set_defaults(action=run)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
