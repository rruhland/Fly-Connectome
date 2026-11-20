"""Train from played camera rounds and compare frozen agents on matched hockey starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from air_hockey import AirHockeyCamera, HZ, PUCK_RADIUS
from air_hockey_learning import BOUNDS, GoalMemory, HockeyPlayer
from arcade_wall_learn import WallFlight, WallMemory, project_crossing


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'artifacts/m1b/2026-11-19-pong-after-mixed-course.json'
COLD_COEFFICIENT = .5
TRAIN_FRAMES = 420
TEST_FRAMES = 540
SCHEMA = 'camera-air-hockey-qualification-v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    target = Path(path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(target)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def donor_pong(seed, *, checkpoint):
    """Attend to the ball in real Pong frames while the saved controller plays."""
    import torch

    from pong_carried_shot_value import ShotMemory, build_carried_shot_controller
    from pong_score_placement import ScorePlacement
    from pong_transfer_arena import play

    torch.set_num_threads(1)
    controller = build_carried_shot_controller(
        seed, checkpoint['motor'], checkpoint['timing'],
        ScorePlacement(**checkpoint['score'], updates_enabled=False),
        memory=ShotMemory(**checkpoint['shot_memory']))

    class FromCamera:
        def __init__(self):
            self.flight = WallFlight(WallMemory(default=COLD_COEFFICIENT), (-.4875, .4875),
                                     position_noise=.006)
            self.bounces = []

        def choose(self, entities, baseline):
            return controller.choose(entities, baseline)

        def observe(self, **transition):
            # The existing binary-camera encoder expresses points as (row, column)
            # centered on the frame. Outside the two visible rails the object is the ball.
            candidates = [(column, row) for entity in transition['public_entities']
                          for row, column in [entity['normalized_position']]
                          if entity['observed'] and -.422 < column < .422]
            point = candidates[0] if len(candidates) == 1 else None
            bounce = self.flight.observe(point, elapsed=transition['elapsed_seconds'])
            if bounce is not None:
                self.bounces.append(asdict(bounce))
            controller.observe(**transition)

    stream = FromCamera()
    result = play(stream, seed, opponent_speed=.55, duration=8.)
    return {**result, 'measured_bounces': stream.bounces}


def fit_pong_wall(seeds, checkpoint, workers):
    wall = WallMemory(default=COLD_COEFFICIENT)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(donor_pong, checkpoint=checkpoint), seeds))
    for row in rows:
        for event in row['measured_bounces']:
            wall.observe(event['coefficient'], event['uncertainty'])
    return wall, rows


def load_pong_wall(directory, registration):
    prior = read(directory / 'registration.json')
    cache = read(directory / 'pong-donor.json')
    prefixes = ('artifacts/m1b/', 'scripts/experimental/arcade_wall_learn.py',
                'scripts/experimental/pong_', 'src/fly_connectome/')
    files = [name for name in registration['hashes'] if name.startswith(prefixes)]
    if (registration['seeds']['pong'] != [round_['seed'] for round_ in cache['rounds']]
            or any(registration['hashes'][name] != prior['hashes'].get(name) for name in files)):
        raise ValueError('cached Pong evidence has different starting points or source files')
    wall = WallMemory(default=COLD_COEFFICIENT)
    for row in cache['rounds']:
        for event in row['measured_bounces']:
            wall.observe(event['coefficient'], event['uncertainty'])
    if wall.to_state() != cache['wall']:
        raise ValueError('cached Pong evidence does not replay its retained state')
    return wall, cache['rounds']


def run_round(case, wall, goals, *, practice=False, passive=False, record=False, fixed_center=False,
              max_frames=TEST_FRAMES):
    """Run a single round, with physics and camera-agent state kept separate."""
    camera = AirHockeyCamera(case['seed'], max_frames=max_frames, opponent=case['opponent'],
                            serve_style=case['style'], wall_restitution=case.get('wall_restitution', 1.))
    agent = HockeyPlayer(wall, goals, seed=case['seed'], learn_goals=practice,
                         explore=case.get('exploration', .65) if practice else 0., fixed_center=fixed_center)
    start = camera.frame()
    agent.observe(start)
    initial_sha = hashlib.sha256(start.tobytes()).hexdigest()
    actions = bytearray()
    images = [start] if record else None
    first_estimate = None
    while not (camera.terminated or camera.truncated):
        # This forecast-only diagnostic compares identical pre-contact image history.
        if camera.frames == 9 and case['style'] != 'neutral' and not camera.events:
            first_estimate = project_crossing(agent.image.puck, agent.flight.velocity, PUCK_RADIUS,
                                              BOUNDS, wall.coefficient)
        request = 0 if passive else agent.action()
        actions.append(request)
        transition = camera.step(request)
        agent.observe(*transition)
        if images is not None:
            images.append(transition[0])
    impacts = [event for event in camera.events if event['kind'] == 'contact']
    stops = [event for event in camera.events if event['kind'] in ('contact', 'goal', 'end_rail')]
    first = stops[0] if stops else None
    opening_return = bool(first and first.get('side') == 'player' and first['incoming'][0] < -.1
                          and first['velocity'][0] > .1)
    hits = [event for event in impacts if event.get('side') == 'player' and event['velocity'][0] > .1]
    # No-interference crossing from the evaluator's initial state.
    actual_crossing = None
    if first_estimate is not None:
        audit = AirHockeyCamera(case['seed'], max_frames=1, opponent=case['opponent'],
                               serve_style=case['style'], wall_restitution=case.get('wall_restitution', 1.))
        actual_crossing = project_crossing(audit.puck, audit.velocity, PUCK_RADIUS,
                                            BOUNDS, audit.wall_restitution)
    row = {**case, 'initial_sha256': initial_sha, 'action_sha256': hashlib.sha256(actions).hexdigest(),
           'state_sha256': state_hash([camera.puck, camera.velocity, camera.player, camera.rival, camera.events]),
           'seconds': len(actions) / HZ, 'frames': len(actions), 'reward': camera.reward,
           'opening_return': opening_return, 'contacts': len(hits), 'visual_bounces': agent.bounces,
           'last_touch_goal': bool(camera.reward > 0 and impacts and impacts[-1].get('side') == 'player'),
           'goal_after_own_hit': bool(camera.reward > 0 and hits), 'goals_conceded': int(camera.reward < 0),
           'forecast': first_estimate, 'actual_crossing': actual_crossing,
           'visual_shots': [asdict(shot) for shot in agent.shots], 'credited': agent.credited,
           'events': camera.events}
    return row, images


def finish_round(job):
    label, case, wall_state, goal_state = job
    wall = WallMemory.from_state(wall_state)
    goals = GoalMemory.from_state(goal_state)
    row, _ = run_round(case, wall, goals, passive=label == 'passive',
                       fixed_center=label in ('pong_center', 'broad_center', 'elastic_center'))
    return label, row


def summarize(rows):
    forecasts = [abs(row['forecast'] - row['actual_crossing']) for row in rows if row['forecast'] is not None]
    return {'rounds': len(rows), 'goals': sum(row['reward'] > 0 for row in rows),
            'last_touch_goals': sum(row['last_touch_goal'] for row in rows),
            'goals_after_own_hit': sum(row['goal_after_own_hit'] for row in rows),
            'conceded': sum(row['goals_conceded'] for row in rows),
            'draws': sum(row['reward'] == 0 for row in rows),
            'first_saves': sum(row['opening_return'] for row in rows),
            'rounds_with_player_contact': sum(row['contacts'] > 0 for row in rows),
            'player_contacts': sum(row['contacts'] for row in rows),
            'seconds': sum(row['seconds'] for row in rows),
            'rounds_with_score_assignment': sum(row['credited'] is not None for row in rows),
            'forecast_mean_error': float(np.mean(forecasts)) if forecasts else None,
            'forecast_count': len(forecasts)}


def paired(left, right, metric):
    if [r['seed'] for r in left] != [r['seed'] for r in right]:
        raise ValueError('paired qualification requires identical starts')
    differences = np.asarray([float(a[metric]) - float(b[metric]) for a, b in zip(left, right, strict=True)])
    rng = np.random.default_rng(1367606312)
    mass = []
    for _ in range(20):
        draw = rng.integers(len(differences), size=(500, len(differences)))
        mass.extend(differences[draw].mean(axis=1))
    return {'mean': float(differences.mean()), 'interval_95': np.quantile(mass, [.025, .975]).tolist(),
            'better': int((differences > 0).sum()), 'worse': int((differences < 0).sum())}


def schedule(seed, practice_count, per_test, donor_count, opponent):
    rng = random.Random(seed)
    used = set()

    def take():
        while True:
            value = rng.randrange(1, 2**31 - 1)
            if value not in used:
                used.add(value)
                return value

    donor = [take() for _ in range(donor_count)]
    styles = ('bank', 'direct', 'neutral', 'mixed')
    practice = [{'seed': take(), 'style': styles[index % len(styles)], 'opponent': opponent,
                 'exploration': .8 if index < practice_count / 2 else .45}
                for index in range(practice_count)]
    heldout = [{'seed': take(), 'style': style, 'opponent': opponent}
               for style in ('bank', 'fast-bank', 'direct', 'neutral') for _ in range(per_test)]
    return {'pong': donor, 'hockey_practice': practice, 'hockey_heldout': heldout}


def run(args):
    paths = [Path(__file__), Path(__file__).with_name('air_hockey.py'),
             Path(__file__).with_name('air_hockey_learning.py'), Path(__file__).with_name('arcade_wall_learn.py'),
             ROOT / 'scripts/experimental/pong_carried_shot_value.py',
             ROOT / 'scripts/experimental/pong_transfer_arena.py',
             ROOT / 'src/fly_connectome/pong.py', ROOT / 'src/fly_connectome/control/environment.py']
    paths = [*paths, CHECKPOINT]
    args.output.mkdir(parents=True, exist_ok=True)
    registration = {'schema': SCHEMA, 'seeds': schedule(args.seed, args.practice, args.test, args.pong, args.opponent),
                    'training_frames': TRAIN_FRAMES, 'test_frames': TEST_FRAMES,
                    'hashes': {str(path.relative_to(ROOT)): sha(path) for path in paths},
                    'predeclared': {'hockey_practice_endpoint': 'last_touch_goal',
                                    'pong_transfer_game_endpoint': 'opening_return on bank and fast-bank starts',
                                    'unchanged_direct_control': 'opening_return on direct starts',
                                    'mechanism_endpoint': 'absolute forecast error'}}
    if args.reuse_pong is not None:
        registration['cached_pong_sha256'] = sha(args.reuse_pong / 'pong-donor.json')
    write(args.output / 'registration.json', registration)
    selected = registration['seeds']
    if args.reuse_pong is None:
        wall, pong_rows = fit_pong_wall(selected['pong'], read(CHECKPOINT), args.workers)
    else:
        wall, pong_rows = load_pong_wall(args.reuse_pong, registration)
    if not wall.count:
        raise RuntimeError('no Pong wall evidence was observed')
    write(args.output / 'pong-donor.json', {'wall': wall.to_state(), 'rounds': pong_rows})
    print('Pong wall', wall.to_state(), 'estimate', wall.coefficient, flush=True)
    goals = GoalMemory()
    practice_rows = []
    for index, case in enumerate(selected['hockey_practice']):
        row, _ = run_round(case, wall, goals, max_frames=TRAIN_FRAMES, practice=True)
        practice_rows.append(row)
        if (index + 1) % 40 == 0:
            print('Hockey practice', index + 1, summarize(practice_rows[-40:]), flush=True)
    write(args.output / 'hockey-practice.json', {'goal': goals.to_state(), 'rounds': practice_rows})
    state = {'schema': SCHEMA, 'wall': wall.to_state(), 'goal': goals.to_state(),
             'provenance': {'pong_file_sha256': sha(args.output / 'pong-donor.json'),
                            'practice_file_sha256': sha(args.output / 'hockey-practice.json'),
                            'registration_sha256': sha(args.output / 'registration.json')}}
    write(args.output / 'checkpoint.json', state)
    blank = GoalMemory().to_state()
    arms = {'practiced': (state['wall'], state['goal']), 'pong_start': (state['wall'], blank),
            'pong_center': (state['wall'], blank),
            'broad_center': (WallMemory(default=COLD_COEFFICIENT).to_state(), blank),
            'elastic_center': (WallMemory(default=1.).to_state(), blank),
            'passive': (state['wall'], blank)}
    cases = selected['hockey_heldout']
    jobs = [(arm, case, *values) for arm, values in arms.items() for case in cases]
    results = {arm: [] for arm in arms}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for arm, row in pool.map(finish_round, jobs, chunksize=4):
            results[arm].append(row)
    if not all([r['initial_sha256'] for r in rows] == [r['initial_sha256'] for r in results['practiced']]
               for rows in results.values()):
        raise AssertionError('paired frames differ')
    subsets = {'all': results}
    for style in ('bank', 'fast-bank', 'direct', 'neutral'):
        subsets[style] = {arm: [row for row in rows if row['style'] == style] for arm, rows in results.items()}
    summaries = {subset: {arm: summarize(rows) for arm, rows in groups.items()}
                 for subset, groups in subsets.items()}
    comparisons = {
        'practice': paired(results['practiced'], results['pong_start'], 'last_touch_goal'),
        'practice_vs_center': paired(results['practiced'], results['pong_center'], 'last_touch_goal'),
        'early_pong': {subset: paired(groups['pong_center'], groups['broad_center'], 'opening_return')
                       for subset, groups in subsets.items() if subset != 'neutral'},
        'elastic_control': {subset: paired(groups['pong_center'], groups['elastic_center'], 'opening_return')
                            for subset, groups in subsets.items() if subset != 'neutral'},
    }
    write(args.output / 'heldout-rounds.json', {'results': results})
    write(args.output / 'results.json', {'registration_sha256': sha(args.output / 'registration.json'),
                                         'checkpoint_sha256': sha(args.output / 'checkpoint.json'),
                                         'heldout_sha256': sha(args.output / 'heldout-rounds.json'),
                                         'summaries': summaries, 'comparisons': comparisons})
    print(json.dumps({'summaries': summaries, 'comparisons': comparisons}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=956132300218)
    parser.add_argument('--practice', type=int, default=240)
    parser.add_argument('--test', type=int, default=64, help='held-out rounds per opening style')
    parser.add_argument('--pong', type=int, default=64)
    parser.add_argument('--opponent', choices=('guard', 'press', 'still', 'laneguard'), default='laneguard')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-pong', type=Path, help='reuse camera receipts from the same Pong starts')
    args = parser.parse_args()
    if min(args.practice, args.test, args.pong, args.workers) < 1:
        parser.error('positive practice, test, Pong and worker counts are required')
    run(args)


if __name__ == '__main__':
    main()
