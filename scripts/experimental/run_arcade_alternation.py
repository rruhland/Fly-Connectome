"""Compare one unlabeled arcade stream with matched, uninterrupted game streams."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from arcade_alternation import AlternatingSession
from arcade_transfer_session import CHECKPOINT
from breakout_arena import BreakoutEpisode, breakout_schedule
from fly_connectome.control.environment import PongCamera
from return_metrics import ReturnCounter


COURSE_SEEDS = {'ordinary': 265049027, 'conflicting': 1327633107}
POLICY_SEED = 85217841
ARMS = ('shared', 'role_recall', 'isolated')
GAMES = ('pong', 'breakout')
NEUTRAL_MAPPING = (0, 1, 2)
BRICK_MAP_CONFLICT = (2, 1, 0)
MAX_TICKS = 2400


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(course, games_per_block):
    if games_per_block < 2 or games_per_block % 2:
        raise ValueError('an even number of games supports balanced starts in each block')
    pong_rng, breakout_rng = np.random.SeedSequence(COURSE_SEEDS[course]).spawn(2)
    rng = np.random.default_rng(pong_rng)
    breakout = iter(breakout_schedule(int(breakout_rng.generate_state(1)[0]), games_per_block * 3))
    used = set()
    cases = []
    for block, game in enumerate(GAMES * 3):
        for index in range(games_per_block):
            mapping = NEUTRAL_MAPPING
            delay = 0
            if game == 'pong':
                while True:
                    seed = int(rng.integers(1, 2147483647))
                    direction = ((seed * 48271 + 1) % 2147483647) % 2
                    if seed not in used and direction == index % 2:
                        used.add(seed)
                        break
                parameters = dict(seed=seed, opponent_speed=.35)
            else:
                parameters = asdict(next(breakout))
                if course == 'conflicting':
                    mapping, delay = BRICK_MAP_CONFLICT, 6
            cases.append(dict(case=len(cases), block=block, index=index, game=game,
                              parameters=parameters, mapping=mapping, delay=delay))
    return cases


class _PongLife:
    def __init__(self, seed, opponent_speed, max_ticks):
        self.arena = PongCamera(seed, max_episode_ticks=max_ticks, opponent_speed=opponent_speed)
        self.seed = seed
        self.counter = ReturnCounter()
        self.contacts = []
        game = self.arena.game
        native_step = game.step
        game.step = lambda drive, **kwargs: self.counter.step(game, native_step, drive, **kwargs)

    def frame(self):
        return self.arena.game.render(64, 64)[0].clone()

    def step(self, action):
        before = self.counter.opportunities
        result = self.arena.step(action)
        if self.counter.opportunities != before:
            self.contacts.append(dict(camera=self.arena.decisions, success=bool(result.hits)))
        return result

    def summary(self, last):
        elapsed = self.arena.total_ticks / 120
        return dict(seed=self.seed, cameras=self.arena.decisions, elapsed=elapsed,
                    contacts=self.contacts, own_opportunities=self.counter.opportunities,
                    own_returns=self.counter.returns, first_incoming=self.counter.first_incoming,
                    score=int(last.reward), terminated=last.terminated, truncated=last.truncated,
                    own_hit_win=bool(last.reward > 0 and self.counter.returns),
                    natural_serve_win=bool(last.reward > 0 and not self.counter.returns))


def play(session, case, max_ticks):
    """Deliver the same camera and reward interface regardless of the arena."""
    if case['game'] == 'pong':
        arena = _PongLife(**case['parameters'], max_ticks=max_ticks)
    else:
        arena = BreakoutEpisode(**case['parameters']).camera(
            max_episode_ticks=max_ticks, mapping=case['mapping'], delay_frames=case['delay'])
    initial_state = session.state()
    life = session.begin_life()
    frame = arena.frame()
    initial_frame_hash = hashlib.sha256(frame.numpy().tobytes()).hexdigest()
    life.reset(frame)
    public = hashlib.sha256(frame.numpy().tobytes())
    requests = bytearray()
    expected_order = tuple(case['mapping'].index(action) for action in range(3))
    expected_axis, expected_station = (0, -.45) if case['game'] == 'pong' else (1, .45)
    ready_camera = correct_role_camera = first_motor = first_delay = first_joint = None
    public_hits = 0
    public_score = 0.
    while True:
        action, _ = life.choose()
        result = arena.step(action)
        life.observe(result)
        requests.append(action)
        public.update(result.frame.numpy().tobytes())
        public.update(struct.pack('!dd??II', result.reward, result.elapsed_seconds,
                                  result.terminated, result.truncated, result.physics_ticks, result.hits))
        public_hits += result.hits
        public_score += result.reward
        if life.detector.ready and ready_camera is None:
            ready_camera = len(requests)
        binding = life.detector.binding
        correct_role = (binding is not None and binding.axis == expected_axis
                        and abs(binding.station - expected_station) < 1.5 / 64)
        if correct_role and life.detector.ready and correct_role_camera is None:
            correct_role_camera = len(requests)
        model = life.timing
        if correct_role and model.belief.states is not None:
            order_correct = (model.belief.order == expected_order
                             and model.belief.probabilities[model.belief.confirmed] >= .85)
            lag_correct = model.delay_commands == case['delay']
            if order_correct and first_motor is None:
                first_motor = len(requests)
            if lag_correct and first_delay is None:
                first_delay = len(requests)
            if order_correct and lag_correct and first_joint is None:
                first_joint = len(requests)
        if result.terminated or result.truncated:
            break
    board = arena.summary(result) if case['game'] == 'pong' else arena.summary()
    if board['own_returns'] != public_hits or board['score'] != public_score:
        raise AssertionError('public feedback and native outcome annotations diverged')
    session.end_life()
    final_state = session.state()
    if life.evidence.appearance != 'target_array' and initial_state['brick_policy'] != final_state['brick_policy']:
        raise AssertionError('target score memory was updated outside a matching visual scene')
    if life.evidence.appearance != 'opposing_rails' and any(
            initial_state['core'][key] != final_state['core'][key] for key in ('score', 'shot_memory')):
        raise AssertionError('opponent score memory was updated outside a matching visual scene')
    return dict(case=case, native=board, appearance=asdict(life.evidence),
                recalled_role=life.recalled_role, ready_camera=ready_camera,
                correct_role_camera=correct_role_camera,
                role_binding_seconds=life.detector.binding_seconds,
                role_probes=life.detector.probe_count, first_motor_camera=first_motor,
                first_delay_camera=first_delay, first_joint_camera=first_joint,
                initial_frame_sha256=initial_frame_hash, public_sha256=public.hexdigest(),
                legal_requests_hex=requests.hex(), action_sha256=hashlib.sha256(requests).hexdigest(),
                initial_state_sha256=digest(initial_state), final_state_sha256=digest(final_state),
                initial_timing=initial_state['core']['timing'], final_timing=final_state['core']['timing'],
                initial_binding=initial_state['core']['binding'], final_binding=final_state['core']['binding'],
                initial_polarity=initial_state['core']['polarity'], final_polarity=final_state['core']['polarity'],
                score_updates=sum(final_state['core']['score']['visits']) - sum(initial_state['core']['score']['visits']),
                initial_brick_updates=initial_state['brick_policy']['updates'],
                brick_updates=final_state['brick_policy']['updates'] - initial_state['brick_policy']['updates'])


def run_arm(out, arm, cases, checkpoint, brick_state, max_ticks):
    torch.set_num_threads(1)
    references = GAMES if arm == 'isolated' else ('shared',)
    sessions = {game: AlternatingSession(POLICY_SEED, checkpoint=checkpoint, brick_state=brick_state,
                                         recall_roles=arm == 'role_recall') for game in references}
    save(out / f'{arm}-initial.json', {key: session.state() for key, session in sessions.items()})
    rows = []
    last = {}
    for case in cases:
        key = case['game'] if arm == 'isolated' else 'shared'
        row = play(sessions[key], case, max_ticks)
        if key in last and row['initial_state_sha256'] != last[key]:
            raise AssertionError('the preceding learned state was not carried')
        last[key] = row['final_state_sha256']
        rows.append(row)
        save(out / f'{arm}-lives.json', rows)
    save(out / f'{arm}-final.json', {key: session.state() for key, session in sessions.items()})
    return rows


def summarize(rows):
    boards = [row['native'] for row in rows]
    contacts = [contact for board in boards for contact in board['contacts']]
    first = [board['contacts'][0] for board in boards if board['contacts']]
    first_five = contacts[:5]
    cameras = 0
    first_return = sustained = None
    for board in boards:
        consecutive = 0
        for contact in board['contacts']:
            if contact['success'] and first_return is None:
                first_return = cameras + contact['camera']
            consecutive = consecutive + 1 if contact['success'] else 0
            if consecutive == 2 and sustained is None:
                sustained = cameras + contact['camera']
        cameras += board['cameras']
    elapsed = sum(board['elapsed'] for board in boards)
    native = dict(lives=len(rows), cameras=cameras, elapsed=elapsed,
                  incoming=len(contacts), returns=sum(contact['success'] for contact in contacts),
                  opening_incoming=len(first), opening_returns=sum(contact['success'] for contact in first),
                  first_five_incoming=len(first_five), first_five_returns=sum(contact['success'] for contact in first_five),
                  first_return_camera=first_return, first_sustained_camera=sustained,
                  role_recall_lives=sum(row['recalled_role'] for row in rows),
                  ready_cameras=[row['correct_role_camera'] for row in rows],
                  joint_recovery_cameras=[row['first_joint_camera'] for row in rows],
                  score_updates=sum(row['score_updates'] for row in rows),
                  brick_updates=sum(row['brick_updates'] for row in rows))
    if rows[0]['case']['game'] == 'pong':
        native.update(own_hit_wins=sum(board['own_hit_win'] for board in boards),
                      natural_serve_wins=sum(board['natural_serve_win'] for board in boards))
    else:
        bricks = sum(board['post_own_bricks'] for board in boards)
        native.update(bricks=bricks, bricks_per_life=bricks / len(boards),
                      bricks_per_second=bricks / elapsed, boards_cleared=sum(board['cleared'] for board in boards))
    return native


def comparison(cases, traces):
    for case in cases:
        matched = [rows[case['case']] for rows in traces.values()]
        if len({row['initial_frame_sha256'] for row in matched}) != 1:
            raise AssertionError('matched policy arms have different opening camera frames')
        keys = ('tick', 'pre_ball', 'pre_velocity') if case['game'] == 'pong' else ('tick', 'ball', 'velocity')
        first = [row['native']['first_incoming'] for row in matched]
        geometry = [None if row is None else [row[key] for key in keys] for row in first]
        if any(row != geometry[0] for row in geometry):
            raise AssertionError('first incoming geometry differs across matched arms')
    result = {}
    for arm, rows in traces.items():
        blocks = [[row for row in rows if row['case']['block'] == block] for block in range(6)]
        result[arm] = dict(blocks=[dict(game=group[0]['case']['game'], reentry=block >= 2,
                                        first_life=summarize(group[:1]), **summarize(group))
                                   for block, group in enumerate(blocks)],
                           by_game={game: summarize([row for row in rows if row['case']['game'] == game]) for game in GAMES},
                           reentry_by_game={game: summarize([row for row in rows if row['case']['game'] == game
                                                          and row['case']['block'] >= 2]) for game in GAMES})
    return result


def run(output, *, course, games_per_block, brick_path, checkpoint, max_ticks, workers):
    cases = schedule(course, games_per_block)
    checkpoint = checkpoint.resolve()
    brick_path = brick_path.resolve()
    brick_state = json.loads(brick_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    implementation = ('arcade_alternation.py', 'arcade_transfer_session.py', 'breakout_arena.py',
                      'breakout_brick_placement.py', 'pong_role_transfer.py', 'return_metrics.py',
                      'pong_carried_shot_value.py', 'pong_timing_controller.py',
                      'pong_rapid_motor_recovery.py', 'pong_relational_persistent_binding.py',
                      'run_arcade_alternation.py')
    vision_model = scripts.parents[1] / 'src/fly_connectome/vision/data/default.pt'
    save(output / 'plan.json', dict(course=course, arms=ARMS, cases=cases, max_ticks=max_ticks,
                                    environment_seed=COURSE_SEEDS[course], policy_seed=POLICY_SEED,
                                    checkpoint_sha256=source_digest(checkpoint),
                                    brick_checkpoint_sha256=source_digest(brick_path),
                                    vision_model_sha256=source_digest(vision_model),
                                    source_sha256={name: source_digest(scripts / name) for name in implementation}))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = {arm: pool.submit(run_arm, output, arm, cases, checkpoint, brick_state, max_ticks) for arm in ARMS}
        traces = {arm: future.result() for arm, future in runs.items()}
    save(output / 'summary.json', comparison(cases, traces))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('brick_state', type=Path)
    parser.add_argument('--course', choices=COURSE_SEEDS, required=True)
    parser.add_argument('--games-per-block', type=int, default=6)
    parser.add_argument('--max-ticks', type=int, default=MAX_TICKS)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    options = parser.parse_args()
    run(options.output, course=options.course, games_per_block=options.games_per_block,
        brick_path=options.brick_state, checkpoint=options.checkpoint,
        max_ticks=options.max_ticks, workers=options.workers)


if __name__ == '__main__':
    main()
