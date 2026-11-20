"""Compare one unlabeled arcade stream with matched, uninterrupted game streams."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import struct

import numpy as np
import torch

from arcade_alternation import AlternatingSession
from arcade_transfer_session import CHECKPOINT
from breakout_arena import BreakoutEpisode, BreakoutLayout, breakout_schedule
from fly_connectome.control.environment import PongCamera
from return_metrics import ReturnCounter


COURSE_SEEDS = {'ordinary': 265049027, 'conflicting': 1327633107}
CONFIRMATION_SEEDS = {'ordinary': 1958120401, 'conflicting': 706290303}
COHORTS = {'initial': COURSE_SEEDS, 'confirmation': CONFIRMATION_SEEDS}
POLICY_SEED = 85217841
ARMS = ('shared', 'role_recall', 'isolated')
CONFIRMATION_ARMS = ('guarded', 'unguarded', 'isolated')
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


def schedule(course, games_per_block, *, cohort='initial'):
    if games_per_block < 2 or games_per_block % 2:
        raise ValueError('an even number of games supports balanced starts in each block')
    pong_rng, breakout_rng = np.random.SeedSequence(COHORTS[cohort][course]).spawn(2)
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
                    if seed not in used and direction == (index + 1) % 2:
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


def continuation_schedule(course, games_per_block):
    extra = schedule(course, games_per_block + 2, cohort='confirmation')
    return [case for case in extra if case['block'] >= 4 and case['index'] == games_per_block]


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


def native_arena(case, max_ticks):
    if case['game'] == 'pong':
        return _PongLife(**case['parameters'], max_ticks=max_ticks)
    return BreakoutEpisode(**case['parameters']).camera(
        max_episode_ticks=max_ticks, mapping=case['mapping'], delay_frames=case['delay'])


def hash_public_transition(public, result):
    public.update(result.frame.numpy().tobytes())
    public.update(struct.pack('!dd??', result.reward, result.elapsed_seconds,
                              result.terminated, result.truncated))


def identify_board(camera):
    return dict(layout=camera.layout.cells,
                initial_frame_sha256=hashlib.sha256(camera.frame().numpy().tobytes()).hexdigest())


def practice_boards(spec):
    if 'cases' in spec or 'phase_cases' in spec:
        practice = spec['cases'] if 'cases' in spec else spec['phase_cases']['practice']
        episodes = [row['episode'] for row in practice]
        layouts = [row['layout']['cells'] for row in practice]
    else:
        practice = spec['schedules']['practice']
        episodes = practice['environment']
        layouts = practice['custom_layouts']
    return [identify_board(BreakoutEpisode(**episode).camera(
        layout=None if layout is None else BreakoutLayout(layout)))
        for episode, layout in zip(episodes, layouts, strict=True)]


def verify_boards(practice_spec, games_per_block, *, cohort='initial'):
    practice = practice_boards(json.loads(practice_spec.read_text()))
    if not practice:
        raise ValueError('the trained brick state needs its practice schedule')
    seen = {key: {row[key] for row in practice} for key in ('layout', 'initial_frame_sha256')}
    if cohort == 'confirmation':
        for course in COURSE_SEEDS:
            for case in schedule(course, games_per_block):
                if case['game'] == 'breakout':
                    board = identify_board(native_arena(case, MAX_TICKS))
                    for key, entries in seen.items():
                        entries.add(board[key])
    evaluations = {}
    for course in COURSE_SEEDS:
        evaluations[course] = []
        for case in schedule(course, games_per_block, cohort=cohort):
            if case['game'] != 'breakout':
                continue
            board = identify_board(native_arena(case, MAX_TICKS))
            if any(board[key] in entries for key, entries in seen.items()):
                raise ValueError('a qualification board or opening frame repeats a known board')
            for key, entries in seen.items():
                entries.add(board[key])
            evaluations[course].append(dict(case=case['case'], **board))
    return dict(practice_spec_sha256=source_digest(practice_spec), practice=practice,
                distinct_from_initial_cohort=cohort == 'confirmation', evaluations=evaluations)


def play(session, case, max_ticks):
    """Deliver the same camera and reward interface regardless of the arena."""
    arena = native_arena(case, max_ticks)
    initial_state = session.state()
    initial_brick_updates = session.brick_policy.updates
    initial_visual_updates = getattr(session.brick_policy, 'visual_updates', 0)
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
        life.observe(replace(result, physics_ticks=0, hits=0))
        requests.append(action)
        hash_public_transition(public, result)
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
                recalled_role=life.recalled_role, recalled_timing=life.recalled_timing, ready_camera=ready_camera,
                correct_role_camera=correct_role_camera,
                role_binding_seconds=life.detector.binding_seconds,
                role_probes=life.detector.probe_count, first_motor_camera=first_motor,
                first_delay_camera=first_delay, first_joint_camera=first_joint,
                initial_frame_sha256=initial_frame_hash, public_sha256=public.hexdigest(),
                legal_requests_hex=requests.hex(), action_sha256=hashlib.sha256(requests).hexdigest(),
                initial_state_sha256=digest(initial_state), final_state_sha256=digest(final_state),
                initial_timing=initial_state['core']['timing'], effective_initial_timing=life.start_timing,
                final_timing=final_state['core']['timing'],
                initial_binding=initial_state['core']['binding'], final_binding=final_state['core']['binding'],
                initial_polarity=initial_state['core']['polarity'], final_polarity=final_state['core']['polarity'],
                score_updates=sum(final_state['core']['score']['visits']) - sum(initial_state['core']['score']['visits']),
                initial_brick_updates=initial_brick_updates,
                brick_updates=session.brick_policy.updates - initial_brick_updates,
                visual_brick_updates=getattr(session.brick_policy, 'visual_updates', 0) - initial_visual_updates)


def run_arm(out, arm, cases, checkpoint, brick_state, max_ticks, *, cohort, context_recall, continuation):
    torch.set_num_threads(1)
    references = GAMES if arm == 'isolated' else ('shared',)
    confirmation = cohort == 'confirmation'
    sessions = {game: AlternatingSession(POLICY_SEED, checkpoint=checkpoint, brick_state=brick_state,
                                         recall_roles=confirmation or arm == 'role_recall',
                                         recall_timing=(confirmation and context_recall) or arm == 'context_recall',
                                         causal_pursuit=confirmation and arm != 'unguarded')
                for game in references}
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
    if arm == 'guarded':
        live = sessions['shared']
        restored = AlternatingSession.from_state(live.state())
        if restored.state() != live.state():
            raise AssertionError('reloading changed the continued learning state')
        resume_rows = []
        for case in continuation:
            row = play(live, case, max_ticks)
            replay = play(restored, case, max_ticks)
            if row != replay:
                raise AssertionError('the loaded learner differs from the in-memory session')
            resume_rows.append(row)
        save(out / f'{arm}-resumed.json', dict(exact_actions_and_final_state=True, lives=resume_rows))
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
    game = rows[0]['case']['game']
    expected_appearance = 'opposing_rails' if game == 'pong' else 'target_array'
    native = dict(lives=len(rows), cameras=cameras, elapsed=elapsed,
                  timeouts=sum(board['truncated'] for board in boards),
                  incoming=len(contacts), returns=sum(contact['success'] for contact in contacts),
                  opening_incoming=len(first), opening_returns=sum(contact['success'] for contact in first),
                  first_five_incoming=len(first_five), first_five_returns=sum(contact['success'] for contact in first_five),
                  first_return_camera=first_return, first_sustained_camera=sustained,
                  matching_appearance_lives=sum(row['appearance']['appearance'] == expected_appearance for row in rows),
                  role_recall_lives=sum(row['recalled_role'] for row in rows),
                  timing_recall_lives=sum(row['recalled_timing'] for row in rows),
                  ready_cameras=[row['correct_role_camera'] for row in rows],
                  joint_recovery_cameras=[row['first_joint_camera'] for row in rows],
                  score_updates=sum(row['score_updates'] for row in rows),
                  brick_updates=sum(row['brick_updates'] for row in rows),
                  visual_brick_updates=sum(row['visual_brick_updates'] for row in rows))
    if game == 'pong':
        native.update(own_hit_wins=sum(board['own_hit_win'] for board in boards),
                      natural_serve_wins=sum(board['natural_serve_win'] for board in boards))
    else:
        bricks = sum(board['post_own_bricks'] for board in boards)
        native.update(bricks=bricks, bricks_per_life=bricks / len(boards),
                      bricks_per_minute=60 * bricks / elapsed, boards_cleared=sum(board['cleared'] for board in boards))
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


def retention_decision(summary, course):
    guarded = summary['guarded']['reentry_by_game']
    isolated = summary['isolated']['reentry_by_game']
    floor = .9 if course == 'ordinary' else .75
    metrics = {}
    for game in GAMES:
        row, comparator = guarded[game], isolated[game]
        rate = row['returns'] / row['incoming'] if row['incoming'] else None
        paired_rate = comparator['returns'] / comparator['incoming'] if comparator['incoming'] else None
        difference = None if rate is None or paired_rate is None else rate - paired_rate
        metrics[game] = dict(return_rate=rate, isolated_return_rate=paired_rate, rate_difference=difference,
                             rate_pass=rate is not None and rate >= floor,
                             isolated_distance_pass=difference is not None and difference >= -.10)
    scores = dict(pong_wins_after_return=guarded['pong']['own_hit_wins'],
                  breakout_bricks_after_return=guarded['breakout']['bricks'])
    useful = all(scores.values()) and all(row['rate_pass'] for row in metrics.values())
    if course == 'ordinary':
        useful = useful and all(row['isolated_distance_pass'] for row in metrics.values())
    return dict(reentry=metrics, scores=scores, useful_retention=bool(useful))


def run(output, *, course, games_per_block, brick_path, practice_spec, checkpoint, max_ticks, workers,
        context_recall=False, cohort='initial'):
    cases = schedule(course, games_per_block, cohort=cohort)
    confirmation = cohort == 'confirmation'
    arms = CONFIRMATION_ARMS if confirmation else (*ARMS, 'context_recall') if context_recall else ARMS
    continuation = continuation_schedule(course, games_per_block) if confirmation else []
    checkpoint = checkpoint.resolve()
    brick_path = brick_path.resolve()
    brick_state = json.loads(brick_path.read_text())
    boards = verify_boards(practice_spec, games_per_block, cohort=cohort)
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    implementation = ('arcade_alternation.py', 'arcade_transfer_session.py', 'breakout_arena.py',
                      'breakout_brick_placement.py', 'breakout_visual_rebound.py', 'pong_role_transfer.py', 'return_metrics.py',
                      'pong_carried_shot_value.py', 'pong_timing_controller.py',
                      'pong_rapid_motor_recovery.py', 'pong_relational_persistent_binding.py',
                      'run_arcade_alternation.py')
    vision_model = scripts.parents[1] / 'src/fly_connectome/vision/data/default.pt'
    save(output / 'plan.json', dict(course=course, cohort=cohort, arms=arms, cases=cases, max_ticks=max_ticks,
                                    continuation=continuation, context_recall=context_recall,
                                    environment_seed=COHORTS[cohort][course], policy_seed=POLICY_SEED,
                                    checkpoint_sha256=source_digest(checkpoint),
                                    brick_checkpoint_sha256=source_digest(brick_path),
                                    brick_checkpoint_schema=brick_state['schema'],
                                    vision_model_sha256=source_digest(vision_model),
                                    source_sha256={name: source_digest(scripts / name) for name in implementation}))
    save(output / 'board-provenance.json', boards)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        runs = {arm: pool.submit(run_arm, output, arm, cases, checkpoint, brick_state, max_ticks,
                                 cohort=cohort, context_recall=context_recall, continuation=continuation)
                for arm in arms}
        traces = {arm: future.result() for arm, future in runs.items()}
    summary = comparison(cases, traces)
    save(output / 'summary.json', summary)
    if confirmation:
        save(output / 'retention.json', retention_decision(summary, course))


def audit_recorded_requests(output):
    """Replay saved requests through native physics and check the complete public trajectory."""
    torch.set_num_threads(1)
    plan = json.loads((output / 'plan.json').read_text())
    boards = json.loads((output / 'board-provenance.json').read_text())
    planned_boards = {row['case']: row for row in boards['evaluations'][plan['course']]}
    traces = {}
    audit = {}
    for arm in plan['arms']:
        rows = json.loads((output / f'{arm}-lives.json').read_text())
        initial = json.loads((output / f'{arm}-initial.json').read_text())
        final = json.loads((output / f'{arm}-final.json').read_text())
        last = {key: digest(value) for key, value in initial.items()}
        boundaries = 0
        cameras = 0
        for case, row in zip(plan['cases'], rows, strict=True):
            key = case['game'] if arm == 'isolated' else 'shared'
            if case != row['case'] or row['initial_state_sha256'] != last[key]:
                raise AssertionError('the saved lives break the registered schedule or learned state chain')
            last[key] = row['final_state_sha256']
            boundaries += 1
            arena = native_arena(case, plan['max_ticks'])
            public = hashlib.sha256(arena.frame().numpy().tobytes())
            if public.hexdigest() != row['initial_frame_sha256']:
                raise AssertionError('the native opening frame differs from the matched receipt')
            if case['game'] == 'breakout' and digest(dict(case=case['case'], **identify_board(arena))) != digest(
                    planned_boards[case['case']]):
                raise AssertionError('the native board differs from the reserved held-out board')
            requests = bytes.fromhex(row['legal_requests_hex'])
            if not requests:
                raise AssertionError('a native life must include a recorded request')
            for camera, action in enumerate(requests, 1):
                result = arena.step(action)
                hash_public_transition(public, result)
                if (result.terminated or result.truncated) != (camera == len(requests)):
                    raise AssertionError('native completion differs from the recorded action trace')
            board = arena.summary(result) if case['game'] == 'pong' else arena.summary()
            if (board != row['native'] or public.hexdigest() != row['public_sha256']
                    or hashlib.sha256(requests).hexdigest() != row['action_sha256']):
                raise AssertionError('public replay or native outcome differs from the receipt')
            cameras += len(requests)
        if last != {key: digest(value) for key, value in final.items()}:
            raise AssertionError('the final saved states differ from the last played lives')
        audit[arm] = dict(lives=len(rows), transitions=cameras, initial_or_carried_boundaries=boundaries,
                          exact_public_replay=True)
        traces[arm] = rows
    if comparison(plan['cases'], traces) != json.loads((output / 'summary.json').read_text()):
        raise AssertionError('the summary differs from the paired receipts')
    save(output / 'native-replay.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('brick_state', type=Path)
    parser.add_argument('--practice-spec', type=Path, required=True)
    parser.add_argument('--course', choices=COURSE_SEEDS, required=True)
    parser.add_argument('--cohort', choices=COHORTS, default='initial')
    parser.add_argument('--games-per-block', type=int, default=6)
    parser.add_argument('--max-ticks', type=int, default=MAX_TICKS)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument('--context-recall', action='store_true', help='also test visually retrieved timing priors')
    options = parser.parse_args()
    run(options.output, course=options.course, games_per_block=options.games_per_block,
        brick_path=options.brick_state, practice_spec=options.practice_spec, checkpoint=options.checkpoint,
        max_ticks=options.max_ticks, workers=options.workers, context_recall=options.context_recall,
        cohort=options.cohort)


if __name__ == '__main__':
    main()
