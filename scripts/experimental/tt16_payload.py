"""Transfer a table motor map to an inertial puck driven by arm displacement."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import MOTOR_BUDGET
from .tt6_memory import load
from .tt10_target import TargetArena
from .tt16_transfer_protocol import FIXTURES, setup_path


RUN_SEED = 8286329
PULSE_FRAMES = 2
BLOCK_FRAMES = 8
GOAL_BLOCKS = 18
TARGETS = 4
PIXELS = 256
VIEW_SCALE = 125.
MODEL_SCALE = .10
GOAL_RADIUS = .039
STOP_SPEED = .019
HOLD_BLOCKS = 3
EARLY_BLOCKS = 12
CASES = ('carried', 'wrong', 'cold')


def display(tool, puck, goal):
    """Two RGB views with a common origin and coincident colors preserved."""
    image = np.zeros((2, PIXELS, PIXELS, 3), dtype=np.uint8)
    markers = np.array((tool, (*puck, 0.), (*goal, 0.)))
    pixels = np.clip(np.rint(VIEW_SCALE * markers) + PIXELS // 2, 1, PIXELS - 2).astype(int)
    colors = np.arange(3)
    image[0, pixels[:, 1], pixels[:, 0], colors] = 255
    image[1, pixels[:, 2], pixels[:, 0], colors] = 255
    return image


def see(image):
    plane, elevation = image.reshape(2, -1, 3).argmax(axis=1)
    y, x = np.divmod(plane, PIXELS)
    z = elevation // PIXELS
    return (np.stack((x, y, z), axis=1) - PIXELS // 2) / (VIEW_SCALE * MODEL_SCALE)


def course(seed):
    rng = np.random.default_rng(seed)
    axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    coupling = np.diag(rng.uniform(.13, .26, 2)) @ axes[:2]
    axes, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    retention = (axes * rng.uniform(.48, .82, 2)) @ axes.T
    goals = [np.zeros(2)]
    while len(goals) <= TARGETS:
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(.16, .29)
        proposal = radius * np.array((np.cos(angle), np.sin(angle)))
        if np.linalg.norm(proposal - goals[-1]) > .18:
            goals.append(proposal)
    return coupling, retention, np.asarray(goals[1:])


class VisualLearner:
    """Fit two local maps from successive RGB frames and requested motor pulses."""

    def __init__(self, prior, seed):
        self.width = prior.shape[1]
        self.motor_prior = .8 * np.eye(self.width)
        self.motor_gram = self.motor_prior.copy()
        self.motor_cross = self.motor_prior @ (prior * PULSE_FRAMES / MODEL_SCALE).T
        self.object_gram = np.diag((.05, .05, .07, .07, .07))
        self.object_cross = np.zeros((5, 2))
        self.object_cross[:2] = .025 * np.eye(2)
        self.requested = np.zeros(self.width)
        self.command = np.zeros(self.width)
        self.previous = None
        self.tick = 0
        self.done = False
        self.rng = np.random.default_rng(seed)

    def models(self):
        return (np.linalg.solve(self.motor_gram, self.motor_cross).T,
                np.linalg.solve(self.object_gram, self.object_cross).T)

    def act(self, image, reward):
        tool, puck, goal = see(image)
        puck, goal = puck[:2], goal[:2]
        velocity = np.zeros(2)
        if self.previous is not None:
            last_tool, last_puck, last_velocity, last_goal = self.previous
            velocity = puck - last_puck
            self.motor_gram += np.outer(self.command, self.command)
            self.motor_cross += np.outer(self.command, tool - last_tool)
            evidence = np.r_[last_velocity, tool]
            self.object_gram += np.outer(evidence, evidence)
            self.object_cross += np.outer(evidence, velocity)
            self.done = (self.done or bool(reward)) and np.array_equal(last_goal, goal)
        self.previous = tool, puck, velocity, goal

        motor, response = self.models()
        target_velocity = np.clip(.31 * (goal - puck), -.78, .78)
        velocity_change = target_velocity - response @ np.r_[velocity, tool]
        effect = response[:, 2:] @ motor
        command = np.linalg.solve(effect.T @ effect + .009 * motor.T @ motor + .018 * np.eye(self.width),
                                  effect.T @ velocity_change - .009 * motor.T @ tool)

        exploration = self.rng.uniform(-1., 1., self.width)
        if self.tick < 3:
            command = .64 * exploration
        elif not self.done:
            command += max(.025, .31 / np.sqrt(self.tick - 1)) * exploration
        lower = np.maximum(-.75, (-MOTOR_BUDGET - self.requested) / PULSE_FRAMES)
        upper = np.minimum(.75, (MOTOR_BUDGET - self.requested) / PULSE_FRAMES)
        self.command = np.clip(command, lower, upper)
        self.requested += PULSE_FRAMES * self.command
        self.tick += 1
        return self.command


def play(arena, wiring, scenario, seed, *, learner=None, tape=None):
    coupling, retention, goals = scenario
    wiring.reset(seed=seed, render=False)
    puck, velocity = np.zeros((2, 2))
    blank = np.zeros(arena.action_size)
    reward = 0
    rows = []
    for target_index, goal in enumerate(goals):
        streak, scored = 0, False
        for offset in range(GOAL_BLOCKS):
            image = display(arena.paddle - arena.origin, puck, goal)
            marks = see(image)
            command = (learner.act(image, reward) if tape is None
                       else np.asarray(tape[target_index * GOAL_BLOCKS + offset]))
            if not np.isfinite(command).all() or np.max(np.abs(command)) > 1.:
                raise ValueError('invalid motor command')
            for tick in range(BLOCK_FRAMES):
                wiring.step(command if tick < PULSE_FRAMES else blank, render=False)
            tool = arena.paddle - arena.origin
            velocity = retention @ velocity + coupling @ tool
            puck += velocity
            distance, speed = float(np.linalg.norm(puck - goal)), float(np.linalg.norm(velocity))
            inside = distance <= GOAL_RADIUS and speed <= STOP_SPEED
            streak = streak + 1 if inside else 0
            reward = int(streak >= HOLD_BLOCKS and not scored)
            scored |= bool(reward)
            rows.append(dict(action=command.tolist(), marks=marks.tolist(), reward=reward,
                             tool=tool.tolist(), puck=puck.tolist(), distance=distance, speed=speed))
    return rows


def summarize(rows):
    acquisition = [None] * TARGETS
    for tick, row in enumerate(rows):
        if row['reward']:
            index, offset = divmod(tick, GOAL_BLOCKS)
            acquisition[index] = (offset + 1) * BLOCK_FRAMES
    return dict(acquisition=acquisition, points=sum(a is not None for a in acquisition),
                first_goal=acquisition[0], early_error=float(np.mean([r['distance'] for r in rows[:EARLY_BLOCKS]])),
                late_error=float(np.mean([r['distance'] for r in rows[-GOAL_BLOCKS:]])),
                scene_edge_blocks=int(sum(np.max(np.abs(r['marks'])) >= (PIXELS // 2 - 2) / (VIEW_SCALE * MODEL_SCALE)
                                          for r in rows)),
                rate_l1_frames=PULSE_FRAMES * sum(sum(map(abs, row['action'])) for row in rows))


def run_fixture(folder, fixture, seed, count, output):
    setup = setup_path(folder, fixture)
    settings = _read(setup)
    cp = folder / f'transfer-{fixture}-coded-player.json.gz'
    game_path = folder / f'transfer-{fixture}-coded.json.gz'
    game = _read(game_path)
    if game['checkpoint'] != _hash(cp) or game['provenance']['setup'] != _hash(setup):
        raise ValueError('checkpoint or installation differs from the coded table receipt')
    library = load(cp)[0]
    port_memory = _read(cp)['port_memory']
    indices = port_memory['indices']
    active = port_memory['active']
    models = (library.models[indices[active]], library.models[indices[0]])
    carried, wrong = (m.jacobian(np.zeros(m.action_size))[:3] for m in models)
    priors = dict(carried=carried, wrong=wrong, cold=np.zeros_like(carried))
    fixture_index = tuple(FIXTURES).index(fixture)
    seeds = np.random.SeedSequence([seed, fixture_index]).spawn(count)
    arena = TargetArena(**dict(settings['arena'], render_mode='projected'))
    runs = []
    try:
        wiring = TT6Wiring(arena, settings['wiring'])
        for episode in seeds:
            world_seed, learner_seed = map(int, episode.generate_state(2))
            scenario = course(world_seed)
            results = {}
            for name, prior in priors.items():
                learner = VisualLearner(prior, learner_seed)
                rows = play(arena, wiring, scenario, world_seed, learner=learner)
                tape = [row['action'] for row in rows]
                if play(arena, wiring, scenario, world_seed, tape=tape) != rows:
                    raise ValueError('RGB, reward, or evaluator state diverged on physical replay')
                motor, response = learner.models()
                results[name] = dict(summary=summarize(rows), motor=motor.tolist(),
                                     object=response.tolist(), blocks=rows)
            if any(results['carried']['blocks'][:3] != results[name]['blocks'][:3] for name in ('wrong', 'cold')):
                raise ValueError('matched exploration prefixes diverged')
            runs.append(dict(world_seed=world_seed, learner_seed=learner_seed,
                             coupling=scenario[0].tolist(), retention=scenario[1].tolist(),
                             goals=scenario[2].tolist(), cases=results))
    finally:
        arena.close()
    result = dict(schema='tt16-payload-v1', fixture=fixture, seed=seed, setup_sha256=_hash(setup),
                  checkpoint_sha256=_hash(cp), game_sha256=_hash(game_path),
                  implementation_sha256=_hash(Path(__file__)),
                  prior={k: v.tolist() for k, v in priors.items()}, control_dt=settings['control_dt'],
                  budget=dict(goal_blocks=GOAL_BLOCKS, block_frames=BLOCK_FRAMES, targets=TARGETS,
                              pulse_frames=PULSE_FRAMES, common_probe_blocks=3, early_blocks=EARLY_BLOCKS,
                              goal_radius=GOAL_RADIUS, stop_speed_per_block=STOP_SPEED, hold_blocks=HOLD_BLOCKS),
                  replayed=True, runs=runs)
    _write(output, result)
    for name in CASES:
        summaries = [r['cases'][name]['summary'] for r in runs]
        print(fixture, name, 'points', sum(s['points'] for s in summaries),
              'first', [s['first_goal'] for s in summaries], 'early',
              round(float(np.mean([s['early_error'] for s in summaries])), 4),
              'late', round(float(np.mean([s['late_error'] for s in summaries])), 4),
              'edge', sum(s['scene_edge_blocks'] for s in summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('fixture', choices=FIXTURES)
    parser.add_argument('--seed', type=int, default=RUN_SEED)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    run_fixture(args.folder, args.fixture, args.seed, args.count,
                args.output or args.folder / f'payload-{args.fixture}.json.gz')
