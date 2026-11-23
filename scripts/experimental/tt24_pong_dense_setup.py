"""Complete a rank-one native Pong motor memory on the same arm before wrist play."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import null_space
import torch

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_current_pause import Fixture
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_joint_pong import WARM, arcade
from .tt24_motor_bank import I, STEP, decoder
from .tt24_played_pong_prefix import from_prefix
from .tt24_pong_arm import ACTUATION, DIRECTION, PADDLE_SCALE, PARKED, wirings
from .tt24_tool_equilibrium import ToolEquilibrium


SOURCE = ROOT / 'played-pong-prefix.json.gz'
OUTPUT = ROOT / 'pong-dense-setup.json.gz'
BANK = ROOT / 'pong-dense-setup-bank.npz'
CAP = 360
PREFIX = 30
ROSTER = 154904621
CALIBRATION_SECONDS = 'one fixture tick = 1/60 s, two native Pong frames'
PROTOCOL = 'continuous prefix and four-direction charged measurement plus shared optical preparation'
CONDITIONS = ('pong', 'archive', 'stale', 'shuffled', 'cold')
SWITCHED = (8, 9)
COMPLEMENT = null_space(DIRECTION[None, :]).T


class PongPort:
    """Native screen paddle moves from actual arm encoder response through the same fixture."""

    def __init__(self, fixture, obs, spec):
        self.fixture = fixture
        self.spec = spec
        self.home = obs['joints'].astype(float)
        self.commands = []
        self.encoders = [self.home.tolist()]
        self.position = torch.full((1,), .5)
        self.ticks = 0

    def step(self, drive):
        if self.ticks % 2 == 0:
            request = -ACTUATION * float(drive[0]) * DIRECTION
            obs, _ = self.fixture.issue(request, 'pong', False)
            q = obs['joints'].astype(float)
            self.encoders.append(q.tolist())
            self.commands.append(request.tolist())
            projected = .5 + PADDLE_SCALE * ((q - self.home) @ DIRECTION)
            self.position[0] = np.clip(projected, self.spec.paddle_half, 1. - self.spec.paddle_half)
        self.ticks += 1


def play_prefix(fixture, initial, native_seed, weights):
    port = None

    def attach(spec):
        nonlocal port
        port = PongPort(fixture, initial, spec)
        return port

    score, hits, frames = arcade(weights, [native_seed], cap=2 * PREFIX, opponent=.35,
                                 paddle_factory=attach)[0, 0]
    if len(port.commands) != PREFIX:
        raise ValueError(f'native point ended before the fixed prefix on seed {native_seed}')
    return dict(seed=native_seed, game_score=float(score), hits=int(hits), native_frames=int(frames),
                commands=port.commands, encoders=port.encoders)


def rest(q, send):
    stable = 0
    for j in range(30):
        received = send(np.zeros(5))
        stable = stable + 1 if np.max(abs(received - q)) < .00045 else 0
        q = received
        if j >= 4 and stable >= 3:
            break
    return q, j + 1


def inspect(fixture, obs, axes):
    """Map physical settled encoder effects only in the span of the issued setup requests."""
    def send(request):
        new, _ = fixture.issue(request, 'complement', False)
        return new['joints']

    q, before = rest(obs['joints'], send)
    points = np.vstack((np.zeros((1, 5)), .084 * axes, np.zeros((1, 5))))
    measured, requests = [], []
    segments = [before]
    for origin, target in zip(points[:-1], points[1:], strict=True):
        delta = target - origin
        old = q.astype(float)
        for _ in range(4):
            q = send(delta / (STEP * 4))
        q, elapsed = rest(q, send)
        requests.append(delta)
        measured.append(q.astype(float) - old)
        segments.append(elapsed + 4)
    x, y = np.asarray(requests), np.asarray(measured)
    restricted = np.linalg.lstsq(x @ axes.T, y, rcond=None)[0].T @ axes
    return restricted, dict(requested_displacements=x.tolist(), measured_displacements=y.tolist(),
                            segment_ticks=segments, setup_ticks=sum(segments),
                            dimension=int(np.linalg.matrix_rank(x)), basis=axes.tolist())


def fixture_for(arm, plan, source_wire):
    return Fixture(arm, plan, (PARKED.copy(), np.zeros(3), np.zeros(3)), source_wire)


def present(fixture, launch, render):
    fixture.location, fixture.velocity, fixture.spin = launch
    env = fixture.env
    env._ball_position[:] = fixture.location
    env._ball_velocity[:] = 0.
    env._ball_spin[:] = 0.
    mujoco.mj_forward(env._model, env._data)
    return env._observation(render)



def prepare(arm, plan, source_wire, target_wire, axes, native_seed, weights, mode):
    fixture = fixture_for(arm, plan, source_wire)
    obs = fixture.start(False)
    received = None
    if mode != 'cold':
        received = play_prefix(fixture, obs, native_seed, weights)
        obs = fixture.env._observation(False)
    handoff = len(fixture.actions)
    fixture.wire = target_wire
    measured, evidence = inspect(fixture, obs, axes)
    appearance = len(fixture.actions)
    return fixture, measured, evidence, received, handoff, appearance


def matched_setup(arm, plan, tape, handoff, appearance, source_wire, target_wire):
    trial = fixture_for(arm, plan, source_wire)
    trial.start(False)
    for tick, (command, request, stage) in enumerate(zip(*(part[:appearance] for part in tape), strict=True)):
        if tick == handoff:
            trial.wire = target_wire
        reconstructed = np.clip(trial.wire @ request, -1., 1.)
        if not np.array_equal(reconstructed, command):
            raise ValueError('setup tape does not match requested commands through actual wiring')
        trial.advance(command, stage, False, request)
    return trial


def play(fixture, policy, plan, launch, handoff, appearance, mode, estimate, setup_map):
    if len(fixture.actions) != appearance:
        raise ValueError('dispenser seen before the charged calibration finished')
    obs = present(fixture, launch, True)
    policy.begin_release(obs)
    setup_inverse = decoder(setup_map)
    for _ in range(plan['prep_ticks']):
        desired = policy.act(obs)
        requested = setup_inverse @ desired
        if mode != 'cold':
            # Shared readiness uses only the same four independently observed directions.
            requested -= (requested @ DIRECTION) * DIRECTION
        requested = np.clip(requested, -1., 1.)
        policy.issued += STEP * (setup_map @ requested - desired)
        obs, _ = fixture.issue(requested, 'optical_setup', True)
    obs = fixture.release(True)
    inverse = decoder(estimate)
    while True:
        request = np.clip(inverse @ policy.act(obs), -1., 1.)
        obs, done = fixture.issue(request, 'playing', True)
        if done:
            break
    return dict(handoff_tick=handoff, ball_visible_tick=appearance, mode=mode,
                estimated_map=estimate.tolist(), **fixture.score())


def verify(arm, plan, launch, source_wire, target_wire, tape):
    fixture = fixture_for(arm, plan, source_wire)
    fixture.start(False)
    for tick, (command, stage, request) in enumerate(zip(tape['actions'], tape['phases'], tape['requests'],
                                                        strict=True)):
        if tick == tape['handoff_tick']:
            fixture.wire = target_wire
        if tick == tape['ball_visible_tick']:
            present(fixture, launch, False)
        if tick == tape['release_tick']:
            fixture.release(False)
        if not np.array_equal(np.clip(fixture.wire @ request, -1., 1.), command):
            raise ValueError('target tape motor effects differ')
        _, done = fixture.advance(command, stage, False, request)
        if done != (tick == len(tape['actions']) - 1):
            raise ValueError('replay differs in terminal tick')
    expected = fixture.score()
    for key in expected:
        prior = tape[key]
        if key == 'release_state':
            prior = {k: v for k, v in prior.items() if k != 'wrist_sha256'}
        if digest_json(expected[key]) != digest_json(prior):
            raise ValueError(f'replay differs in {key}')


def roster():
    return [dict(index=i, **item) for i, item in enumerate(x for x in schedule(ROSTER, 10)
                                                         if x['body'] == 'stock')]


def installations(index):
    """Generate unseen genuinely dense physical maps, invisible to the fit."""
    rng = np.random.default_rng(836440126 + index)
    posed_column = wirings()[0] @ DIRECTION
    perpendicular = null_space(posed_column[None, :])
    mixing, reflection = np.linalg.qr(rng.normal(size=(4, 4)))
    mixing *= np.sign(np.diag(reflection)) * rng.uniform(.82, 1.03, 4)
    other = perpendicular @ mixing + np.outer(posed_column, rng.uniform(-.14, .14, 4))
    source_wire = np.outer(posed_column, DIRECTION) + other @ COMPLEMENT
    receiver = source_wire.copy()
    if index in SWITCHED:
        # Replacement reverses the uninspected direction and preserves all other responses.
        receiver -= 2 * np.outer(posed_column, DIRECTION)
    directions, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    axes = directions @ COMPLEMENT
    return source_wire, receiver, axes


def dispatcher(arm, server, policy, plan, archived, weights):
    index = plan['index']
    source_wire, target_wire, complement = installations(index)
    native_seed = archived['source_seeds'][index % len(archived['source_seeds'])]
    launch = launch_from_plan(server, plan)

    warm, submap, inspected, source, handoff, appearance = prepare(
        arm, plan, source_wire, target_wire, complement, native_seed, weights, 'pong')
    if handoff != PREFIX or inspected['dimension'] != 4:
        raise ValueError('setup did not use one source and four receiving dimensions')
    good, source_fit = from_prefix(source, PREFIX)
    shuffled, shuffle_fit = from_prefix(source, PREFIX, shuffled=True)

    conditions = {}
    for label, observation in (('pong', good), ('stale', I), ('shuffled', shuffled)):
        estimate = submap + np.outer(observation @ DIRECTION, DIRECTION)
        fixture = warm if label == 'pong' else matched_setup(
            arm, plan, (warm.actions, warm.requests, warm.phases), handoff, appearance, source_wire, target_wire)
        result = play(fixture, policy, plan, launch, handoff, appearance, label, estimate, submap)
        verify(arm, plan, launch, source_wire, target_wire, result)
        conditions[label] = result
    for result in list(conditions.values())[1:]:
        first = conditions['pong']
        if (result['release_state'] != first['release_state'] or result['release_tick'] != first['release_tick']
                or result['actions'][:result['release_tick']] != first['actions'][:first['release_tick']]):
            raise ValueError('Pong, stale and shuffled fits did not reach the exact same physical release')

    warm, estimate, cold_fit, no_source, handoff, appearance = prepare(
        arm, plan, source_wire, target_wire, np.vstack((complement, DIRECTION)),
        native_seed, weights, 'cold')
    if no_source is not None or handoff != 0 or cold_fit['dimension'] != 5:
        raise ValueError('cold fit did not use five individual physical input directions')
    cold = play(warm, policy, plan, launch, handoff, appearance, 'cold', estimate, estimate)
    verify(arm, plan, launch, source_wire, target_wire, cold)
    conditions['cold'] = cold

    archived_fit, _ = from_prefix(archived['source'][index % len(archived['source'])], PREFIX)
    evaluator = {}
    for label, row in conditions.items():
        estimated = np.asarray(row['estimated_map'])
        evaluator[label] = dict(relative_map_error=float(np.linalg.norm(estimated - target_wire)
                                                         / np.linalg.norm(target_wire)),
                                column_error=float(np.linalg.norm((estimated - target_wire) @ DIRECTION)))
    return dict(plan=plan, continuity=index not in SWITCHED, conditions=conditions, source=source,
                source_fit=source_fit, shuffled_fit=shuffle_fit, setup=inspected, cold_setup=cold_fit,
                archived_column=(archived_fit @ DIRECTION).tolist(),
                judge=dict(source_wiring=source_wire.tolist(), target_wiring=target_wire.tolist(), error=evaluator))


def summarize(cases):
    results = {}
    for name, subset in (('continuous', [x for x in cases if x['continuity']]),
                         ('switched', [x for x in cases if not x['continuity']])):
        if not subset:
            continue
        results[name] = {}
        for group in CONDITIONS:
            if not all(group in case['conditions'] for case in subset):
                continue
            rows = [case['conditions'][group] for case in subset]
            counts = [r['metrics']['rebounds'] for r in rows]
            pre = [r['release_tick'] for r in rows]
            live = [r['live_ticks'] for r in rows]
            setup = [r['ball_visible_tick'] - r['handoff_tick'] for r in rows]
            result = dict(counts=counts, total=sum(counts), max=max(counts),
                          longest=[r['metrics']['longest_consecutive_rebounds'] for r in rows],
                          first=sum(k > 0 for k in counts), second=sum(k > 1 for k in counts),
                          five_seconds=sum(n >= 300 for n in live),
                          mean_motor_setup_ticks=float(np.mean(setup)),
                          mean_total_pre_release_ticks=float(np.mean(pre)),
                          rebounds_per_wall_simulated_min=60 * 60 * sum(counts) / sum(np.add(pre, live)),
                          mean_map_error=float(np.mean([c['judge']['error'][group]['relative_map_error']
                                                        for c in subset])))
            results[name][group] = result
    return results


def launch_from_plan(server, plan):
    server.reset(seed=plan['launcher_seed'], split='held', render=False)
    return tuple(value.copy() for value in (server._ball_position, server._ball_velocity, server._ball_spin))


def stock_arm():
    arm = InstalledArm('stock')
    arm.arena.close()
    arm.arena = BounceArena(arm='stock', view='wrist', width=192, height=144, max_steps=CAP)
    return arm


def wrist_policy():
    policy = ToolEquilibrium(BUNDLE, .0557)
    policy.mode = 'workspace_local'
    return policy


def archive_trials(report, arm, server, policy, source, path):
    """Qualify the exact first historical Pong coefficient under the same hardware preparation."""
    archive_estimate, fit = from_prefix(source['source'][0], PREFIX)
    report['archived_control_source'] = dict(seed=source['source_seeds'][0], fit=fit)
    for case in report['cases']:
        if 'archive' in case['conditions']:
            continue
        genuine = case['conditions']['pong']
        plan = case['plan']
        source_wire, target_wire, _ = installations(plan['index'])
        handoff, appearance = genuine['handoff_tick'], genuine['ball_visible_tick']
        launch = launch_from_plan(server, plan)
        tape = tuple(genuine[key] for key in ('actions', 'requests', 'phases'))
        fixture = matched_setup(arm, plan, tape, handoff, appearance, source_wire, target_wire)
        submap = np.asarray(genuine['estimated_map']) @ (I - np.outer(DIRECTION, DIRECTION))
        estimate = submap + np.outer(archive_estimate @ DIRECTION, DIRECTION)
        played = play(fixture, policy, plan, launch, handoff, appearance, 'archive', estimate, submap)
        verify(arm, plan, launch, source_wire, target_wire, played)
        if (played['release_state'] != genuine['release_state'] or
                played['actions'][:played['release_tick']] != genuine['actions'][:genuine['release_tick']]):
            raise ValueError('exact archived estimate changed the matched release')
        case['conditions']['archive'] = played
        case['judge']['error']['archive'] = dict(
            relative_map_error=float(np.linalg.norm(estimate - target_wire) / np.linalg.norm(target_wire)),
            column_error=float(np.linalg.norm((estimate - target_wire) @ DIRECTION)))
        report['summary'] = summarize(report['cases'])
        _write(path, report)
        print('exact archived Pong column', plan['index'], played['metrics']['rebounds'], flush=True)


def run(path, bank_path):
    torch.set_num_threads(1)
    archived = _read(SOURCE)
    native = _read(WARM)
    if not native['gate'] or archived['native_checkpoint_sha256'] != _hash(WARM):
        raise ValueError('native Pong policy checkpoint mismatch')
    weights = np.array(native['weights'])
    digest = _hash(SOURCE)
    plans = roster()
    report = dict(schema='tt24-pong-dense-setup-1', source_sha256=digest, motor_checkpoint_sha256=_hash(BUNDLE),
                  native_checkpoint_sha256=_hash(WARM), source_physical_ticks=PREFIX,
                  clock=CALIBRATION_SECONDS, protocol=PROTOCOL, target_reward_updates=0, plans=plans, cases=[])
    if path.exists():
        old = _read(path)
        for key in report:
            if key != 'cases' and digest_json(report[key]) != digest_json(old[key]):
                raise ValueError(f'cannot resume with a changed condition: {key}')
        report = old
    arm = stock_arm()
    server = BounceArena(view='none')
    policy = wrist_policy()
    try:
        for plan in plans[len(report['cases']):]:
            case = dispatcher(arm, server, policy, plan, archived, weights)
            report['cases'].append(case)
            report['summary'] = summarize(report['cases'])
            _write(path, report)
            print('dense receiving case', plan['index'], plan['learner_seed'],
                  [row['metrics']['rebounds'] for row in case['conditions'].values()],
                  [row['release_tick'] for row in case['conditions'].values()], flush=True)
        archive_trials(report, arm, server, policy, archived, path)
    finally:
        arm.arena.close()
        server.close()
    estimates = np.asarray([[case['conditions'][mode]['estimated_map'] for mode in ('pong', 'cold', 'archive')]
                            for case in report['cases']])
    np.savez_compressed(bank_path, schema=np.array('tt24-motor-bank-1'), maps=estimates[:, 0],
                        cold_maps=estimates[:, 1], archived_maps=estimates[:, 2],
                        fit_case_indices=np.arange(len(estimates)),
                        architecture=np.array('native Pong direction plus four calibrated directions'),
                        source_sha256=np.array(digest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--bank', type=Path, default=BANK)
    args = parser.parse_args()
    run(args.output, args.bank)


if __name__ == '__main__':
    main()
