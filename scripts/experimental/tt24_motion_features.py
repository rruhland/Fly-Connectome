"""Learn a contextual velocity choice in real Pong before a frozen wrist test."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.special import expit
import torch

from fly_connectome.pong import Physics, Pong

from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_joint_pong import CAP, DIRECTIONS, HEIGHT, STRIDE, WARM, WIDTH, field, native_reward, preference
from .tt24_long_wrist import play
from .tt24_tool_equilibrium import ToolEquilibrium


TAG = 'motion-features'
PLAN = ROOT / f'{TAG}-plan.json'
SOURCE = ROOT / f'{TAG}-native.json.gz'
WRIST = ROOT / f'{TAG}-wrist.json.gz'
SEED = 1630057810
LEARN_SEED = 451002297
UPDATES = 8
SHAPE = 5
TIMES = (.055, .110)
FIXED = {'ordinary': 0., 'half': .5, 'fast': 1.}
GROUPS = ('trained', 'shuffled')
WRIST_CAP = 360


def context(offset, blade_velocity, fresh, normal, radius, scale):
    """Rotation-invariant relationship of the mover and controllable blade."""
    gap = offset / radius
    own = blade_velocity / scale
    relative = fresh / scale - own
    fields = np.column_stack((np.ones(len(gap)), np.linalg.norm(own, axis=1),
                             np.sum(gap * relative, axis=1), np.sum(gap * own, axis=1), np.abs(normal)))
    return np.clip(fields, -2., 2.)


@torch.no_grad()
def native_points(readout, selection, seeds, time):
    count = len(seeds) * len(selection)
    game = Pong(seeds * len(selection), Physics(opponent_speed=.55, tau_activation=time))
    options = np.repeat(np.arange(len(selection)), len(seeds))
    constant = np.array([mode if np.isscalar(mode) else -1. for mode in selection])[options]
    rules = np.array([np.zeros(SHAPE) if np.isscalar(mode) else mode for mode in selection])[options]
    old = np.zeros((count, 2))
    motion = np.zeros_like(old)
    prior_visible = np.zeros(count, dtype=bool)
    active = np.ones(count, dtype=bool)
    score, hits, frames, u, paddle_speed, blends, samples = [np.zeros(count) for _ in range(7)]
    for tick in range(CAP):
        if tick % STRIDE == 0:
            old_position, last_seen = old.copy(), prior_visible.copy()
            paddle, seen = field(game.render(HEIGHT, WIDTH).numpy(), old, motion, prior_visible)
            if tick == 0:
                home = paddle.copy()
                previous_paddle = home.copy()
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (120 / STRIDE)
            raw_gap = old[:, 1] - paddle
            fresh = (old[:, 1] - old_position[:, 1]) * (120 / STRIDE)
            features = context(raw_gap[:, None], paddle_speed[:, None], fresh[:, None], motion[:, 0] / .55, .10, .60)
            balance = np.where(constant < 0., expit(np.sum(features * rules, axis=1)), constant)
            balance[~(seen & last_seen)] = 0.
            blends += balance * active
            samples += active
            tangential = motion[:, 1] + balance * (fresh - motion[:, 1])
            gap = np.clip(raw_gap * .76 / .065, -3., 3.)
            tangent = np.clip(tangential * 3. / .38, -3., 3.)
            pose = np.clip((paddle - home) * .76 / .04, -3., 3.)
            normal = np.clip(motion[:, 0] * 3. / 1.5, -3., 3.)
            input_ = np.column_stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)))
            offset = .36 * np.tanh((input_ * readout).sum(axis=1))
            request = np.clip(-11. * (home + offset - paddle) + 1.35 * paddle_speed, -1., 1.)
            u[seen] = request[seen]
            previous_paddle = paddle
        outcome = game.step(torch.from_numpy(u.astype(np.float32)), auto_reset=False)
        terminal = outcome.scores.numpy()
        frames += active
        hits += outcome.hits.numpy() * active
        score += terminal * active
        active &= terminal == 0.
        if not active.any():
            break
    return (np.column_stack((score, hits, frames)).reshape(len(selection), len(seeds), 3),
            (blends / np.maximum(1., samples)).reshape(len(selection), len(seeds)))


def register():
    rng = np.random.default_rng(SEED)
    launch = (rng.choice(2**31 - 2, UPDATES * 12 + 96, replace=False) + 1).tolist()
    training = np.array(launch[:UPDATES*12]).reshape(UPDATES, 12).tolist()
    held = [launch[-96:-48], launch[-48:]]
    physical = [item for item in schedule(1771163007, 6) if item['body'] == 'alternate']
    for item in physical:
        item['changes'] = [list(pair) for pair in item['changes']]
    data = dict(schema='tt24-motion-feature-choice-1', seed=SEED, learning_seed=LEARN_SEED, native_train=training,
                native_held=held, motor_response_times=list(TIMES), scalar_native_reward='terminal + .43 elapsed fraction',
                feature_choice='fresh two-frame velocity or existing smoothed velocity mixed by contextual reward-learned logistic',
                native_gate='at least .012 higher pooled payoff than every fixed mode and shuffled, with neither profile falling .04 below ordinary',
                native_actor_sha256=_hash(WARM), carried_wrist_sha256=_hash(BUNDLE), wrist=physical,
                wrist_scored_learning_attempts=0, wrist_cap=WRIST_CAP)
    if PLAN.exists():
        if _read(PLAN) != data:
            raise ValueError('registered contextual feature test changed')
    else:
        _write(PLAN, data)
    return data


def fit(plan):
    if SOURCE.exists():
        saved = _read(SOURCE)
        if saved['plan_sha256'] != _hash(PLAN):
            raise ValueError('native feature source differs from its registration')
        return saved
    readout = np.asarray(_read(WARM)['weights'])
    rng = np.random.default_rng(LEARN_SEED)
    modes = np.tile(np.array((-1.2, 0., 0., 0., 0.)), (2, 1))
    history = []
    for index, seeds in enumerate(plan['native_train']):
        directions = rng.normal(size=(DIRECTIONS, SHAPE))
        variants = np.r_[np.zeros((1, SHAPE)), .28 * directions, -.28 * directions]
        population = np.clip(modes[:, None] + variants, -3.5, 3.5).reshape(-1, SHAPE)
        result, mixing = native_points(readout, population, seeds, TIMES[index % 2])
        payoffs = native_reward(result).reshape(2, -1)
        signed = np.array([preference(q) for q in payoffs])
        exchange = rng.choice((-1., 1.), DIRECTIONS)
        signed[1] *= exchange
        modes = np.clip(modes + .24 * np.mean(signed[..., None] * directions[None], axis=1), -3.5, 3.5)
        history.append(dict(outcomes=result.tolist(), blend=mixing.mean(axis=1).tolist(),
                            payoffs=payoffs.tolist(), shuffled_signs=exchange.tolist(), fit=modes.tolist()))
        print('native feature update', index + 1, np.round(modes, 3).tolist(), flush=True)
    frozen = dict(zip(GROUPS, modes.tolist(), strict=True)) | FIXED
    qualification = []
    for time, seeds in zip(TIMES, plan['native_held'], strict=True):
        result, mixing = native_points(readout, list(frozen.values()), seeds, time)
        qualification.append(dict(time=time, points={name: values.tolist() for name, values in zip(frozen, result, strict=True)},
                                  payoff={name: float(v) for name, v in zip(frozen, native_reward(result), strict=True)},
                                  mean_mixing={name: float(v) for name, v in zip(frozen, mixing.mean(1), strict=True)}))
    scores = {name: float(np.mean([row['payoff'][name] for row in qualification])) for name in frozen}
    reference = max(FIXED, key=scores.get)
    gate = (scores['trained'] >= .012 + max(scores[name] for name in frozen if name != 'trained') and
            all(row['payoff']['trained'] >= row['payoff']['ordinary'] - .04 for row in qualification))
    record = dict(plan_sha256=_hash(PLAN), fitted=frozen, source_updates=history, held=qualification, means=scores,
                  fixed_reference=reference, advance_to_wrist=gate, wrist_examples_at_fit=0)
    _write(SOURCE, record)
    print('LOCKED NATIVE FEATURES', scores, 'wrist gate', gate, fetch_summary(qualification), flush=True)
    return record


def fetch_summary(rows):
    return [{name: int((np.array(values)[:, 1] > 0).sum()) for name, values in row['points'].items()} for row in rows]


class BlendObserver:
    def __init__(self, eye, rule):
        self.eye = eye
        self.decoder = eye.decoder
        self.rule = rule
        self.previous = None
        self.history = []

    @property
    def home(self):
        return self.eye.home

    @home.setter
    def home(self, value):
        self.eye.home = value

    def observe(self, observation):
        lost = self.eye.gaps
        state = self.eye.observe(observation)
        rate = 0.
        if self.eye.gaps == lost:
            world_ball, world_blade = state[:3] + self.home, state[6:9] + self.home
            if self.previous is not None:
                before_ball, before_blade = self.previous
                fresh = np.clip((world_ball[:2] - before_ball[:2]) * 60., -1.8, 1.8)
                own = (world_blade[:2] - before_blade[:2]) * 60.
                cue = context((world_ball[:2]-world_blade[:2])[None], own[None], fresh[None], [state[5]/1.5], .076, .42)[0]
                rate = float(self.rule if np.isscalar(self.rule) else expit(cue @ self.rule))
                state[3:5] += rate * (fresh - state[3:5])
            self.previous = world_ball, world_blade
        else:
            self.previous = None
        self.history.append(rate)
        return state


class Wrist(ToolEquilibrium):
    def __init__(self, rule):
        super().__init__(BUNDLE, .0557)
        self.mode = 'workspace_local'
        self.rule = rule

    def begin_release(self, observation):
        super().begin_release(observation)
        if not np.isscalar(self.rule) or self.rule != 0.:
            self.sensor = BlendObserver(self.sensor, self.rule)


def test_body(task):
    plan, frozen = task
    installation = InstalledArm(plan['body'])
    installation.arena.max_steps = WRIST_CAP
    server = BounceArena(view='none')
    try:
        server.reset(seed=plan['launcher_seed'], split='held', render=False)
        launch = tuple(item.copy() for item in (server._ball_position, server._ball_velocity, server._ball_spin))
        tested = {}
        for name, rule in frozen.items():
            controller = Wrist(rule)
            observed = play(installation, controller, plan, launch, 'live')
            replayed = play(installation, controller, plan, launch, 'live', observed['actions'])
            if digest_json(observed) != digest_json(replayed):
                raise ValueError('altered-wrist replay changed')
            weights = controller.sensor.history if isinstance(controller.sensor, BlendObserver) else [0.] * len(observed['actions'])
            tested[name] = observed | dict(mixing=weights)
        print('motion-feature wrist', plan['learner_seed'], {k: v['metrics']['longest_consecutive_rebounds']
                                                           for k, v in tested.items()}, flush=True)
        return dict(plan=plan, results=tested)
    finally:
        server.close()
        installation.arena.close()


def wrist(plan, source, workers):
    rules = {name: source['fitted'][name] for name in ('trained', 'shuffled', 'ordinary', source['fixed_reference'])}
    result = _read(WRIST) if WRIST.exists() else dict(plan_sha256=_hash(PLAN), source_sha256=_hash(SOURCE), trials=[])
    if result['plan_sha256'] != _hash(PLAN) or result['source_sha256'] != _hash(SOURCE):
        raise ValueError('wrist motion features do not match the frozen source')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as workers:
        for trial in workers.map(test_body, [(p, rules) for p in plan['wrist'][len(result['trials']):]]):
            result['trials'].append(trial)
            _write(WRIST, result)
    result['streaks'] = {name: [p['results'][name]['metrics']['longest_consecutive_rebounds'] for p in result['trials']]
                         for name in rules}
    _write(WRIST, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = register()
    source = fit(plan)
    if source['advance_to_wrist']:
        wrist(plan, source, args.workers)


if __name__ == '__main__':
    main()
