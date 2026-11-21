"""Test whether learned Pong relation relevance helps choose real SO-101 motions."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt20_camera_carry import SETTINGS, Station, verify
from .tt22_relation import BASE, Observation
from .tt22_relation_trial import OUTPUT, TT20, TT21, library


SEED = 737044646
DECISION = 9
LEADS = (0, 2, 4)
BUDGETS = (12, 24)
MODES = ('pong', 'score_shuffled', 'relation_shuffled', 'scratch')


class Values:
    """Use shared real practice and transfer only the metric on pair relationships."""

    def __init__(self, practice, prior=None):
        self.x = np.asarray([row['visual'] for row in practice])
        self.actions = np.asarray([row['option'] for row in practice])
        self.score = np.asarray([row['reward'] for row in practice])
        self.scale = np.maximum(.035, np.std(self.x, axis=0))
        relation = np.ones(self.x.shape[1] - BASE)
        if prior is not None:
            weights = np.abs(prior[BASE:])
            relation = .15 + weights / max(weights.mean(), 1e-7)
        self.weights = np.r_[np.full(BASE, .40 / BASE), .60 * relation / relation.sum()]

    def choose(self, cue):
        squared = np.square(np.clip((self.x - cue) / self.scale, -5., 5.))
        distance = squared @ self.weights
        close = .05 + .95 * np.exp(-distance / .70)
        count = np.bincount(self.actions, weights=close, minlength=len(LEADS))
        reward = np.bincount(self.actions, weights=close * self.score, minlength=len(LEADS))
        return int(np.argmax(reward / (.45 + count)))


class Play:
    """Common stock physics receiver with a choice of three forecast offsets."""

    def __init__(self, shared):
        self.shared = shared
        self.station = Station('left', ranging='temporal')
        self.arena, self.player = self.station.arena, self.station.player
        self.option = 0
        self.selection = self.player.player.choose
        self.player.player.choose = self.predict

    def predict(self, estimate):
        offset = LEADS[self.option]
        if offset:
            law = self.player.player.law
            p, v = law.trajectory(*estimate, offset, bounds=self.station.tracker.bounds)
            estimate = p[-1], v[-1]
        return self.selection(estimate)

    def close(self):
        self.station.close()

    def context(self, seed):
        self.arena.reset(seed=seed, server=-1, render=False)
        observer = Observation(self.shared)
        neutral = np.zeros(6)
        for tick in range(DECISION + 1):
            images = self.arena.render_cameras()
            visual = observer.update(images['left'], neutral if tick else None)
            if tick < DECISION:
                self.arena.step(neutral, render=False)
        return visual

    def point(self, seed, option, expected=None):
        self.arena.reset(seed=seed, server=-1, render=False)
        self.player.reset(-1)
        self.option = option
        self.station.tracker.registry.known.clear()
        camera = self.station.tracker.tracker()
        observer = Observation(self.shared)
        previous = None
        frames = sha256()
        actions = []
        for tick in range(self.arena.max_steps):
            images = self.arena.render_cameras()
            if tuple(images) != ('left',):
                raise ValueError('the target branch received an unexpected camera')
            frames.update(images['left'])
            if tick <= DECISION:
                visual = observer.update(images['left'], previous)
                if tick == DECISION and expected is not None and not np.array_equal(expected, visual):
                    raise ValueError('matched early visual histories changed')
            camera.update(images, tick=tick)
            ball = camera.ball
            point = ball.position if ball is not None and ball.observed else None
            command = self.player.command(tick, point, estimated_velocity=camera.ball_velocity)
            if tick < DECISION and np.any(command):
                raise ValueError('a stock motor action preceded the anonymous relation decision')
            actions.append(command.tolist())
            _, reward, finished, capped, _ = self.arena.step(command, render=False)
            previous = command
            if finished or capped:
                break
        return dict(seed=seed, role=-1, option=option, visual=visual.tolist(), reward=reward, actions=actions,
                    outcome=self.arena.point_result(), events=self.arena.events.copy(), rgb_sha256=frames.hexdigest())


def _metrics(rows, budgets):
    result = {}
    for budget in map(str, budgets):
        stats = {}
        for mode in MODES:
            tapes = [row['options'][str(row['selected'][budget][mode])] for row in rows]
            returns = [row['outcome']['legal_returns'] > 0 for row in tapes]
            options = [row['selected'][budget][mode] for row in rows]
            stats[mode] = dict(first_returns=sum(returns), wins=sum(row['reward'] > 0 for row in tapes),
                               choices=np.bincount(options, minlength=len(LEADS)).tolist())
            if mode != 'pong':
                genuine = [row['options'][str(row['selected'][budget]['pong'])]['outcome']['legal_returns'] > 0
                           for row in rows]
                gain = sum(a and not b for a, b in zip(genuine, returns, strict=True))
                loss = sum(b and not a for a, b in zip(genuine, returns, strict=True))
                stats[mode]['pong_paired_return_gain_loss'] = [gain, loss]
                stats[mode]['pong_paired_exact_p'] = float(binomtest(gain, gain + loss).pvalue) if gain + loss else 1.
        result[budget] = stats
    return result


def run(folder, count, *, cache=False):
    forecast_path = folder / 'relation-forecast.json.gz'
    forecast = _read(forecast_path)
    prior = {mode: np.asarray(weights) for mode, weights in forecast['prior']['terminal'].items()}
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(2**31, size=max(BUDGETS) + count).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds) & (previous_seeds(TT20) | previous_seeds(TT21)):
        raise ValueError('the action schedule overlaps earlier experiments')
    practice_seed, assessment = seeds[:max(BUDGETS)], seeds[max(BUDGETS):]
    order = np.concatenate([rng.permutation(len(LEADS)) for _ in range(len(practice_seed) // len(LEADS))])
    practice_path = folder / 'relation-action-practice.json.gz'
    with TemporaryDirectory() as temp:
        sim = Play(library(temp))
        try:
            if cache:
                practice = _read(practice_path)['points']
                if [row['seed'] for row in practice] != practice_seed:
                    raise ValueError('the saved practice uses another schedule')
            else:
                practice = []
                for seed, option in zip(practice_seed, order.tolist(), strict=True):
                    row = sim.point(seed, option)
                    practice.append(row)
                    print('practice', len(practice), seed, option, row['outcome']['legal_returns'], row['reward'], flush=True)
                _write(practice_path, dict(schema='tt22-relation-action-practice-1', leads=LEADS,
                                           points=practice, decision_tick=DECISION, seed=SEED))
            learners = {str(budget): {
                mode: Values(practice[:budget], prior.get(mode)) for mode in MODES} for budget in BUDGETS}
            rows = []
            for seed in assessment:
                cue = sim.context(seed)
                selected = {budget: {mode: learner.choose(cue) for mode, learner in branch.items()}
                            for budget, branch in learners.items()}
                options = sorted({option for branch in selected.values() for option in branch.values()})
                played = {str(option): sim.point(seed, option, cue) for option in options}
                rows.append(dict(seed=seed, selected=selected, options=played))
                print('held', len(rows), seed, selected, {
                    option: row['outcome']['legal_returns'] for option, row in played.items()}, flush=True)
        finally:
            sim.close()
    all_tapes = [branch for row in rows for branch in row['options'].values()]
    verified = verify(dict(arena=SETTINGS, eye='left', points=practice + all_tapes, strong=False))
    result = dict(schema='tt22-relation-action-1', seed=SEED, leads=LEADS, decision_tick=DECISION, budgets=BUDGETS,
                  arena=SETTINGS, eye='left', practice_sha256=_hash(practice_path), forecast_sha256=_hash(forecast_path),
                  held_seeds=assessment, physical_held_tapes=len(all_tapes), verified_tapes=verified,
                  rows=rows, metrics=_metrics(rows, BUDGETS))
    _write(folder / 'relation-action.json.gz', result)
    print(result['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--held-starts', type=int, default=16)
    parser.add_argument('--cache-practice', action='store_true')
    args = parser.parse_args()
    run(args.folder, args.held_starts, cache=args.cache_practice)
