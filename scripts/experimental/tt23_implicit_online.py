"""Train a camera-conditioned selector among three fixed jogs on scored point duration."""

import argparse

import numpy as np
from scipy.special import betaincinv

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import STREAMS
from .tt22_joint_practice import draw
from .tt23_anonymous_action import WIDTH
from .tt23_anonymous_target import target_context
from .tt23_direction import MIRROR, NATIVE, SOURCE, pick
from .tt23_jog_station import ExplicitJogStation
from .tt23_optical_contact import OUT
from .tt23_sequence import summarize, verify

NAMES = ('left', 'center', 'right')
PRIOR_MODES = ('learned', 'flat', 'shuffled', 'reversed')
LIVE_SEED = 1138013972
DRAWS = 256


class JogLearner:
    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self.history = []
        self.points = 0

    def prior(self, x):
        means = np.full(3, .5)
        if self.mode == 'flat' or x is None:
            return means
        direction = pick(x, self.model, self.mode) - 2
        if direction != 1:
            means[direction], means[2 - direction] = .75, .25
        return means

    def choose(self, x, draws, *, greedy=False):
        n = len(draws)
        if x is None:
            return np.ones(n, dtype=int)
        means = self.prior(x)
        if self.points == 0:
            choice = 1 if np.ptp(means) == 0 else means.argmax()
            return np.full(n, choice, dtype=int)
        a = np.tile(2 * means, (n, 1))
        b = 2 - a
        indices = np.arange(n)
        for past, chosen, reward in self.history:
            for key, action in ((past, chosen), (past * MIRROR, 2 - chosen)):
                weight = .5 * np.exp(-.5 * np.sum(((x - key) / WIDTH) ** 2))
                a[indices, action] += weight * reward
                b[indices, action] += weight * (1 - reward)
        if greedy:
            posterior = a / (a + b)
            chosen = posterior.argmax(axis=1)
            chosen[posterior[:, 1] == posterior.max(axis=1)] = 1
            return chosen
        return betaincinv(a, b, draws).argmax(axis=1)

    def update(self, x, chosen, ticks):
        self.points += 1
        if x is not None:
            utility = np.minimum(np.asarray(ticks, dtype=float) / 300., 1.)
            self.history.append((np.asarray(x), chosen.copy(), utility))


def paired_draws(seed, n=DRAWS):
    return np.random.default_rng(seed).uniform(1.e-7, 1. - 1.e-7, (8, n, 3))


def play_live():
    model = _read(SOURCE)['pooled']
    learners = {key: JogLearner(model, key) for key in ('learned', 'flat')}
    draws = paired_draws(LIVE_SEED + 1, 1)
    rng = np.random.default_rng(LIVE_SEED)
    strata = [draw(rng, 4, band) for band in STREAMS.values()]
    schedule = [(seed, lane) for pair in zip(*strata, strict=True) for seed, lane in pair]
    path = OUT / 'implicit-online-live.json.gz'
    record = _read(path) if path.exists() else dict(source=_hash(SOURCE), cases=[])
    if record['source'] != _hash(SOURCE):
        raise ValueError('The source direction fit changed')
    decisions = {}

    def jog(mode, sight, ready):
        x = target_context(sight, ready)
        action = learners[mode].choose(x, draws[i])
        decisions[mode] = x, action
        return .09 * (int(action[0]) - 1)

    station = ExplicitJogStation(jog)
    try:
        for i, (seed, lane) in enumerate(schedule):
            if i == len(record['cases']):
                record['cases'].append(dict(seed=seed, lane=lane, results={}, choices={}))
            case = record['cases'][i]
            if (case['seed'], case['lane']) != (seed, lane):
                raise ValueError('Schedule changed')
            for mode, learner in learners.items():
                if mode in case['results']:
                    row = case['results'][mode]
                    sight = row['opening']
                    if sight is None or sight['mover'] is None:
                        x, action = None, np.ones(1, dtype=int)
                    else:
                        x = np.array(case['context'])
                        action = learner.choose(x, draws[i])
                    if int(action[0]) != case['choices'][mode]:
                        raise ValueError('Cannot regenerate selected action')
                else:
                    decisions.pop(mode, None)
                    if mode == 'flat':
                        x = None if case.get('context') is None else np.array(case['context'])
                        matched = learner.choose(x, draws[i])
                    else:
                        matched = None
                    if matched is not None and int(matched[0]) == case['choices']['learned']:
                        row = case['results']['learned']
                        action = matched
                    else:
                        row = station.play(seed, lane, mode)
                        x, action = decisions.get(mode, (None, np.ones(1, dtype=int)))
                    context = None if x is None else x.tolist()
                    if 'context' in case and context != case['context']:
                        raise ValueError('The paired conditions diverged before the selected action')
                    case['context'] = context
                    case['choices'][mode] = int(action[0])
                    case['results'][mode] = row
                    _write(path, record)
                    print(i + 1, mode, NAMES[int(action[0])], row['longest'], flush=True)
                learner.update(x, action, [len(row['actions'])])
        record['verified'] = verify(record['cases'])
        record['summary'] = {key: summarize([case['results'][key] for case in record['cases']]) for key in learners}
        _write(path, record)
    finally:
        station.close()


def native_retention(learner, n):
    before = JogLearner(learner.model, learner.mode)
    per_agent = np.zeros((n, 3))
    native = _read(NATIVE)['rows']
    stub = np.full((n, 3), .5)
    for row in native:
        context = None if row['context'] is None else np.asarray(row['context'])
        now = learner.choose(context, stub, greedy=True)
        original = before.choose(context, stub, greedy=True)
        choices = row['options'][2:5]
        per_agent[:, 0] += np.asarray([trial['ticks'] for trial in choices])[now]
        per_agent[:, 1] += np.asarray([bool(trial['contacts'] and trial['contacts'][0]['returned'])
                                       for trial in choices])[now]
        per_agent[:, 2] += now != original
    return dict(mean_native_seconds=float(per_agent[:, 0].mean() / (120 * len(native))),
                first_returns=float(per_agent[:, 1].mean()), changed_actions=float(per_agent[:, 2].mean()))


def score(bank, model, mode, uniforms, *, retention=False):
    cases = bank['cases']
    replications = len(uniforms[0])
    elapsed = np.zeros(replications, dtype=int)
    first_time = np.full(replications, -1)
    first_point = np.full(replications, 9)
    ten_points = np.zeros(replications)
    twenty_points = np.zeros(replications)
    arm = np.zeros(replications)
    learner = JogLearner(model, 'learned' if mode == 'frozen' else mode)
    for i, case in enumerate(cases):
        context = case['context']
        x = None if context is None else np.array(context)
        if mode == 'center':
            action = np.ones(replications, dtype=int)
        elif mode == 'frozen':
            direction = pick(x, model, 'learned') - 2
            action = np.full(replications, direction, dtype=int)
        else:
            action = learner.choose(x, uniforms[i])
        physical = [case['results'][key] for key in NAMES]
        ticks = np.array([len(row['actions']) for row in physical])[action]
        milestones = np.array([row['milestones'].get('10', -1) for row in physical])[action]
        successes = milestones >= 0
        new = (first_time < 0) & successes
        first_time[new] = elapsed[new] + milestones[new]
        first_point[new] = i + 1
        ten_points += successes
        twenty_points += np.array([row['longest'] >= 20 for row in physical])[action]
        arm += np.array([row['outcome']['legal_returns'] for row in physical])[action]
        elapsed += ticks
        learner.update(x, action, ticks)
    success = first_time >= 0
    result = dict(chance_ten=float(success.mean()), launches_capped=float(first_point.mean()),
                by_40_seconds=float((success & (first_time <= 1200)).mean()),
                ended_early=float((~success & (elapsed < 1200)).mean()),
                ten_points=float(ten_points.mean()), twenty_points=float(twenty_points.mean()),
                arm=float(arm.mean()))
    if retention:
        result['native_after'] = native_retention(learner, replications)
    return result


def analyze():
    source = _read(SOURCE)
    streams = []
    seeds = set()
    for i in range(4):
        path = OUT / f'jog-bank-{i+1}.json.gz'
        bank = _read(path)
        if len(bank['cases']) != 8 or bank.get('verified') != 24:
            raise ValueError('The three physical choices must each pass replay on all eight starts')
        starts = {case['seed'] for case in bank['cases']}
        if len(starts) != 8 or starts & seeds:
            raise ValueError('Repeated physical start')
        seeds.update(starts)
        uniforms = paired_draws(90143421 + i)
        rows = {mode: score(bank, source['pooled'], mode, uniforms, retention=mode in ('learned', 'flat'))
                for mode in (*PRIOR_MODES, 'frozen', 'center')}
        runs = [{k: score(bank, model, k, uniforms) for k in ('learned', 'shuffled')}
                for model in source['sets']]
        lengths = np.array([[case['results'][name]['longest'] for name in NAMES] for case in bank['cases']])
        attainable = lengths.max(axis=1)
        first = np.flatnonzero(attainable >= 10)
        ceiling = dict(point_lengths=attainable.tolist(), ten=int(sum(attainable >= 10)),
                       twenty=int(sum(attainable >= 20)), first_ten=None if not len(first) else int(first[0]+1))
        streams.append(dict(bank=_hash(path), rows=rows, independent_source_runs=runs, evaluator_ceiling=ceiling))
        print(i + 1, rows, flush=True)
    _write(OUT / 'implicit-online-results.json.gz', dict(source=_hash(SOURCE), streams=streams, draws=DRAWS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    play_live() if args.live else analyze()
