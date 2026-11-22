"""Audit the contribution of native duration labels to the fixed online jog bank."""

from unittest.mock import patch

import numpy as np

from . import tt23_implicit_online as online
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import ROLE_MODEL
from .tt23_anonymous_action import CHECK
from .tt23_anonymous_replication import TRAIN
from .tt23_direction import MIRROR, SOURCE, durations, marginal_votes, pick, policy
from .tt23_direction_permutation import kernel


RESULTS = online.OUT/'implicit-online-results.json.gz'
RECEIPT = online.OUT/'implicit-permutation.json.gz'
JOINT_DRAWS = 4096
SEED = 2011018763
METRICS = ('launches_capped', 'arm')


class StreamScores:
    """Replay the stock online learner once per distinct eight-launch prior pattern."""

    def __init__(self, bank, stream):
        self.bank = bank
        self.uniforms = online.paired_draws(90143421 + stream)
        self.contexts = [None if case['context'] is None else np.asarray(case['context']) for case in bank['cases']]
        self._scores = {}

    def pattern(self, model, mode):
        return tuple(pick(x, model, mode) for x in self.contexts)

    def score(self, pattern):
        if pattern not in self._scores:
            by_context = {}
            for context, action in zip(self.contexts, pattern, strict=True):
                if context is not None and by_context.setdefault(tuple(context), action) != action:
                    raise ValueError('a directional fit cannot give different priors to the same context')

            def frozen_pick(context, _model, _mode):
                return by_context[tuple(context)]

            with patch.object(online, 'pick', frozen_pick):
                self._scores[pattern] = online.score(self.bank, None, 'learned', self.uniforms)
        return self._scores[pattern]

    def summarize_patterns(self, patterns):
        unique, inverse, counts = np.unique(patterns, axis=0, return_inverse=True, return_counts=True)
        values = [self.score(tuple(pattern)) for pattern in unique]
        summaries = [dict(pattern=pattern.tolist(), count=int(count), **score)
                     for pattern, count, score in zip(unique, counts, values, strict=True)]
        return np.array([[row[key] for key in METRICS] for row in values])[inverse], summaries


def fixed_banks(receipt):
    seen = set()
    streams = []
    for index, reference in enumerate(receipt['streams']):
        path = online.OUT/f'jog-bank-{index + 1}.json.gz'
        bank = _read(path)
        seeds = {row['seed'] for row in bank['cases']}
        if (_hash(path) != reference['bank'] or not bank['complete'] or bank['verified'] != 24
                or len(bank['cases']) != 8 or len(seeds) != 8 or seen & seeds
                or any(row['verified'] != 3 for row in bank['cases'])):
            raise ValueError(f'expected the archived verified 24 options for stream {index + 1}')
        seen.update(seeds)
        streams.append(StreamScores(bank, index))
    return streams


def prepare_source(source):
    if any(source['sources'][path.name] != _hash(path) for path in (CHECK, TRAIN, ROLE_MODEL)):
        raise ValueError('the frozen native source inputs changed')
    if source['width'] != online.WIDTH.tolist():
        raise ValueError('the frozen source kernel widths changed')
    old, replicated = _read(CHECK), _read(TRAIN)
    archives = [(1559000000, old['training']), *((row['offset'], row['rows']) for row in replicated['sets'])]
    packed, seen = [], set()
    for (offset, archive), fit in zip(archives, source['sets'], strict=True):
        x, ticks, seeds = durations(archive)
        if offset != fit['offset'] or seeds != fit['starts'] or len(seeds) != len(set(seeds)) or seen & set(seeds):
            raise ValueError('native source launches must match five disjoint archived fits')
        seen.update(seeds)
        actual = marginal_votes(ticks)
        fixed = marginal_votes(np.take_along_axis(ticks, np.asarray(fit['shuffle_indexes']), axis=1))
        if policy(x, actual) != fit['learned'] or policy(x, fixed) != fit['shuffled']:
            raise ValueError('could not recreate a frozen real or fixed shuffled source fit')
        packed.append((offset, x, ticks, actual, fixed))
    x = np.concatenate([row[1] for row in packed])
    for mode, column in (('learned', 3), ('shuffled', 4)):
        if policy(x, np.concatenate([row[column] for row in packed])) != source['pooled'][mode]:
            raise ValueError('could not recreate the pooled source fit')
    return packed


def reproduction(stream, model, mode, expected):
    pattern = stream.pattern(model, mode)
    exact = online.score(stream.bank, model, mode, stream.uniforms, retention='native_after' in expected)
    if exact != expected or stream.score(pattern) != {k: v for k, v in expected.items() if k != 'native_after'}:
        raise ValueError(f'the archived {mode} results or pattern replay changed')
    return dict(pattern=list(pattern), **exact)


def directional_weights(contexts, x):
    visible = np.array([context is not None for context in contexts])
    query = np.asarray([context if context is not None else np.zeros(x.shape[1]) for context in contexts])
    local, mirrored = kernel(query, x), kernel(query, x * MIRROR)
    threshold = 1.e-12 * ((local + mirrored).sum(axis=1) + .7)
    return local - mirrored, threshold, visible


def directions(weights, threshold, visible, votes):
    predicted = votes @ weights.T
    choices = np.where(predicted > threshold, 4, np.where(predicted <= -threshold, 2, 3))
    choices[:, ~visible] = 3
    return choices


def shuffled_votes(offset, ticks):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, offset]))
    order = rng.random((JOINT_DRAWS, *ticks.shape)).argsort(axis=2)
    return marginal_votes(np.take_along_axis(ticks[None], order, axis=2).reshape(-1, ticks.shape[1])).reshape(
        JOINT_DRAWS, len(ticks))


def metrics(rows):
    return np.mean([[row[key] for key in METRICS] for row in rows], axis=0)


def tail(match, beat):
    n = int(match.sum())
    return dict(match_or_beat=n, strictly_better=int(beat.sum()), fraction=n / JOINT_DRAWS,
                monte_carlo_tail=(1 + n) / (1 + JOINT_DRAWS))


def run():
    receipt, source = _read(RESULTS), _read(SOURCE)
    if receipt['source'] != _hash(SOURCE) or receipt['draws'] != online.DRAWS or len(receipt['streams']) != 4:
        raise ValueError('expected the completed standard online analysis')
    streams = fixed_banks(receipt)
    archives = prepare_source(source)
    if len(archives) != 5:
        raise ValueError('expected five independently trained native fits')
    contexts = [x for stream in streams for x in stream.contexts]
    references = receipt['streams']
    pooled = {mode: [reproduction(stream, source['pooled'], mode, reference['rows'][mode])
                     for stream, reference in zip(streams, references, strict=True)] for mode in ('learned', 'shuffled')}
    if any(any(left[key] != right[key] for key in METRICS + ('pattern',))
           for left, right in zip(pooled['learned'], pooled['shuffled'], strict=True)):
        raise ValueError('expected the fixed pooled shuffled fit to tie the real fit')

    observed_rows, null_rows, fit_records = [], [], []
    for number, ((offset, x, ticks, actual, fixed), source_fit) in enumerate(zip(archives, source['sets'], strict=True)):
        by_mode = {mode: [reproduction(stream, source_fit, mode, reference['independent_source_runs'][number][mode])
                          for stream, reference in zip(streams, references, strict=True)]
                   for mode in ('learned', 'shuffled')}
        observed_rows.extend(by_mode['learned'])
        weight, threshold, visible = directional_weights(contexts, x)
        actual_and_fixed = directions(weight, threshold, visible, np.stack((actual, fixed)))
        for i, mode in enumerate(by_mode):
            if actual_and_fixed[i].tolist() != [p for row in by_mode[mode] for p in row['pattern']]:
                raise ValueError('vectorized prediction changed a frozen source prior pattern')
        votes = shuffled_votes(offset, ticks)
        pattern_draws = directions(weight, threshold, visible, votes)
        for index in range(8):
            equivalent = dict(learned=policy(x, votes[index]))
            if pattern_draws[index].tolist() != [pick(context, equivalent, 'learned') for context in contexts]:
                raise ValueError('vectorized prediction changed an independently shuffled prior pattern')
        per_stream, patterns = [], []
        for index, stream in enumerate(streams):
            scores, summaries = stream.summarize_patterns(pattern_draws[:, index * 8:(index + 1) * 8])
            per_stream.append(scores)
            patterns.append(summaries)
        null_rows.extend(per_stream)
        null_per_fit = np.mean(per_stream, axis=0)
        fit_records.append(dict(offset=offset, scenes=len(x), learned=by_mode['learned'],
                                fixed_shuffled=by_mode['shuffled'], shuffled_patterns=patterns,
                                null_mean=dict(zip(METRICS, null_per_fit.mean(axis=0).tolist(), strict=True))))
        print(offset, 'unique patterns by stream', list(map(len, patterns)), flush=True)

    observed = metrics(observed_rows)
    null = np.mean(null_rows, axis=0)
    matches = np.c_[null[:, 0] <= observed[0], null[:, 1] >= observed[1]]
    beats = np.c_[null[:, 0] < observed[0], null[:, 1] > observed[1]]
    tails = {key: tail(matches[:, i], beats[:, i]) for i, key in enumerate(METRICS)}
    tails['both'] = tail(matches.all(axis=1), beats.all(axis=1))
    summary = dict(observed=dict(zip(METRICS, observed.tolist(), strict=True)),
                   pooled={mode: dict(zip(METRICS, metrics(rows).tolist(), strict=True)) for mode, rows in pooled.items()},
                   flat=dict(zip(METRICS, metrics([row['rows']['flat'] for row in references]).tolist(), strict=True)),
                   null={key: dict(mean=float(null[:, i].mean()), p05_p50_p95=np.quantile(null[:, i], [.05, .5, .95]).tolist())
                         for i, key in enumerate(METRICS)}, tails=tails)
    result = dict(schema='tt23-implicit-permutation-1', analysis_sha256=_hash(RESULTS), source_sha256=_hash(SOURCE),
                  banks=[dict(sha256=row['bank'], seeds=[point['seed'] for point in stream.bank['cases']])
                         for row, stream in zip(references, streams, strict=True)], exploration_draws=online.DRAWS,
                  joint_permutations=JOINT_DRAWS, seed=SEED, metrics=list(METRICS), summary=summary,
                  pooled=pooled, independent_fits=fit_records, permutation_means=null.tolist(),
                  cached_patterns=[len(stream._scores) for stream in streams])
    _write(RECEIPT, result)
    print(summary, flush=True)


if __name__ == '__main__':
    run()
