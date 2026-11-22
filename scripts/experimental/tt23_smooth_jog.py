"""Select a smooth fixed-jog scorer using whole native Pong sources held out."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_anonymous_action import CHECK, OPTIONS, WIDTH
from .tt23_anonymous_replication import TRAIN
from .tt23_optical_contact import OUT


RECEIPT = OUT/'smooth-jog-source.json.gz'
NULL_DRAWS = 64
JOG = OPTIONS/max(abs(OPTIONS))
FAMILIES = ('linear_context', 'interaction', 'quartic')
PARAMETERS = tuple((family, ridge) for family in FAMILIES for ridge in (1., 8., 40.))


def load_sources():
    original, replications = _read(CHECK), _read(TRAIN)
    archives = [(1559000000, original['training']), *((source['offset'], source['rows'])
                                                       for source in replications['sets'])]
    if len(archives) != 5:
        raise ValueError('expected the original native source and four independent replications')
    groups, all_starts = [], set()
    for offset, group in archives:
        positions, durations, returns, identified, starts = [], [], [], [], []
        for runs in zip(*(group[str(index)] for index in range(7)), strict=True):
            reference = runs[0]
            if any(run['case'] != reference['case'] or run['context'] != reference['context'] or run['option'] != index
                   for index, run in enumerate(runs)):
                raise ValueError('physical source outcomes must share their scene through the motor decision')
            start = reference['case']['serve']
            if start in all_starts:
                raise ValueError('source starts intersect')
            all_starts.add(start)
            starts.append(start)
            context = reference['context']
            positions.append(context if context is not None else np.zeros(4))
            identified.append(context is not None)
            durations.append([run['ticks'] for run in runs])
            returns.append([bool(run['contacts'] and run['contacts'][0]['returned']) for run in runs])
        groups.append(dict(offset=offset, seeds=starts, x=np.asarray(positions), ticks=np.asarray(durations),
                           first=np.asarray(returns), seen=np.asarray(identified)))
    return groups


def features(x, center, scale, family):
    a, b, c, d = np.clip((x-center)/scale, -3., 3.).T
    odd = np.column_stack((a, b))
    even = np.column_stack((np.ones(len(x)), c, d))
    if family != 'linear_context':
        odd = np.column_stack((odd, a*c, a*d, b*c, b*d))
        even = np.column_stack((even, a*a, b*b, a*b, c*c, c*d, d*d))
    bases = [(odd, JOG), (even, JOG**2-np.mean(JOG**2))]
    if family == 'quartic':
        bases.extend(((odd, JOG**3), (even, JOG**4-np.mean(JOG**4))))
    return np.concatenate([context[:, None, :]*action[None, :, None] for context, action in bases], axis=2)


def design(x, family, ridge):
    center = np.r_[0., 0., x[:, 2:].mean(axis=0)]
    scale = np.maximum(.04, np.sqrt(np.mean((x-center)**2, axis=0)))
    held = features(x, center, scale, family)
    matrix = held.reshape(-1, held.shape[-1])
    reg = ridge*np.eye(matrix.shape[1])
    solver = np.linalg.solve(matrix.T@matrix+reg, matrix.T)
    return center, scale, solver


def reward(ticks):
    duration = np.log1p(ticks/120.)
    return duration-duration.mean(axis=1, keepdims=True)


def compare(ticks, first, seen, choices):
    selected = np.where(seen, choices, 3)
    index = np.arange(len(ticks))
    return dict(options=selected.tolist(), ticks=int(ticks[index, selected].sum()),
                first_returns=int(first[index, selected].sum()), points=len(ticks))


def source_kernel(train_x, labels, query):
    reflection = np.array([-1., -1., 1., 1.])
    position = np.concatenate((train_x, train_x*reflection))
    values = np.concatenate((labels, labels[:, ::-1]))
    distances = np.sum(((query[:, None]-position)/WIDTH)**2, axis=2)
    return np.exp(-.5*np.minimum(distances, 100.))@values


def mean_delta(records, reference):
    return float(np.mean([(a['ticks']-b['ticks'])/a['points']/120. for a, b in zip(records, reference, strict=True)]))


def run(output=RECEIPT):
    groups = load_sources()
    rng = np.random.default_rng(2095633630)
    permutation = []
    for group in groups:
        labels = reward(group['ticks'][group['seen']])
        order = rng.random((NULL_DRAWS, *labels.shape)).argsort(axis=2)
        permutation.append(np.take_along_axis(labels[None, :, :], order, axis=2))
    trials = {setting: [] for setting in PARAMETERS}
    null = {setting: [] for setting in PARAMETERS}
    control = {name: [] for name in ('neutral', 'kernel')}
    for i, withheld in enumerate(groups):
        donors = [group for j, group in enumerate(groups) if j != i]
        x = np.concatenate([group['x'][group['seen']] for group in donors])
        y = reward(np.concatenate([group['ticks'][group['seen']] for group in donors]))
        shuffled = np.concatenate([draw for j, draw in enumerate(permutation) if j != i], axis=1)
        shuffled = shuffled.reshape(NULL_DRAWS, -1)
        query, ticks, seen, returns = (withheld[key] for key in ('x', 'ticks', 'seen', 'first'))
        control['neutral'].append(compare(ticks, returns, seen, np.full(len(query), 3)))
        control['kernel'].append(compare(ticks, returns, seen, np.argmax(source_kernel(x, y, query), axis=1)))
        for family, ridge in PARAMETERS:
            center, scale, solver = design(x, family, ridge)
            feature_map = features(query, center, scale, family)
            fitted = solver@y.ravel()
            choice = np.argmax(feature_map@fitted, axis=1)
            trials[family, ridge].append(compare(ticks, returns, seen, choice))
            permuted = (feature_map @ (solver@shuffled.T)).argmax(axis=1)
            indexes = np.where(seen[:, None], permuted, 3)
            duration = ticks[np.arange(len(query))[:, None], indexes].mean(axis=0)/120.
            null[family, ridge].append(duration-ticks[:, 3].mean()/120.)
    neutral = control['neutral']
    chosen = max(PARAMETERS, key=lambda setting: mean_delta(trials[setting], neutral))
    reference = trials[chosen]
    deltas = np.asarray([mean_delta(trials[setting], neutral) for setting in PARAMETERS])
    max_null = np.max([np.mean(null[setting], axis=0) for setting in PARAMETERS], axis=0)
    gain = mean_delta(reference, neutral)
    wins = sum(r['ticks'] > n['ticks'] for r, n in zip(reference, neutral, strict=True))
    kernel = mean_delta(control['kernel'], neutral)
    empirical_p = (1+sum(max_null >= gain))/(1+len(max_null))
    gate = wins >= 4 and gain > max(.10, kernel) and empirical_p <= .10
    family, ridge = chosen
    x = np.concatenate([group['x'][group['seen']] for group in groups])
    y = reward(np.concatenate([group['ticks'][group['seen']] for group in groups]))
    center, scale, solver = design(x, family, ridge)
    fixed_rng = np.random.default_rng(1838522914)
    shuffled = np.take_along_axis(y, fixed_rng.random(y.shape).argsort(axis=1), axis=1)
    model = dict(family=family, ridge=ridge, center=center.tolist(), scale=scale.tolist(),
                 coefficients=(solver@y.ravel()).tolist(), shuffled=(solver@shuffled.ravel()).tolist())
    report = dict(schema='tt23-smooth-jog-source-1', sources={p.name: _hash(p) for p in (CHECK, TRAIN)},
                  starts=[dict(offset=g['offset'], seeds=g['seeds']) for g in groups], options=OPTIONS.tolist(),
                  model=model, control=control, gate=bool(gate), chosen_folds_won=wins,
                  gain_seconds=gain, kernel_gain_seconds=kernel, max_shuffle_p=float(empirical_p),
                  max_shuffled_duration_gains=max_null.tolist(), model_shuffled_gains=np.mean(null[chosen], axis=0).tolist(),
                  validation=[dict(family=setting[0], ridge=setting[1], folds=trials[setting], gain_seconds=float(delta))
                              for setting, delta in zip(PARAMETERS, deltas, strict=True)])
    _write(output, report)
    return report


def choose(model, query, *, shuffle=False):
    matrix = features(np.atleast_2d(query), np.asarray(model['center']), np.asarray(model['scale']), model['family'])
    return (matrix@np.asarray(model['shuffled' if shuffle else 'coefficients'])).argmax(axis=1)


if __name__ == '__main__':
    result = run()
    print({key: result[key] for key in ('gate', 'chosen_folds_won', 'gain_seconds', 'kernel_gain_seconds',
                                         'max_shuffle_p', 'model')}, flush=True)
    for entry in result['validation']:
        print(entry['family'], entry['ridge'], round(entry['gain_seconds'], 4),
              [round(fold['ticks']/fold['points']/120., 3) for fold in entry['folds']], flush=True)
