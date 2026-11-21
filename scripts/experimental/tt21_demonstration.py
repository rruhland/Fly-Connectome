"""Distill a small receiver repertoire from native, legally credited motor demonstrations."""

from collections import defaultdict

import numpy as np

from .tt4_transfer_trial import _hash, _read
from .tt20_camera_carry import OUTPUT as SOURCE


TEACHERS = (
    'curriculum-frozen-2082473549', 'curriculum-frozen-2082473549-nominal',
    'camera-carry-left-game-40582930-0', 'camera-carry-wrist-game-40582930-0',
    'camera-carry-left-game-68239503-1-strong', 'camera-carry-wrist-game-68239503-2-strong',
    'mono-flight-outside', 'mono-flight-wrist',
)
SHUFFLE = 122718311
LIMIT = 18


def _receives(data):
    if 'points' in data:
        yield from (p for p in data['points'] if p.get('role', p.get('server')) == -1)
    else:
        for pairs in data['pairs'].values():
            for pair in pairs:
                for ranging in ('size', 'temporal'):
                    yield pair[ranging]


def _opening_prefix(actions, tapes):
    active = np.flatnonzero(np.any(actions[:, :5] != 0., axis=1))
    if not len(active):
        return []
    start = int(active[0])
    duration = min(5, len(actions) - start)
    candidates = np.flatnonzero(np.all(tapes[:, :duration] == actions[start:start+duration, :5], axis=(1, 2)))
    if len(candidates) != 1:
        raise ValueError('archived stroke cannot be uniquely matched to its executed command prefix')
    # Receipts without explicit plans can identify only the first shot.
    return [dict(index=int(candidates[0]), tick=start, duration=len(tapes[0]), verified=duration)]


def _credited_plan(row, actions, tapes, plan, next_start):
    index, start, duration = (int(plan[key]) for key in ('index', 'tick', 'duration'))
    length = int(plan.get('verified', min(duration, len(actions) - start)))
    if length < 1 or not np.array_equal(actions[start:start+length, :5], tapes[index, :length]):
        raise ValueError(f'archived command mismatch at seed {row["seed"]}, tick {start}')
    contacts = [event['step'] for event in row['events']
                if event['kind'] == 'contact' and event.get('side') == 1]
    first = next((t for t in contacts if start < t <= min(next_start, start + duration + 1)), None)
    stop = next((t for t in contacts if first is not None and t > first), np.inf)
    legal = first is not None and any(event['kind'] == 'legal_return' and event.get('side') == 1
                                      and first < event['step'] < stop for event in row['events'])
    return dict(index=index, tick=start, command_frames=length, contact=first, legal=bool(legal))


def demonstrations(tapes):
    observed, manifests = {}, {}
    for name in TEACHERS:
        path = SOURCE / f'{name}.json.gz'
        data = _read(path)
        manifests[name] = _hash(path)
        for row in _receives(data):
            actions = np.asarray(row['actions'])
            plans = row.get('plans')
            if plans is None:
                plans = _opening_prefix(actions, tapes)
            for ordinal, plan in enumerate(plans):
                stop = plans[ordinal+1]['tick'] if ordinal+1 < len(plans) else len(actions)
                example = _credited_plan(row, actions, tapes, plan, stop)
                key = (row['seed'], example['index'], example['tick'], example['contact'], example['legal'])
                observed.setdefault(key, dict(source=name, seed=row['seed'], stage=ordinal, **example))
    return list(observed.values()), manifests


def _repertoire(records, credit, size):
    scores = defaultdict(list)
    for row, reward in zip(records, credit, strict=True):
        scores[row['index']].append(reward)
    # Neutral prior keeps one-success tapes behind repeatedly demonstrated shots.
    rank = sorted(scores, key=lambda i: (-(1. + sum(scores[i])) / (2 + len(scores[i])), -len(scores[i]), i))
    return rank[:size]


def distill(tapes, size=LIMIT):
    records, manifests = demonstrations(tapes)
    credit = [int(row['legal']) for row in records]
    count = min(size, len({row['index'] for row in records if row['legal']}))
    if count < 4 or len(set(credit)) != 2:
        raise ValueError('insufficient audited demonstrations for a matched control')
    taught = _repertoire(records, credit, count)
    shuffled_credit = np.random.default_rng(SHUFFLE).permutation(credit).tolist()
    shuffled = _repertoire(records, shuffled_credit, count)
    if taught == shuffled:
        raise ValueError('the credit manipulation must change the repertoire')
    return dict(taught=taught, shuffled=shuffled, records=records, manifests=manifests,
                shuffled_credit=shuffled_credit, shuffle_seed=SHUFFLE)
