"""Change an all-live joint policy through delayed, scalar returns from practice points."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import REPLICATION, course, verify
from .tt24_rally_bank import CACHE, DIMS, HOLDOUT
from .tt24_rally_improve import stereo_motion
from .tt24_rally_improve_phase import PhaseHistory
from .tt24_rally_improve_trial import CONFIRMATION, policy_for
from .tt24_rally_trial import COACHED, OUTPUT, Station, difference_count, returns, summary


RECEIPT = OUTPUT / 'rally-improve-practice.json.gz'
FRESH = {'i': 1048644851, 'j': 1825127553}
CONTINUATION_FRESH = {'o': 1207435807, 'p': 1047196904}
CONTINUATION_RECEIPT = OUTPUT / 'rally-improve-practice-continuation.json.gz'
PRACTICE_SEEDS = (1173305880, 1570727075, 1176636968)
CONTINUATION_SEEDS = (1173305880, 146477391, 1176636968)
SEARCH_SEED = 402561144
MODES = ('broad', 'practiced', 'sight_delay', 'hold')
WORKER = None


class PracticedPolicy:
    """Apply a rewarded sensor-conditioned perturbation to each active joint rate."""

    def __init__(self, base, weights, gated=False):
        self.base = base
        self.weights = weights
        self.gated = gated

    def command(self, state, issued_position):
        gate = state[-1] if self.gated else 1.
        if self.gated:
            state = state[:-1]
        prior = self.base.command(state, issued_position)
        optical = stereo_motion(state)
        basis = np.r_[1., np.tanh(optical[[0, 1, 2, 4]]/[.16, .64, .20, .065])]
        delta = gate*np.clip(self.weights @ basis, -.32, .32)
        prior[:5] = np.clip(prior[:5] + delta, -1., 1.)
        return prior


def prepare(profile, target, gated=False):
    global WORKER
    station = Station(profile, render_dims=DIMS)
    with np.load(CACHE) as data:
        broad = policy_for('broad', data)
    WORKER = station, broad, None if target is None else PracticedPolicy(broad, np.asarray(target), gated), gated


def play_practice(start, weights):
    station, base, _, gated = WORKER
    policy = PracticedPolicy(base, np.asarray(weights), gated)
    if gated:
        return station.play(start, 'practice', policy, PhaseHistory, live_opening=True)
    return station.play(start, 'practice', policy, live_opening=True)


def play_held(start, order):
    station, base, policy, gated = WORKER
    rows = {}
    for name in order:
        if gated and name in ('practiced', 'phase_delay'):
            history = lambda: PhaseHistory(delayed=name == 'phase_delay')
            rows[name] = station.play(start, name, policy, history, live_opening=True)
        else:
            controller = base if name == 'broad' else policy
            rows[name] = station.play(start, name, controller, live_opening=True)
    if len({row['prefix_rgb'] for row in rows.values()}) != 1:
        raise ValueError('held conditions have different initial camera input')
    for row in rows.values():
        row['vs_broad'] = difference_count(row, rows['broad'], 0)
    return dict(start, results=rows)


def utility(rows):
    earned = [len(returns(row, 1)) for row in rows]
    # Unit payoff for each legal return, clipped per point to balance practice starts.
    return int(sum(min(8, n) for n in earned))


def candidates(trial, anchor, direction, rng, gated=False):
    if trial == 0:
        return anchor, direction
    if trial % 2:
        direction = rng.normal(size=(5, 1)) * rng.normal(size=(1, 5))
        scale = rng.choice((.018, .038, .062) if gated else (.045, .085, .14))
        direction = scale*direction
    sign = 1 if trial % 2 else -1
    return np.clip(anchor + sign*direction, -.24, .24), direction


def learn(training, profile, workers, budget, report, path, gated=False):
    rng = np.random.default_rng(SEARCH_SEED)
    anchor = np.zeros((5, 5))
    direction, best = np.zeros_like(anchor), (-1, anchor.copy())
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(profile, None, gated)) as pool:
        for trial in range(budget):
            weights, direction = candidates(trial, anchor, direction, rng, gated)
            if trial < len(report['practice']):
                old = report['practice'][trial]
                if not np.array_equal(weights, old['weights']):
                    raise ValueError('checkpoint practice selection changed')
                rows = old['rows']
            else:
                rows = list(pool.map(play_practice, training, [weights]*len(training)))
                report['practice'].append(dict(weights=weights.tolist(), rows=rows, scalar=utility(rows)))
                _write(path, report)
                print('practice', trial + 1, utility(rows), [len(returns(row, 1)) for row in rows], flush=True)
            score = utility(rows)
            if score > best[0]:
                best = score, weights.copy()
            if trial % 2 == 0:
                anchor = best[1].copy()
    return best


def experiment(path, workers, budget, *, gated=False):
    lookup = {row['seed']: row for row in course(HOLDOUT)}
    training = [lookup[seed] for seed in (CONTINUATION_SEEDS if gated else PRACTICE_SEEDS)]
    testing = course(CONTINUATION_FRESH if gated else FRESH)
    modes = ('broad', 'practiced', 'phase_delay', 'hold') if gated else MODES
    previous = set(lookup) | {row['seed'] for row in course(REPLICATION) + course(CONFIRMATION)}
    with np.load(CACHE) as bank:
        if (previous | set(map(int, bank['seed']))) & {row['seed'] for row in testing}:
            raise ValueError('frozen practice test contains a prior launch')
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-scalar-practice-from-release-1', conditions=list(modes), practice=training,
                assessment=testing, search_seed=SEARCH_SEED, practice_candidates=budget,
                dimensions=list(DIMS), demonstration_sha256=_hash(CACHE), profile=profile,
                teacher_commands_in_practice_or_assessment=0)
    if gated:
        meta['reward_updates_only_after_observed_reversal'] = True
    report = _read(path) if path.exists() else dict(metadata=meta, practice=[], cases=[None]*len(testing))
    if report['metadata'] != meta:
        raise ValueError('checkpoint has different demonstration, practice or held starts')
    path.parent.mkdir(parents=True, exist_ok=True)
    score, weights = learn(training, profile, workers, budget, report, path, gated)
    report['frozen'] = dict(weights=weights.tolist(), practice_score=score)
    training_cases = [dict(results={'practice': row}) for visit in report['practice'] for row in visit['rows']]
    report['practice_frames'] = sum(len(row['actions']) for visit in report['practice'] for row in visit['rows'])
    if not np.any(weights):
        held_cases = [case for case in report['cases'] if case is not None]
        for case in held_cases:
            base, adapted = (case['results'][name] for name in ('broad', 'practiced'))
            if any(base[key] != adapted[key] for key in ('actions', 'rival_actions', 'rgb', 'events')):
                raise ValueError('zero residual changed the audited physical trajectory')
        report['verified_tapes'] = verify(training_cases + held_cases)
        report['summary'] = dict(changed_policy=False, held_required=False, practice_score=score,
                                 held_replay_audits=len(held_cases))
        _write(path, report)
        return report['summary']
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(profile, weights, gated)) as pool:
        tasks = {}
        for i, start in enumerate(testing):
            if report['cases'][i] is None:
                offset = i % len(modes)
                order = modes[offset:] + modes[:offset]
                tasks[pool.submit(play_held, start, order)] = i
        for task in as_completed(tasks):
            case = task.result()
            report['cases'][tasks[task]] = case
            _write(path, report)
            print('held', case['seed'], {name: returns(row, 1) for name, row in case['results'].items()}, flush=True)
    report['verified_tapes'] = verify(training_cases + report['cases'])
    report['summary'] = {}
    for name in modes:
        counts = summary(report['cases'], name, 0)
        counts['reach_first'] = sum(n > 0 for n in counts['own_returns'])
        report['summary'][name] = counts
    _write(path, report)
    return report['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--budget', type=int, default=15)
    parser.add_argument('--continuation', action='store_true')
    args = parser.parse_args()
    if args.budget < 1 or args.budget % 2 != 1:
        parser.error('practice budget must include the start and a whole number of pairs')
    path = args.output or (CONTINUATION_RECEIPT if args.continuation else RECEIPT)
    print(experiment(path, args.workers, args.budget, gated=args.continuation), flush=True)
