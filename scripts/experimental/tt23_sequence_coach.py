"""Test fixed source openers against a far partner fitted to sustain legal rallies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import EXACT_CHECKS
from .tt23_coach_trial import PRACTICE, RANK_WEIGHT, ReplyRanking
from .tt23_sequence import DESTINATION as STOCK
from .tt23_sequence import REPLICATION, SequenceStation, milestones, setup, summarize, verify


OUTPUT = STOCK.with_name('tt23_sequence_coach.json.gz')
CONFIRMATION = STOCK.with_name('tt23_sequence_coach_confirmation.json.gz')
MODES = ('duration', 'neutral')


def first_ranking(row):
    return next((plan['tick'] for plan in row['rival'] if plan['predicted_legal']), None)


def audit_inactive(station, stock):
    """Replay the near tape while the new partner actually computes every far command."""
    arena, rival = station.arena, station.rival
    arena.lane = stock['lane']
    arena.reset(seed=stock['seed'], server=-1, render=False)
    station.far.steps.clear()
    live_confirm = rival.confirm

    def reject_ranking(*args):
        raise ValueError('the learned ranker was called on an allegedly inactive tape')

    rival.confirm = reject_ranking
    digest = sha256()
    try:
        for tick, action in enumerate(stock['actions'], 1):
            for eye, frame in arena.render_cameras().items():
                digest.update(eye.encode())
                digest.update(frame)
            _, reward, done, cap, _ = arena.step(action, render=False)
            if bool(done or cap) != (tick == len(stock['actions'])):
                raise ValueError('the inactive point duration diverged')
    finally:
        rival.confirm = live_confirm
    far = station.far.tape(len(stock['actions']))
    if (far != stock['rival_actions'] or reward != stock['reward'] or arena.point_result() != stock['outcome']
            or digest_json(arena.events) != digest_json(stock['events']) or digest.hexdigest() != stock['rgb']):
        raise ValueError('the inactive source, physical or RGB tape audit diverged')
    return dict(stock, rival=rival.decisions.copy(), execution='inactive_audit')


def check_prefix(stock, learned):
    stock_tick = first_ranking(stock)
    if stock_tick != first_ranking(learned) or stock['opening'] != learned['opening']:
        raise ValueError('the coach changed the common opening before ranking')
    divergence = next((tick for tick, (a, b) in enumerate(zip(stock['rival_actions'], learned['rival_actions']))
                       if a != b), None)
    if divergence is not None and (stock_tick is None or divergence < stock_tick):
        raise ValueError('far commands changed before the learned ranking was eligible')
    stop = len(stock['actions']) if divergence is None else divergence
    stock_events = [event for event in stock['events'] if event['step'] <= stop]
    learned_events = [event for event in learned['events'] if event['step'] <= stop]
    if (stock['actions'][:stop + 1] != learned['actions'][:stop + 1]
            or stock['rival_actions'][:stop] != learned['rival_actions'][:stop]
            or digest_json(stock_events) != digest_json(learned_events)):
        raise ValueError('physical play diverged before the first changed far command')
    if divergence is None and (stock['rgb'] != learned['rgb'] or stock['outcome'] != learned['outcome']):
        raise ValueError('identical far commands produced changed RGB or a changed outcome')
    return divergence


def configuration(confirmation):
    stock, practice = _read(STOCK), _read(PRACTICE)
    if stock['verified'] != 72 or not practice['live_gate'] or practice['profile'] != stock['far_profile']:
        raise ValueError('expected the verified opening baseline and frozen matching coach fit')
    original = setup()
    opening = setup(REPLICATION) if confirmation else original
    if any(digest_json(stock[key]) != digest_json(value) for key, value in original.items()):
        raise ValueError('archived A/B schedule or near controller inputs changed')
    fitted = {row['seed'] for row in practice['rows']}
    if any(visit['seed'] in fitted for visit in opening['starts']):
        raise ValueError('the coach fit used an assessment point')
    return stock, dict(schema='tt23-coach-receiving-sequence-1', confirmation=confirmation, mode=list(MODES),
                       stock_sha256=_hash(STOCK), practice_sha256=_hash(PRACTICE),
                       coach_weight=RANK_WEIGHT, exact_checks=EXACT_CHECKS, opening=opening,
                       permuted_live_checks=0 if confirmation else 2)


WORKER = None


def prepare(opening, practice):
    global WORKER
    WORKER = SequenceStation(opening['macro'], opening['evidence'], opening['far_profile'])
    WORKER.rival.confirm = ReplyRanking(WORKER.rival, practice).confirm


def branch(start, mode, gains, stock):
    if stock is not None and mode != 'permuted' and first_ranking(stock) is None:
        return dict(audit_inactive(WORKER, stock), first_far_difference=None)
    row = WORKER.play(start['seed'], start['lane'], np.asarray(gains))
    changed = None if stock is None else check_prefix(stock, row)
    return dict(row, **milestones(row), execution='live', first_far_difference=changed)


def summaries(stock, cases, confirmation):
    result = {'coach': {mode: summarize([case['results'][mode] for case in cases]) for mode in MODES}}
    if confirmation:
        return result
    old = {mode: [stock['cases'][i]['results'][mode] for i in range(len(cases))] for mode in MODES}
    result['stock'] = {mode: summarize(rows) for mode, rows in old.items()}
    result['paired'] = paired = {}
    for mode in MODES:
        new = [case['results'][mode] for case in cases]
        paired[mode] = {}
        for threshold in (10, 20):
            transitions = [(a['longest'] >= threshold, b['longest'] >= threshold)
                           for a, b in zip(old[mode], new, strict=True)]
            paired[mode][str(threshold)] = dict(stock_only=transitions.count((True, False)),
                                                coach_only=transitions.count((False, True)),
                                                both=transitions.count((True, True)))
    return result


def evaluate(path, workers, confirmation):
    stock, metadata = configuration(confirmation)
    opening = metadata['opening']
    data = _read(path) if path.exists() else dict(metadata=metadata, cases=[dict(start, results={})
                                                                          for start in opening['starts']])
    if digest_json(data['metadata']) != digest_json(metadata):
        raise ValueError('checkpoint schema, course or policy changed')
    work = []
    for i, (case, start) in enumerate(zip(data['cases'], opening['starts'], strict=True)):
        if {k: case[k] for k in start} != start:
            raise ValueError('checkpoint start or launch lane changed')
        modes = MODES[::1 if i % 2 == 0 else -1]
        if i < metadata['permuted_live_checks']:
            modes += ('permuted',)
        for mode in modes:
            if mode not in case['results']:
                old = None if confirmation else stock['cases'][i]['results'][mode]
                work.append((i, mode, start, opening['gains'][mode], old))
    _write(path, data)
    # Complete the inexpensive, strict equivalence checks first when resuming or inspecting progress.
    work.sort(key=lambda item: item[-1] is None or first_ranking(item[-1]) is not None)
    if work:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                 initializer=prepare, initargs=(opening, _read(PRACTICE))) as pool:
            futures = {pool.submit(branch, start, mode, gains, old): (i, mode)
                       for i, mode, start, gains, old in work}
            for task in as_completed(futures):
                i, mode = futures[task]
                try:
                    row = task.result()
                except Exception as exc:
                    print(i + 1, mode, str(exc), flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                data['cases'][i]['results'][mode] = row
                _write(path, data)
                print(i + 1, mode, row['execution'], row['longest'], row['first_far_difference'], flush=True)
    data['verified'] = verify(data['cases'])
    data['summary'] = summaries(stock, data['cases'], confirmation)
    data['work'] = {kind: sum(row['execution'] == kind for case in data['cases'] for row in case['results'].values())
                    for kind in ('live', 'inactive_audit')}
    _write(path, data)
    print(data['summary'], data['work'], 'verified', data['verified'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    path = args.output or (CONFIRMATION if args.confirmation else OUTPUT)
    if args.verify:
        print('verified', verify(_read(path)['cases']), flush=True)
    elif args.plan:
        if path.exists():
            parser.error('the proposed plan path already has a receipt')
        _, metadata = configuration(args.confirmation)
        _write(path, dict(metadata=metadata, cases=[dict(start, results={}) for start in metadata['opening']['starts']]))
    else:
        evaluate(path, args.workers, args.confirmation)


if __name__ == '__main__':
    main()
