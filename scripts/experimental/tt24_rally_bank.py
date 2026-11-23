"""Harvest broad archived optical command examples and assess a new all-live receiving pupil."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, REPLICATION, SequenceArena, course, verify
from .tt24_rally_from_release import OPENING
from .tt24_rally_policy import WINDOW, JointPolicy, ViewHistory
from .tt24_rally_trial import (COACHED, DATA, OUTPUT, STEP_SIZE, PublicBody, Station, alternating,
                               archive_schedule, contacts, difference_count, returns, summary)


ARCHIVE = COACHED.parent
SOURCE_FILES = re.compile(r'(implicit-(confirm|wide)|online-rally-(confirm|stream)|jog-bank)-[0-9]+\.json\.gz')
HOLDOUT = {'e': 2098474181, 'f': 1260741303}
ORDER_SEED = 310574883
DIMS = (240, 180)
CACHE = OUTPUT / 'rally-bank-demonstrations.npz'
RECEIPT = OUTPUT / 'rally-bank-held.json.gz'
DATA_RECEIPT = OUTPUT / 'rally-bank-demonstrations.json.gz'
CONDITIONS = ('original', 'broad', 'sight_delay', 'hold')
BASE_SETTINGS = dict(neighbors=20, body=.68, mix=0.)
BROAD_SETTINGS = dict(neighbors=28, body=.68, mix=0.)
ARENA = LIVE = None


def fresh_releases():
    released = course(HOLDOUT)
    seeds = sorted(row['seed'] for row in released)
    if len(set(seeds)) != len(seeds):
        raise ValueError('the two reserved streams repeat a launch')
    number = re.compile(rb'(?<![0-9.])(?:' + b'|'.join(str(seed).encode() for seed in seeds) + rb')(?![0-9.])')
    for folder in (ARCHIVE.with_name('tt22'), ARCHIVE):
        for path in sorted(folder.iterdir()):
            if path.name.endswith('.json.gz'):
                raw = gzip.decompress(path.read_bytes())
            elif path.suffix == '.json':
                raw = path.read_bytes()
            else:
                continue
            if number.search(raw) is not None:
                raise ValueError(f'a reserved launch occurs in the prior artifact {path}')
    return released


def source_rows():
    for path in sorted(ARCHIVE.iterdir()):
        if SOURCE_FILES.fullmatch(path.name) is None:
            continue
        document = _read(path)
        if document['verified'] != sum(len(case['results']) for case in document['cases']):
            raise ValueError(f'archive has unverified physical tapes at {path}')
        archive_hash = _hash(path)
        for case in document['cases']:
            for condition, row in case['results'].items():
                if (case['seed'], case['lane']) != (row['seed'], row['lane']):
                    raise ValueError('archive row seed or lane changed')
                if len(returns(row, 1)) >= 2 and row['outcome']['rival_returns'] > 0:
                    tag = dict(file=path.name, archive_sha256=archive_hash, modality=condition)
                    yield tag, row


def source_schedule(count, held):
    excluded = {row['seed'] for row in held + course(REPLICATION) + archive_schedule()}
    with np.load(DATA) as prior:
        excluded.update(map(int, prior['seed']))
    strongest = {}
    for source, row in source_rows():
        old = strongest.get(row['seed'])
        if row['seed'] not in excluded and (old is None or row['longest'] > old[1]['longest']):
            strongest[row['seed']] = source, row
    strata = {}
    for source, row in strongest.values():
        key = row['lane'] > 0., abs(row['lane']) > .48
        strata.setdefault(key, []).append((source, row))
    random = np.random.default_rng(ORDER_SEED)
    blocks = []
    for key in sorted(strata):
        options = sorted(strata[key], key=lambda item: (-item[1]['longest'], item[1]['seed']))
        head, tail = options[:2], options[2:]
        blocks.append(head + [tail[index] for index in random.permutation(len(tail))])
    schedule = [block[i] for i in range(max(map(len, blocks))) for block in blocks if i < len(block)]
    if len(schedule) < count:
        raise ValueError('requested demonstration budget exceeds disjoint eligible tapes')
    return schedule[:count], len(schedule)


def sample_ranges(row):
    legal = returns(row, 1)
    hits = contacts(row, 1)
    chosen = [max(tick for tick in hits if tick <= landing) for landing in legal[1:]]
    ranks = sorted({0, len(chosen)//2, len(chosen)-1})
    ends = [(0, legal[0] + 6)]
    ends.extend((max(0, chosen[i]-18), chosen[i]+7) for i in ranks)
    frames = np.arange(len(row['actions']))
    keep = np.zeros(len(frames), dtype=bool)
    render = np.zeros(len(frames), dtype=bool)
    for start, stop in ends:
        keep |= (frames >= start) & (frames < stop)
        render |= (frames >= max(0, start-WINDOW)) & (frames < stop)
    return keep, render


def prepare_collection():
    global ARENA
    ARENA = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))


def extract(item):
    source, row = item
    arena = ARENA
    arena.lane = row['lane']
    arena.reset(seed=row['seed'], server=-1, render=False)
    body = PublicBody(arena)
    features, positions, commands = [], [], []
    keep, render = sample_ranges(row)
    previous = np.zeros(arena.action_size)
    history = None
    seen = sha256()
    for tick, (action, opponent) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
        action, opponent = np.asarray(action), np.asarray(opponent)
        if not render[tick]:
            history = None
        else:
            if history is None:
                history = ViewHistory()
                history.issued(previous)
            frames = arena.render_cameras()
            for key, frame in frames.items():
                seen.update(key.encode())
                seen.update(frame)
            state = history.observe(frames, arena.joints, body.target)
            history.issued(action)
        arena._aim = arena._command_to = lambda *args, opponent=opponent: opponent
        _, reward, done, cap, _ = arena.step(action, render=False)
        body.issued(action)
        previous = action
        if keep[tick]:
            features.append(state)
            positions.append(body.target.copy())
            commands.append(action)
        if bool(done or cap) != (tick + 1 == len(row['actions'])):
            raise ValueError(f'replayed demonstration terminated differently on {row["seed"]}')
    if (arena.point_result() != row['outcome'] or reward != row['reward']
            or digest_json(arena.events) != digest_json(row['events']) or alternating(arena.events) != row['longest']):
        raise ValueError(f'replayed demonstration changed the referee record on {row["seed"]}')
    receipt = dict(source, seed=row['seed'], lane=row['lane'], legal_returns=row['outcome']['legal_returns'],
                   alternating=row['longest'], physical_frames=len(row['actions']), rendered_frames=int(render.sum()),
                   fit_frames=int(keep.sum()), sampled_rgb_sha256=seen.hexdigest(), strict_replay=True)
    result = dict(features=np.asarray(features), postures=np.asarray(positions), rates=np.asarray(commands),
                  seed=np.full(len(features), row['seed']))
    return receipt, result


def collect(path, report_path, *, workers, count, held):
    items, available = source_schedule(count, held)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=prepare_collection) as pool:
        blocks, entries = [], []
        for row, block in pool.map(extract, items):
            entries.append(row)
            blocks.append(block)
            print('harvest', len(entries), row['seed'], row['alternating'], row['fit_frames'], flush=True)
    array = {key: np.concatenate([block[key] for block in blocks]) for key in ('features', 'postures', 'rates', 'seed')}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **array)
    temporary.replace(path)
    report = dict(schema='tt24-broad-rgb-joint-demo-1', candidate_releases=available, demos=count,
                  fit_samples=len(array['seed']), archive_seconds=sum(row['physical_frames'] for row in entries)/30.,
                  rendered_frames=sum(row['rendered_frames'] for row in entries), dimensions=list(DIMS),
                  samples_sha256=_hash(path), selection_generator=ORDER_SEED, releases=entries,
                  withheld_launches=held)
    _write(report_path, report)
    return report


def prepare_held(bank, original_opening, original_data, profile):
    global LIVE
    keys = ('features', 'postures', 'rates')
    with np.load(original_opening) as first, np.load(original_data) as next_, np.load(bank) as broad:
        original = [np.concatenate((first[key], next_[key])) for key in keys]
        samples = [broad[key] for key in keys]
    models = dict(original=JointPolicy(*original, STEP_SIZE, **BASE_SETTINGS),
                  broad=JointPolicy(*samples, STEP_SIZE, **BROAD_SETTINGS))
    models['sight_delay'] = models['broad']
    LIVE = Station(profile, render_dims=DIMS), models


def evaluate_start(index, row):
    station, models = LIVE
    order = CONDITIONS[index % len(CONDITIONS):] + CONDITIONS[:index % len(CONDITIONS)]
    outcomes = {name: station.play(row, name, models.get(name), live_opening=True) for name in order}
    common = {outcome['prefix_rgb'] for outcome in outcomes.values()}
    if len(common) != 1:
        raise ValueError('initial RGB differs across paired receiving learners')
    for outcome in outcomes.values():
        outcome['vs_hold'] = difference_count(outcome, outcomes['hold'], 0)
    outcomes['broad']['vs_original'] = difference_count(outcomes['broad'], outcomes['original'], 0)
    return index, dict(row, results=outcomes)


def evaluate(path, bank, data_report, held, *, workers):
    source = _read(data_report)
    seeds = {row['seed'] for row in held}
    if source['samples_sha256'] != _hash(bank) or held != source['withheld_launches']:
        raise ValueError('the broad bank or held roster changed')
    with np.load(bank) as broad, np.load(OPENING) as first, np.load(DATA) as old:
        if seeds.intersection(map(int, np.r_[broad['seed'], first['seed'], old['seed']])):
            raise ValueError('a held release occurs in the teacher data')
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-broad-joint-from-release-1', dimensions=list(DIMS), cases=held,
                opening_teacher_commands_in_assessment=0, seed_streams=HOLDOUT, settings_original=BASE_SETTINGS,
                settings_broad=BROAD_SETTINGS, conditions=list(CONDITIONS), source_manifest_sha256=_hash(data_report),
                archive_sha256=source['samples_sha256'], original_opening_sha256=_hash(OPENING),
                original_continuation_sha256=_hash(DATA))
    report = _read(path) if path.exists() else dict(metadata=meta, cases=[None]*len(held))
    if report['metadata'] != meta:
        raise ValueError('wider from-release assessment has different training data or roster')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare_held,
                             initargs=(bank, OPENING, DATA, profile)) as pool:
        pending = [pool.submit(evaluate_start, i, row) for i, row in enumerate(held) if report['cases'][i] is None]
        for task in as_completed(pending):
            index, case = task.result()
            report['cases'][index] = case
            _write(path, report)
            print('bank-held', case['seed'], {key: returns(row, 1) for key, row in case['results'].items()}, flush=True)
    report['verified_tapes'] = verify(report['cases'])
    report['summary'] = {name: summary(report['cases'], name, 0) for name in CONDITIONS}
    for counts in report['summary'].values():
        counts['reach_first'] = sum(count > 0 for count in counts['own_returns'])
    _write(path, report)
    return report['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=CACHE)
    parser.add_argument('--data-receipt', type=Path, default=DATA_RECEIPT)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--demos', type=int, default=48)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    starts = fresh_releases()
    if not args.bank.exists() or not args.data_receipt.exists():
        collect(args.bank, args.data_receipt, workers=args.workers, count=args.demos, held=starts)
    if not args.collect_only:
        print(evaluate(args.output, args.bank, args.data_receipt, starts, workers=args.workers), flush=True)
