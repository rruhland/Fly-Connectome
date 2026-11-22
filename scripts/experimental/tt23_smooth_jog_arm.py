"""Collect one shared fresh bank of real arm jogs, then score frozen policies."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO, STREAMS
from .tt22_joint_practice import READS, draw
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive
from .tt22_motor_transfer_trial import MotorArena
from .tt23_anonymous_target import target_context
from .tt23_smooth_jog import OPTIONS, OUT, RECEIPT, choose


SEED = 846093157
COUNT = 64
BANK = OUT/'smooth-jog-arm.json.gz'
RESULT = OUT/'smooth-jog-assessment.json.gz'


def schedule(seed):
    rng = np.random.default_rng(seed)
    names = tuple(STREAMS)
    batches = [draw(rng, COUNT//len(names), STREAMS[name]) for name in names]
    points = [(name, int(start), float(lane)) for pair in zip(*batches, strict=True)
              for name, (start, lane) in zip(names, pair, strict=True)]
    used = set()
    for archive in ('anonymous-action-arm.json.gz', 'anonymous-replication-arm.json.gz'):
        used.update(row['seed'] for row in _read(OUT/archive)['rows'])
    starts = {point[1] for point in points}
    if len(starts) != COUNT or starts & used:
        raise ValueError('bank schedule overlaps earlier arm evaluation or contains duplicates')
    return points


def outcome(arena, seed, lane, macro, ready, jog, identified):
    arena.lane = lane
    arena.reset(seed=seed, server=-1, render=False)
    move = MotorPrimitive(macro, None)
    if identified:
        move._translate(ready[0]+jog*.848, ready)
    for tick in range(arena.max_steps):
        action = np.zeros(6) if tick < READS else move.command(tick)
        _, _, done, capped, info = arena.step(action, render=False)
        if done or capped:
            break
    return dict(ticks=arena.steps, contacts=info['contacts'], legal=info['legal_returns'], fault=info['reason'])


def collect(path, seed):
    points = schedule(seed)
    macro = np.asarray(_read(MACRO)['parameters'])
    ready = MotorPrimitive(macro, None).ready_world
    metadata = dict(schema='tt23-smooth-jog-physical-bank-1', seed_generator=seed, count=COUNT,
                    macro_hash=_hash(MACRO), options=OPTIONS.tolist())
    data = _read(path) if path.exists() else dict(metadata, rows=[], complete=False)
    if any(data.get(key) != value for key, value in metadata.items()):
        raise ValueError('cannot resume a different physical bank')
    rows = data['rows']
    if [(row['domain'], row['seed'], row['lane']) for row in rows] != points[:len(rows)]:
        raise ValueError('the saved bank does not match the registered schedule')
    if not data['complete']:
        arena = MotorArena()
        try:
            for domain, start, lane in points[len(rows):]:
                sight = prefix(arena, start, lane)
                identified = target_context(sight, ready) is not None
                options = [outcome(arena, start, lane, macro, ready, jog, identified) for jog in OPTIONS]
                rows.append(dict(domain=domain, seed=start, lane=lane, sight=sight, options=options))
                if len(rows) % 8 == 0:
                    _write(path, data)
                    print('physical starts', len(rows), flush=True)
        finally:
            arena.close()
        data['complete'] = True
        _write(path, data)
    if len(rows) != COUNT or any(len(row['options']) != len(OPTIONS) for row in rows):
        raise ValueError('the physical bank is incomplete')
    return data


def assess(bank_path, source_path, output):
    data, source = _read(bank_path), _read(source_path)
    if len(data['rows']) != COUNT or not data['complete'] or not source['gate']:
        raise ValueError('assessment requires the complete physical bank and the frozen passing source model')
    ready = MotorPrimitive(np.asarray(_read(MACRO)['parameters']), None).ready_world
    choices = []
    names = ('learned', 'shuffled', 'neutral')
    for row in data['rows']:
        context = target_context(row['sight'], ready)
        learned = int(choose(source['model'], context)[0]) if context is not None else 3
        shuffled = int(choose(source['model'], context, shuffle=True)[0]) if context is not None else 3
        picks = dict(zip(names, (learned, shuffled, 3), strict=True))
        choices.append(dict(seed=row['seed'], domain=row['domain'], picks=picks,
                            outcomes={name: row['options'][picks[name]] for name in names}))
    summary = {name: {metric: sum(row['outcomes'][name][metric] > 0 for row in choices)
                      for metric in ('contacts', 'legal')} for name in names}
    result = dict(schema='tt23-smooth-jog-assessment-1', bank_sha256=_hash(bank_path),
                  source_sha256=_hash(source_path), rows=choices, summary=summary)
    _write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--source', type=Path, default=RECEIPT)
    parser.add_argument('--output', type=Path, default=RESULT)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    collect(args.bank, args.seed)
    if not args.collect_only:
        print(assess(args.bank, args.source, args.output)['summary'], flush=True)


if __name__ == '__main__':
    main()
