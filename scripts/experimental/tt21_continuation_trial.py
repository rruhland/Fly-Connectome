"""Practice only the continuation of a real rally, then score uninterrupted play."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_model_rollout import DrySimulator
from .tt20_curriculum import (OUTPUT as PREVIOUS, apparatus, configuration, model_arena,
                              receiver, transport)
from .tt21_continuation import ContinuationPlayer, PHASES, learn_fallback, learn_phase


OUTPUT = ROOT / 'artifacts/tt21'
ARCHIVE = PREVIOUS / 'curriculum-frozen-2082473549.json.gz'
TRAIN = 'continuation-practice.json.gz'
HELD = 'continuation-held.json.gz'
ASSESSMENT_SEED = 2134971899
PRACTICE_SEED = 1103826372
OPPONENTS = ('standard', 'search')


def own(event, kind):
    return event['kind'] == kind and event.get('side') == 1


def first_reply(events):
    """Use the first opponent strike only if the referee credits that reply."""
    legal = next((i for i, event in enumerate(events)
                  if event['kind'] == 'legal_return' and event.get('side') == -1), None)
    if legal is None:
        return None
    contact = next(i for i in range(legal - 1, -1, -1)
                   if events[i]['kind'] == 'contact' and events[i].get('side') == -1)
    return events[contact]['step']


def training_prefixes():
    archive = _read(ARCHIVE)
    prefixes = []
    for row in archive['points']:
        tick = first_reply(row['events'])
        if tick is None:
            continue
        actions = row['actions'][:tick]
        if np.max(abs(np.sum(actions, axis=0))) > .003:
            raise ValueError('archived opening did not return the commanded arm to neutral')
        prefixes.append(dict(seed=row['seed'], server=row['server'], actions=actions,
                             events=[e for e in row['events'] if e['step'] <= tick]))
    if len(prefixes) != 6:
        raise ValueError('expected all six physical reply opportunities in the archive')
    return prefixes


def player():
    _, catalogue, law, response, serve = configuration(PREVIOUS)
    match = receiver(False, PREVIOUS)
    match.player.close()
    motor = ContinuationPlayer(catalogue, law, response)
    motor.sim.close()
    motor.sim = DrySimulator(law, arena=model_arena())
    motor.phase = 0
    match.player = motor
    match.serves.force = serve
    return match


def measure_credit(events, result, boundary):
    after = [event for event in events if boundary is not None and event['step'] > boundary]
    return dict(legal=any(own(event, 'legal_return') for event in after),
                contact=any(own(event, 'contact') for event in after),
                win=bool(after and result['points_for']))


def sight(camera, arena):
    camera.update(arena.render_cameras())
    ball = camera.ball
    return None if ball is None or not ball.observed else list(map(float, ball.position))


def play(arena, match, seed, role, phase, *, prefix=None):
    # No live simulator values reach the motor policy. The practice reset's
    # actions reconstruct a legal opening, just like pressing replay at a drill.
    arena.reset(seed=seed, server=role, render=False)
    camera = transport(arena, False).tracker()
    match.reset(role)
    motor = match.player
    motor.phase = phase
    commands, optical = [], []
    for tick in range(arena.max_steps):
        restoring = prefix is not None and tick < len(prefix['actions'])
        if restoring:
            position = sight(camera, arena) if tick >= len(prefix['actions']) - 8 else None
            action = prefix['actions'][tick]
        else:
            if prefix is not None and tick == len(prefix['actions']):
                if json.dumps(arena.events) != json.dumps(prefix['events']):
                    raise ValueError('the physical practice prefix changed')
                match.phase = 'receive'
            position = sight(camera, arena) if tick % 2 == 0 or match.phase == 'receive' else None
            motor.followup = (prefix is not None or bool(motor.plans)
                              or (role == 1 and match.phase == 'receive'))
            action = match.command(tick, position).tolist()
        commands.append(action)
        optical.append(position)
        _, _, done, capped, _ = arena.step(action, render=False)
        if done or capped:
            if restoring:
                raise ValueError('practice point ended during prefix replay')
            break
    result = arena.point_result()
    returned = first_reply(arena.events)
    return dict(seed=seed, server=role, phase=phase, prefix=0 if prefix is None else len(prefix['actions']),
                actions=commands, optical=optical, rgb=camera.digest.hexdigest(), plans=motor.plans,
                followup_planning=motor.followup_planning, events=arena.events, result=result,
                first_rival_reply=returned, credit=measure_credit(arena.events, result, returned))


def schedule(seed, count):
    starts = np.random.default_rng(seed).integers(2**31, size=count).tolist()
    earlier = previous_seeds(PREVIOUS) | previous_seeds(ROOT/'artifacts/tt19')
    if len(set(starts)) != count or set(starts) & earlier:
        raise ValueError('a fresh point schedule overlaps old fixtures')
    return [(launch, -1 if i % 4 < 2 else 1) for i, launch in enumerate(starts)]


def train(folder, match):
    arena = apparatus(False)
    prefixes = training_prefixes()
    starts = schedule(PRACTICE_SEED, len(prefixes))
    rows = {'reset': [], 'ordinary': []}
    try:
        for i, prefix in enumerate(prefixes):
            for phase in PHASES[i % len(PHASES):] + PHASES[:i % len(PHASES)]:
                row = play(arena, match, prefix['seed'], prefix['server'], phase, prefix=prefix)
                rows['reset'].append(row)
                print('reset', row['seed'], phase, row['credit'], flush=True)
        for seed, role in starts:
            group = compare(arena, match, seed, role, PHASES)
            rows['ordinary'].extend(group.values())
            print('ordinary', seed, {k: row['credit'] for k, row in group.items()}, flush=True)
    finally:
        arena.close()
    learned = {}
    for key, trials in rows.items():
        selection, scores = learn_phase(trials)
        learned[key] = dict(phase=selection, scores=scores, attempts=len(trials),
                            control_frames=sum(len(row['actions']) for row in trials),
                            prefix_frames=sum(row['prefix'] for row in trials),
                            visible_estimates=sum(sum(p is not None for p in row['optical']) for row in trials),
                            replied_attempts=sum(row['first_rival_reply'] is not None for row in trials))
    receipt = dict(schema='tt21-continuation-practice-1', source=ARCHIVE.name, source_sha256=_hash(ARCHIVE),
                   factory_sha256=_hash(PREVIOUS/'curriculum-factory.json.gz'),
                   candidates=list(PHASES), ordinary_seed=PRACTICE_SEED, ordinary_schedule=starts,
                   practice=rows, learned=learned)
    receipt['replays'] = replay(receipt)
    _write(folder/TRAIN, receipt)
    print('learned', learned, flush=True)
    return receipt


def compare(arena, match, seed, role, phases):
    rows = {0: play(arena, match, seed, role, 0)}
    for phase in sorted(set(phases) - {0}, key=str):
        nominal = rows[0]
        if nominal['followup_planning']:
            rows[phase] = play(arena, match, seed, role, phase)
        else:
            # No conditional planner invocation means the alternate command
            # tape is exactly the nominal command tape on this start.
            rows[phase] = dict(nominal, phase=phase, same_before_first_continuation=True)
    return rows


def summarize(rows):
    return dict(starts=len(rows), replies=sum(row['first_rival_reply'] is not None for row in rows),
                followup_planned_points=sum(bool(row['followup_planning']) for row in rows),
                continuation_contacts=sum(row['credit']['contact'] for row in rows),
                continuation_returns=sum(row['credit']['legal'] for row in rows),
                second_return_points=sum(row['result']['legal_returns'] >= 2 for row in rows),
                all_returns=sum(row['result']['legal_returns'] for row in rows),
                point_wins=sum(row['result']['points_for'] for row in rows))


def held(folder, match, practice, count):
    if (practice['source_sha256'] != _hash(ARCHIVE)
            or practice['factory_sha256'] != _hash(PREVIOUS/'curriculum-factory.json.gz')):
        raise ValueError('the continuation practice provenance changed')
    learned = {key: learn_fallback(rows) for key, rows in practice['practice'].items()}
    modes = dict(frozen=0, **{name: fit[0] for name, fit in learned.items()})
    starts = schedule(ASSESSMENT_SEED, count)
    if {seed for seed, _ in starts} & {point['seed'] for rows in practice['practice'].values() for point in rows}:
        raise ValueError('held starts overlap continuation practice')
    report = dict(schema='tt21-continuation-held-1', seed=ASSESSMENT_SEED, schedule=starts,
                  modes=modes, guarded_training_credit={name: fit[1] for name, fit in learned.items()},
                  practice_sha256=_hash(folder/TRAIN), matches={}, metrics={})
    for opponent in OPPONENTS:
        arena = apparatus(False, opponent=opponent == 'search')
        try:
            rows = []
            for seed, role in starts:
                group = compare(arena, match, seed, role, modes.values())
                rows.append({name: group[phase] for name, phase in modes.items()})
                print(opponent, seed, {name: row['credit'] for name, row in rows[-1].items()}, flush=True)
            report['matches'][opponent] = rows
            report['metrics'][opponent] = {mode: summarize([row[mode] for row in rows]) for mode in modes}
        finally:
            arena.close()
    report['replays'] = replay(report)
    _write(folder/HELD, report)
    print(report['metrics'], flush=True)


def replay(receipt):
    if 'practice' in receipt:
        sections = {'standard': [row for rows in receipt['practice'].values() for row in rows]}
    else:
        sections = {opponent: [row for variants in sets for row in variants.values()]
                    for opponent, sets in receipt['matches'].items()}
    count = 0
    for opponent, section in sections.items():
        arena = apparatus(False, opponent=opponent == 'search')
        try:
            for row in section:
                arena.reset(seed=row['seed'], server=row['server'], render=False)
                for command in row['actions']:
                    arena.step(command, render=False)
                if (row['result'] != arena.point_result() or
                        json.dumps(row['events']) != json.dumps(arena.events)):
                    raise ValueError(f"physical replay changed for {opponent} point {row['seed']}")
                count += 1
        finally:
            arena.close()
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('train', 'held', 'both', 'verify'), default='both')
    parser.add_argument('--starts', type=int, default=16)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.stage == 'verify':
        print({filename: replay(_read(args.folder/filename)) for filename in (TRAIN, HELD)})
    else:
        match = player()
        try:
            trained = train(args.folder, match) if args.stage in ('train', 'both') else _read(args.folder/TRAIN)
            if args.stage in ('held', 'both'):
                held(args.folder, match, trained, args.starts)
        finally:
            match.player.close()
