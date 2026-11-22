"""Collect streaming physical outcomes for three fixed jogs on fresh full rally starts."""

import argparse
import gzip
from pathlib import Path
import re

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_duration_transfer import STREAMS
from .tt22_joint_practice import READS, draw
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_target import target_context
from .tt23_coach import PRACTICE as COACH
from .tt23_jog_station import ExplicitJogStation
from .tt23_opening_motor import CHECKPOINT, HANDOVER, HELD, OUTPUT
from .tt23_sequence import setup, summarize, verify


GENERATORS = (37418286, 189532461, 1486349668, 308641972)
OPTIONS = {'left': -.09, 'center': 0., 'right': .09}
PER_STRATUM = 4


def schedule(generator):
    rng = np.random.default_rng(generator)
    widths = {name: draw(rng, PER_STRATUM, bounds) for name, bounds in STREAMS.items()}
    return [dict(stratum=name, seed=seed, lane=lane) for pair in zip(*widths.values(), strict=True)
            for name, (seed, lane) in zip(widths, pair, strict=True)]


def check_archives():
    seeds = {case['seed'] for generator in GENERATORS for case in schedule(generator)}
    if len(seeds) != len(GENERATORS) * PER_STRATUM * len(STREAMS):
        raise ValueError('the four streams must contain distinct launch IDs')
    numbers = b'|'.join(str(seed).encode() for seed in sorted(seeds))
    identifier = re.compile(rb'(?<![0-9.])(?:' + numbers + rb')(?![0-9.])')
    for directory in (OUTPUT.with_name('tt22'), OUTPUT):
        for path in directory.iterdir():
            if path.name.startswith('jog-bank-'):
                continue
            if path.name.endswith('.json.gz'):
                text = gzip.decompress(path.read_bytes())
            elif path.suffix == '.json':
                text = path.read_bytes()
            else:
                continue
            found = identifier.search(text)
            if found is not None:
                raise ValueError(f'new launch {found[0].decode()} already occurs in {path}')


def metadata(stream):
    held = _read(HELD)
    if not held['full_point_gate'] or held['weights_sha256'] != _hash(CHECKPOINT):
        raise ValueError('use the frozen physically qualified near stroke correction')
    common = setup()
    return dict(schema='tt23-jog-full-rally-bank-1', stream=stream, generator=GENERATORS[stream - 1],
                roster=schedule(GENERATORS[stream - 1]), options=OPTIONS, handover=HANDOVER, prefix_frames=READS,
                correction_sha256=_hash(CHECKPOINT), correction_held_sha256=_hash(HELD),
                coach_sha256=_hash(COACH), continuation_sha256=common['source_sha256'],
                far_profile=common['far_profile'], arena=common['arena'])


def issued_prefix(parameters, length):
    primitive = MotorPrimitive(parameters, None)
    return [[0.] * 6 if tick < READS else primitive.command(tick).tolist()
            for tick in range(min(HANDOVER, length))]


def play(station, case, action):
    for prior, row in case['results'].items():
        goal, option, parameters, error = station.plan(row['opening'], action, 'conditional')
        requested = issued_prefix(parameters, len(row['actions']))
        if requested == row['actions'][:len(requested)]:
            return dict(row, goal=goal, option=option, peak_x_change=error, jog=action,
                        execution='identical_issued_prefix', same_commands_as=prior)
    row = station.play(case['seed'], case['lane'], action)
    return dict(row, jog=action, execution='live', same_commands_as=None)


def audit_case(case, station):
    rows = case['results']
    if set(rows) != set(OPTIONS):
        raise ValueError('each scored start needs all three alternatives')
    for name, row in rows.items():
        if (row['seed'] != case['seed'] or row['lane'] != case['lane'] or row['jog'] != OPTIONS[name]
                or row['opening'] != case['opening']):
            raise ValueError('the option did not use the same registered physical and optical prefix')
        provenance = row['same_commands_as']
        if provenance is not None:
            original = rows[provenance]
            goal, option, parameters, error = station.plan(case['opening'], OPTIONS[name], 'conditional')
            prefix = issued_prefix(parameters, len(original['actions']))
            if prefix != original['actions'][:len(prefix)] or (goal, option, error) != (
                    row['goal'], row['option'], row['peak_x_change']):
                raise ValueError('the shared tape no longer follows the requested optical motor plan')
            if any(row[key] != original[key] for key in ('actions', 'rival_actions', 'events', 'outcome', 'rgb')):
                raise ValueError('a reused option does not carry the identical physical trace')
    return verify([case])


def collect(path, stream, span=None):
    frozen = metadata(stream)
    first, end = span or (0, len(frozen['roster']))
    if not 0 <= first < end <= len(frozen['roster']):
        raise ValueError('the requested span is outside the registered stream')
    roster = frozen['roster'][first:end]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = _read(path)
    else:
        check_archives()
        data = dict(metadata=frozen, cases=[], verified=0, complete=False)
    if digest_json(data['metadata']) != digest_json(frozen):
        raise ValueError('bank checkpoint uses another schedule or physical controller')
    cases = data['cases']
    if any({key: row[key] for key in visit} != visit
           for row, visit in zip(cases, roster[:len(cases)], strict=True)):
        raise ValueError('the saved starts do not form a prefix of the registered roster')
    _write(path, data)
    station = ExplicitJogStation(lambda action, _sight, _ready: action)
    ready = MotorPrimitive(station.macro, None).ready_world
    try:
        for i, start in enumerate(roster):
            if i == len(cases):
                cases.append(dict(start, results={}, verified=0))
            case = cases[i]
            options = tuple(OPTIONS)
            offset = (first + i + stream - 1) % len(options)
            for name in options[offset:] + options[:offset]:
                if name in case['results']:
                    continue
                row = play(station, case, OPTIONS[name])
                sight = row['opening']
                if not case['results']:
                    context = None if sight is None else target_context(sight, ready)
                    case.update(opening=sight, context=None if context is None else np.asarray(context).tolist())
                elif row['opening'] != case['opening'] or any(
                        row['actions'][:READS] != prior['actions'][:READS] for prior in case['results'].values()):
                    raise ValueError('matched policies differ before the jog command')
                case['results'][name] = row
                _write(path, data)
                print(stream, i + 1, name, row['execution'], row['longest'], len(row['actions']), flush=True)
            if not case['verified']:
                case['verified'] = audit_case(case, station)
                data['verified'] = sum(row['verified'] for row in cases)
                _write(path, data)
                print(stream, i + 1, 'verified alternatives', data['verified'], flush=True)
    finally:
        station.close()
    if len(cases) != len(roster) or data['verified'] != len(cases) * len(OPTIONS):
        raise ValueError('the registered bank is incomplete')
    data['summary'] = {name: summarize([row['results'][name] for row in cases]) for name in OPTIONS}
    data['live'] = sum(row['execution'] == 'live' for case in cases for row in case['results'].values())
    data['complete'] = len(cases) == len(frozen['roster'])
    _write(path, data)
    print('complete', stream, data['verified'], 'live', data['live'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=int, choices=range(1, len(GENERATORS) + 1), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--span', type=int, nargs=2, metavar=('FIRST', 'END'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    path = args.output or OUTPUT / f'jog-bank-{args.stream}.json.gz'
    if args.verify:
        data = _read(path)
        if digest_json(data['metadata']) != digest_json(metadata(args.stream)):
            raise ValueError('receipt has a different stream or frozen physical controller')
        station = ExplicitJogStation(lambda action, _sight, _ready: action)
        try:
            print('verified', sum(audit_case(case, station) for case in data['cases']
                                  if set(case['results']) == set(OPTIONS)))
        finally:
            station.close()
    else:
        collect(path, args.stream, span=args.span)


if __name__ == '__main__':
    main()
