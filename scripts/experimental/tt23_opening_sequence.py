"""Run a physically qualified opening correction into uninterrupted native rallies."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt22_joint_practice import READS, draw
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive
from .tt22_rally_trial import alternating
from .tt23_opening_motor import CHECKPOINT, HANDOVER, HELD, MACRO, OUTPUT, PRACTICE, STREAMS
from .tt23_opening_motor import described, select
from .tt23_sequence import LivePoint, OpeningEnded, SequenceStation, milestones, setup, summarize, verify

RECEIPT = OUTPUT / 'opening-sequence.json.gz'
SCHEDULE = 2091836240
CONFIRM = 1527059223
CONFIRM_RECEIPT = OUTPUT / 'opening-sequence-confirmatory.json.gz'
MODES = ('duration', 'neutral', 'permuted', 'duration_stock')


class CorrectedOpeningStation(SequenceStation):
    """Share the native point, cameras and continuation with the reference opening."""

    def __init__(self, macro, evidence, profile, model):
        super().__init__(macro, evidence, profile)
        self.model = model
        self.law = FlightLaw()

    def plan(self, sight, theta, method):
        stock = MotorPrimitive(self.macro, None)
        goal, context = (None, None) if sight is None else described(stock, sight, theta, self.law)
        option, parameters, lateral_error = select(stock, context, self.model, method)
        return goal, option, parameters, lateral_error

    def play(self, seed, lane, theta, *, method='conditional'):
        live = LivePoint(self.arena, self.optics, self.player, self.far)
        try:
            sight = prefix(live, seed, lane)
        except OpeningEnded:
            sight = None
        goal, option, parameters, lateral_error = self.plan(sight, theta, method)
        primitive = MotorPrimitive(parameters, None)
        for tick in range(len(live.actions), self.arena.max_steps):
            if live.terminal:
                break
            live.render_cameras()
            action = primitive.command(tick) if tick < HANDOVER else self.player.command(tick, live.position)
            live.step(action)
        ticks = len(live.actions)
        row = dict(seed=seed, lane=lane, reward=live.reward, longest=alternating(self.arena.events),
                   outcome=self.arena.point_result(), events=self.arena.events.copy(), opening=sight, goal=goal,
                   handover=HANDOVER if ticks > HANDOVER else None, option=option, peak_x_change=lateral_error,
                   actions=live.actions, rival_actions=self.far.tape(ticks), optical=live.sightings,
                   rgb=live.tracker.digest.hexdigest(), shots=self.player.shots, rival=self.rival.decisions.copy())
        return dict(row, **milestones(row))


def roster(schedule=SCHEDULE, count=3):
    rng = np.random.default_rng(schedule)
    points = {width: draw(rng, count, span) for width, span in STREAMS.items()}
    return [dict(width=width, seed=seed, lane=lane)
            for paired in zip(*points.values(), strict=True)
            for width, (seed, lane) in zip(points, paired, strict=True)]


def multi(rows):
    return sum(row['outcome']['legal_returns'] >= 2 for row in rows)


def qualification(pilot):
    cases = pilot['cases']
    if len(cases) != 6 or pilot['verified'] != 24 or any(set(case['results']) != set(MODES) for case in cases):
        raise ValueError('a complete replay-verified six-start pilot is required')
    rows = {name: [case['results'][name] for case in cases] for name in MODES}
    ten = {name: sum(row['longest'] >= 10 for row in results) for name, results in rows.items()}
    repeated = {name: multi(results) for name, results in rows.items()}
    return (ten['duration'] > ten['neutral']
            or (repeated['duration'] >= repeated['neutral'] + 2
                and repeated['duration'] > repeated['duration_stock']))


def variant(name, gains):
    key = 'duration' if name == 'duration_stock' else name
    return np.asarray(gains[key]), 'stock' if name == 'duration_stock' else 'conditional'


def play_variant(station, case, gains, name):
    theta, method = variant(name, gains)
    for prior, replay in case['results'].items():
        goal, option, parameters, error = station.plan(replay['opening'], theta, method)
        reference = station.plan(replay['opening'], *variant(prior, gains))[2]
        if np.array_equal(parameters, reference):
            # With exactly the same initial primitive, the frozen continuation sees
            # the same RGB and has the same commanded body state throughout.
            return dict(replay, goal=goal, option=option, peak_x_change=error, same_primitive_as=prior)
    return station.play(case['seed'], case['lane'], theta, method=method)


def write_result(result, output):
    result['summary'] = {}
    names = dict.fromkeys(name for case in result['cases'] for name in case['results'])
    for name in names:
        rows = [case['results'][name] for case in result['cases'] if name in case['results']]
        result['summary'][name] = dict(summarize(rows), multiple_near=multi(rows))
    result['verified'] = verify(result['cases'])
    _write(output, result)
    print(digest_json(result['metadata']), result['summary'], flush=True)


def merge(paths, output):
    receipts = [_read(path) for path in paths]
    metadata = receipts[0]['metadata']
    if any(receipt['metadata'] != metadata for receipt in receipts):
        raise ValueError('cannot combine different launch schedules')
    rows = [case for receipt in receipts for case in receipt['cases']]
    by_seed = {case['seed']: case for case in rows}
    if len(rows) != len(by_seed) or len(by_seed) != len(metadata['roster']):
        raise ValueError('need nonoverlapping shards for the complete schedule')
    ordered = [by_seed[p['seed']] for p in metadata['roster']]
    planned = metadata['modes']
    if isinstance(planned[0], str):
        planned = [planned] * len(ordered)
    if any(set(row['results']) != set(expected) for row, expected in zip(ordered, planned, strict=True)):
        raise ValueError('shards are missing planned variants')
    write_result(dict(metadata=metadata, cases=ordered), output)


def run(output, *, confirm=False, span=None):
    held, frozen = _read(HELD), _read(CHECKPOINT)
    if not held.get('full_point_gate') or held['weights_sha256'] != _hash(CHECKPOINT):
        raise ValueError('use the frozen motor model that passed the held opening gate')
    standard = setup()
    pilot = _read(RECEIPT) if confirm else None
    if confirm and not qualification(pilot):
        raise ValueError('the continuous-point pilot did not reach the confirmatory gate')
    rostered = roster(CONFIRM, 6) if confirm else roster()
    tested = ([('duration', 'neutral', 'duration_stock')] * 4 + [('duration', 'neutral')] * 8
              if confirm else [MODES] * len(rostered))
    if len({p['seed'] for p in rostered}) != len(rostered) or any(
        {p['seed'] for p in rostered} & {p['seed'] for p in previously}
        for previously in (_read(PRACTICE)['schedule'], held['schedule'], standard['starts'], roster() if confirm else [])
    ):
        raise ValueError('sequence launches overlap earlier evaluation or motor practice')
    metadata = dict(schema='tt23-opening-sequence-1', schedule=CONFIRM if confirm else SCHEDULE, roster=rostered,
                    modes=[list(names) for names in tested] if confirm else list(MODES), weights_sha256=_hash(CHECKPOINT),
                    short_held_sha256=_hash(HELD), continuation_hashes=standard['source_sha256'], handover=HANDOVER)
    if confirm:
        metadata['qualified_pilot_sha256'] = _hash(RECEIPT)
    result = _read(output) if output.exists() else dict(metadata=metadata, cases=[])
    if result['metadata'] != metadata:
        raise ValueError('sequence receipt has different weights or launch schedule')
    begin, end = span or (0, len(rostered))
    if not 0 <= begin < end <= len(rostered):
        raise ValueError('requested span is outside the frozen schedule')
    march = rostered[begin:end]
    archived = {case['seed']: case for case in result['cases']}
    if not archived.keys() <= {case['seed'] for case in march}:
        raise ValueError('existing receipt covers a different span')
    macro = _read(MACRO)['parameters']
    station = CorrectedOpeningStation(macro, standard['evidence'], standard['far_profile'], frozen['model'])
    began = perf_counter()
    try:
        for i, start in enumerate(march, begin):
            case = archived.setdefault(start['seed'], dict(start, results={}))
            result['cases'] = [archived[p['seed']] for p in march if p['seed'] in archived]
            if any(case[key] != value for key, value in start.items()):
                raise ValueError('sequence receipt order differs')
            names = tested[i]
            rotation = i % len(names)
            for name in names[rotation:] + names[:rotation]:
                if name in case['results']:
                    continue
                row = play_variant(station, case, frozen['gains'], name)
                if any(row['opening'] != other['opening'] or row['actions'][:READS] != other['actions'][:READS]
                       for other in case['results'].values()):
                    raise ValueError('sequence treatments differed before the opening decision')
                case['results'][name] = row
                _write(output, result)
                print(i + 1, start['width'], name, row['opening_return'], row['longest'],
                      round(perf_counter() - began, 1), flush=True)
    finally:
        station.close()
    write_result(result, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confirmatory', action='store_true')
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--span', nargs=2, type=int, metavar=('FIRST', 'END'))
    phase.add_argument('--merge', nargs='+', type=Path)
    args = parser.parse_args()
    path = args.output or (CONFIRM_RECEIPT if args.confirmatory else RECEIPT)
    if args.merge:
        merge(args.merge, path)
    else:
        run(path, confirm=args.confirmatory, span=args.span)


if __name__ == '__main__':
    main()
