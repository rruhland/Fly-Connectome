"""Read the next nominal motor command from a causal optical history."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_rally_improve_optical import OpticalPolicy
from .tt24_rally_improve_phase import PhasePolicy
from .tt27_compact import COMBINED, OPTICAL, OUTPUT, CompactStation, Options, audit, brief
from .tt27_compact_motor import PROBE_STREAM, SETTINGS, SOURCE, interface, play
from .tt27_motor_interface import IDENTITY


PILOT = OUTPUT / 'delay_readout-development.json.gz'
HELD = OUTPUT / 'delay_readout-held.json.gz'
COMMAND_HELD = OUTPUT / 'delay_readout-command-held.json.gz'
CANDIDATES = ('future', 'blend', 'state')
CONTROLS = ('timing', 'pixels', 'intact')
DEVELOPMENT = (1261056994, 1321462031, 1956094064, 86399306)
ROSTER_SEED = 1000910279
WORKER = None


def adjacencies(data):
    """Retain same-tape adjacent rows with matching action and measured displacement."""
    states = data['features']
    same_trial = data['seed'][1:] == data['seed'][:-1]
    observed_previous = states[1:, 32:38]
    action_matches = np.all(np.isclose(observed_previous, data['rates'][:-1], rtol=0., atol=1e-7), axis=1)
    measured_delta = states[1:, 20:26] - states[:-1, 20:26]
    displacement_matches = np.all(np.isclose(measured_delta, states[1:, 26:32], rtol=0., atol=1e-7), axis=1)
    return np.flatnonzero(same_trial & action_matches & displacement_matches)


def relabel(data, pairs):
    return dict(features=data['features'][pairs], postures=data['postures'][pairs+1], rates=data['rates'][pairs+1])


class NextFeatures:
    """Fit optical increments in the same local observation neighborhoods."""

    visual = np.r_[np.arange(14), np.arange(38, 54)]

    def __init__(self, data, pairs):
        shifted = relabel(data, pairs)
        self.neighbors = OpticalPolicy(shifted, 'interaction')
        states = data['features']
        self.deltas = (states[pairs+1]-states[pairs])[:, self.visual]

    def predict(self, state):
        model = self.neighbors
        query = state[model.columns]/model.scale
        distance, ids = model.tree.query(query, k=44)
        root = np.exp(-.25*((distance-distance[0])/max(.12, .48*distance[-1]))**2)
        design = np.column_stack((np.ones(len(ids)), model.train[ids]-query))*root[:, None]
        delta = np.linalg.solve(design.T @ design + model.ridge,
                                design.T @ (self.deltas[ids]*root[:, None]))[0]
        forward = state.copy()
        forward[self.visual] += delta
        forward[[6, 13]] = np.clip(forward[[6, 13]], 0., 7.)
        forward[47:49] = np.maximum(forward[47:49], 0.)
        forward[49] = np.clip(forward[49], 0., 7.)
        forward[53] = np.clip(forward[53], 0., 1.)
        return forward


class StateLookahead:
    def __init__(self, nominal, evolution):
        self.nominal = nominal
        self.evolution = evolution

    def command(self, state, issued):
        forward = self.evolution.predict(state)
        forward[14:20] = issued
        return self.nominal.command(forward, issued)


class MixedTiming:
    def __init__(self, nominal, future):
        self.nominal = nominal
        self.future = future

    def command(self, state, issued):
        a = self.nominal.command(state, issued)
        b = self.future.command(state, issued)
        return .35*a + .65*b


class ReadoutOptions:
    def __init__(self):
        self.stock = Options()
        nominal = self.stock.models['combined'][0]
        with np.load(OPTICAL) as original, np.load(COMBINED) as added:
            future = PhasePolicy(OpticalPolicy(relabel(original, adjacencies(original)), 'interaction'),
                                 OpticalPolicy(relabel(added, adjacencies(added)), 'interaction'), .88)
            state = StateLookahead(nominal, NextFeatures(added, adjacencies(added)))
        self.actors = dict(future=future, blend=MixedTiming(nominal, future), state=state)
        self.mode = 'timing'

    def new(self, name):
        chosen = self.stock.new(name)
        chosen.actor = self.actors.get(self.mode, chosen.actor)
        return chosen


def roster(count):
    rng = np.random.default_rng(ROSTER_SEED)
    releases = []
    for index in range(count):
        low, high = (.10, .36) if index % 2 == 0 else (.31, .58)
        # Balanced placements on both sides in the compact course's ordinary range.
        lane = .10 + (-1 if index % 4 < 2 else 1)*rng.uniform(low, high)
        releases.append(dict(seed=int(rng.integers(1, 2**31 - 1)), lane=float(lane), role='fresh'))
    return releases


def old_starts():
    first = _read(OUTPUT / 'compact-assessment.json.gz')
    rows = [case['retained'] for stream in first['cases'].values() for block in stream.values() for case in block]
    second = _read(OUTPUT / 'compact_motor.json.gz')
    rows.extend(case['results']['intact'] for block in second['blocks'] for case in block['cases'])
    by_seed = {row['seed']: row for row in rows}
    return [dict(seed=seed, lane=by_seed[seed]['lane'], role='diagnostic') for seed in DEVELOPMENT]


def audit_fresh(releases):
    proposed = {row['seed'] for row in releases}
    archives = audit([{'held': releases}])
    literals = [str(seed).encode() for seed in proposed]
    for source in OUTPUT.glob('compact*.json.gz'):
        content = gzip.decompress(source.read_bytes())
        for literal in literals:
            field = b'"seed":'+literal
            if field+b',' in content or field+b'}' in content:
                raise ValueError(f'held seed appears in {source.name}')
    return archives


def prepare(modes):
    global WORKER
    data = _read(SOURCE)
    stream = next(row for row in data['streams'] if row['seed'] == PROBE_STREAM)
    fit = next(row for row in stream['settings'] if row['number'] == SETTINGS[-1])
    actual = interface(fit['actual'])
    estimated = interface(fit['estimates']['active']['interface'])
    if estimated != actual or actual.delay != 1:
        raise ValueError('test requires the empirically decoded one-frame FIFO')
    WORKER = CompactStation('rebound'), ReadoutOptions(), actual, estimated, modes


def trial(item):
    index, start = item
    station, options, actual, estimated, modes = WORKER
    results = {}
    offset = index % len(modes)
    for mode in modes[offset:]+modes[:offset]:
        options.mode = mode
        physical = IDENTITY if mode == 'intact' else actual
        fit = IDENTITY if mode == 'intact' else actual if mode in ('timing', 'pixels') else estimated
        result = play(station, options, start, physical, fit, lead=float(mode == 'pixels'))
        results[mode] = result
        print(start['role'], start['seed'], mode, result['outcome']['legal_returns'],
              result['outcome']['rival_returns'], result['longest'], flush=True)
    if len({result['first_rgb'] for result in results.values()}) != 1:
        raise ValueError('paired timing branches saw different first frames')
    return index, dict(start, results=results)


def pick(practice):
    def quality(mode):
        s = practice['summary'][mode]
        return s['second'], s['near'], -s['reversals_per_second']
    return max(CANDIDATES, key=quality)


def run(path, develop, count, workers, command_only=False):
    starts = old_starts() if develop else roster(count)
    if develop:
        selected = None
        modes = (*CANDIDATES, *CONTROLS)
    else:
        practice = _read(PILOT)
        if practice['verified_tapes'] != len(DEVELOPMENT)*(len(CANDIDATES)+len(CONTROLS)):
            raise ValueError('complete the diagnostic comparisons before freezing the held branches')
        selected = 'blend' if command_only else pick(practice)
        modes = (selected,) if command_only else (selected, *CONTROLS)
    metadata = dict(schema='tt27-delay-readout-1', held=not develop, modes=list(modes), selected=selected,
                    releases=starts, archives_checked=None if develop else audit_fresh(starts),
                    practice_sha256=None if develop else _hash(PILOT), motor_probe_sha256=_hash(SOURCE),
                    fit_timing='one-tick-ahead motor labels in original compact demonstrations', source={})
    for source in (OPTICAL, COMBINED):
        with np.load(source) as data:
            metadata['source'][source.name] = dict(sha256=_hash(source), paired_rows=len(adjacencies(data)),
                                                  launches=len(set(data['seed'].tolist())))
    report = _read(path) if path.exists() else dict(metadata=metadata, cases=[None]*len(starts))
    if report['metadata'] != metadata:
        raise ValueError('stored timing evaluation has different inputs')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(modes,)) as pool:
        futures = [pool.submit(trial, (i, start)) for i, start in enumerate(starts) if report['cases'][i] is None]
        for future in as_completed(futures):
            index, case = future.result()
            report['cases'][index] = case
            complete = [row for row in report['cases'] if row is not None]
            report['verified_tapes'] = len(complete)*len(modes)
            report['summary'] = {mode: brief([row['results'][mode] for row in complete]) for mode in modes}
            _write(path, report)
    print(report['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--develop', action='store_true')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--command-only', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.develop and args.command_only:
        parser.error('--command-only assesses the additional held blend branch')
    output = PILOT if args.develop else COMMAND_HELD if args.command_only else HELD
    run(args.output or output, args.develop, args.count, args.workers, args.command_only)
