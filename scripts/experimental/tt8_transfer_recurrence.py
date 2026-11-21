"""Recognize returning rebound dynamics from past RGB crossings."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_transfer_model import Catcher, optical_prior
from .tt8_transfer_rebound import LIMIT, ReboundCourt, pixels
from .tt8_transfer_switch import CHANGE_HAZARD, SwitchingCatcher
from .tt8_transfer_trial import MATERIALS, play, verify


CROSSING_NOISE = .045


class RecognitionCatcher(Catcher):
    """Use a terminal RGB observation to compare two recurring rebound models."""

    def __init__(self, modes, *, calibrate):
        self.modes = modes
        self.state = np.array([.5, .5])
        self.calibrate = calibrate
        self.offset_samples = 2.
        self.offset_sum = 0.
        self._report_state()

    @property
    def offset(self):
        return self.offset_sum / self.offset_samples

    def _report_state(self):
        self.log_probability = np.log(self.state @ self.modes)

    def reset(self):
        self.state = (1. - CHANGE_HAZARD) * self.state + CHANGE_HAZARD * self.state[::-1]
        self._report_state()
        super().reset()

    def _aim(self):
        earlier = self.pre_bounce
        super()._aim()
        if self.pre_bounce is not earlier:
            self.direction = np.sign(self.pre_bounce[-1] - self.pre_bounce[0])
            self.goal = float(np.clip(self.goal + self.direction * self.offset, -LIMIT, LIMIT))

    def finish(self, rgb):
        ball, _ = pixels(rgb)
        prediction = self.pre_bounce
        if ball is None or prediction is None:
            return
        error = (ball[0] - prediction - self.direction * self.offset) / CROSSING_NOISE
        likelihood = self.modes @ np.exp(-.5 * np.minimum(36., error**2))
        self.state = self.state * likelihood
        self.state /= self.state.sum()
        if self.calibrate:
            strength = 2. * self.state.max() - 1.
            expected = self.state @ self.modes @ prediction
            signed_error = self.direction * (ball[0] - expected)
            self.offset_sum += strength * np.clip(signed_error, -.12, .12)
            self.offset_samples += strength
        self._report_state()


def players(donors):
    learned = [(data['center'], data['spread']) for data in donors.values()]
    generic = [( .94, .055), (.65, .070)]
    result = {}
    for name, distribution in [('source', learned), ('generic', generic)]:
        options = np.array([Catcher(*parameters, learn=False).probability for parameters in distribution])
        result[f'{name}_remix'] = SwitchingCatcher(options.mean(axis=0))
        result[f'{name}_recognition'] = RecognitionCatcher(options, calibrate=False)
        result[f'{name}_calibrated'] = RecognitionCatcher(options, calibrate=True)
    return result


def summarize(rows, stations):
    rng = np.random.default_rng(956429)
    station_draw = rng.integers(0, stations, (5000, stations))
    station_ids = np.array([row['station'] for row in rows])
    first = np.array([row['within_segment'] == 0 for row in rows])
    switching = np.array([row['segment'] > 0 for row in rows])
    returning = np.array([row['segment'] == 2 for row in rows])
    recovery = np.array([0 < row['within_segment'] <= 2 for row in rows])
    outcome = {name: np.array([row['players'][name]['catch'] for row in rows], dtype=int)
               for name in rows[0]['players']}
    stats = {}
    for name, totals in outcome.items():
        stats[name] = dict(catches=int(totals.sum()), first_at_entry=int(totals[first & ~switching].sum()),
                           first_switch=int(totals[first & switching].sum()),
                           first_return=int(totals[first & returning].sum()),
                           next_two_switch=int(totals[recovery & switching].sum()),
                           next_two_return=int(totals[recovery & returning].sum()))
    comparisons = {}
    sizes = np.bincount(station_ids)[station_draw].sum(axis=1)
    for name, base in [('source_recognition', 'source_remix'), ('source_calibrated', 'source_remix'),
                       ('source_calibrated', 'source_recognition'), ('generic_recognition', 'generic_remix'),
                       ('generic_calibrated', 'generic_recognition'), ('source_calibrated', 'generic_calibrated'),
                       ('source_recognition', 'generic_recognition')]:
        delta = outcome[name] - outcome[base]
        gains = np.bincount(station_ids, weights=delta)[station_draw].sum(axis=1)
        boot = 100. * gains / sizes
        comparisons[f'{name}_vs_{base}'] = dict(mean_pp=float(100. * delta.mean()),
                                                interval_pp=np.percentile(boot, [2.5, 97.5]).tolist())
    return dict(by_condition=stats, comparisons=comparisons)


def run(source, seed, stations):
    receipt = _read(source)
    donors = {name: optical_prior(receipt, name) for name in MATERIALS}
    rng = np.random.default_rng(seed)
    initial = np.resize(np.array(MATERIALS), stations)
    rng.shuffle(initial)
    court = {name: ReboundCourt(name) for name in MATERIALS}
    rows = []
    for station, first in enumerate(initial):
        second = next(name for name in MATERIALS if name != first)
        agents = players(donors)
        for j, material in enumerate([str(first), second, str(first)]):
            count = rng.integers(3, 7)
            for i, fall in enumerate(rng.integers(0, 2**31 - 1, size=count).tolist()):
                opening = court[material].opening(fall)
                tapes = {}
                for name, agent in agents.items():
                    result = play(court[material], opening, agent, visibility='curtain')
                    if isinstance(agent, RecognitionCatcher):
                        result.update(posterior=agent.state.tolist(), calibrated_offset_m=float(agent.offset))
                    tapes[name] = result
                rows.append(dict(station=station, material=material, segment=j, within_segment=i,
                                 fall=fall, visibility='curtain', crossing_m=opening.endpoint, time_s=opening.crossing_time,
                                 players=tapes))
    return dict(schema='tt8-blind-rebound-recurrence-v1', seed=seed, stations=stations,
                source_sha256=_hash(source), donors=donors, rows=rows,
                implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                       ('tt4_arena.xml', 'tt7_surface_trial.py', 'tt8_transfer_rebound.py',
                                        'tt8_transfer_model.py', 'tt8_transfer_trial.py', 'tt8_transfer_switch.py',
                                        'tt8_transfer_recurrence.py')},
                summary=summarize(rows, stations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt7/surface-read-only.json.gz')
    parser.add_argument('--seed', type=int, default=65710314)
    parser.add_argument('--stations', type=int, default=16)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify(_read(args.output))} rebound catch action tapes.')
        return
    if args.stations < 2 or args.stations % 2:
        parser.error('use an even number of stations')
    receipt = run(args.source, args.seed, args.stations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, receipt)
    print(len(receipt['rows']), 'falls', receipt['summary'])


if __name__ == '__main__':
    main()
