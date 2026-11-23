"""Test Pong visual credit with a framewise, demonstration-assisted SO-101 pupil."""

import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import find_objects, label

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import HANDOVER, course, verify
from .tt24_rally_fresh import OpeningTape, SETTINGS
from .tt24_rally_policy import EYES, ViewHistory
from .tt24_rally_trial import COACHED, DATA, DEVELOPMENT, OUTPUT, Station, archive_schedule
from .tt24_rally_trial import difference_count, make_policy, returns, summary
from .tt24_source import CHECKPOINT, MODES, VisualConsequence


RECEIPT = OUTPUT / 'source-rally-credit.json.gz'
SCHEDULE = 1200549009
CANDIDATES = 7
PRACTICE = 3
FRESH_COHORTS = {'source_rally': 1496828947}
CREDIT_WEIGHT = 1.25
CREDIT_FRAMES = 8


def blade_sighting(rgb):
    """Read the ordinary near-side blade's local red silhouette in one eye."""
    step = 1 if rgb.shape[0] <= 180 else 2
    pixel = step * 360. / rgb.shape[0]
    red, green, blue = rgb[::step, ::step].transpose(2, 0, 1)
    mask = (red > 76) & (green < .52 * red) & (blue < .62 * red) & (green < 1.15 * blue)
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    top, left = y.min(), x.min()
    regions, count = label(mask[top:y.max() + 1, left:x.max() + 1])
    components = []
    for i, (ys, xs) in enumerate(find_objects(regions, count), 1):
        yy, xx = np.nonzero(regions[ys, xs] == i)
        size = max(ys.stop - ys.start, xs.stop - xs.start)
        if len(xx) >= 4 and size < .18 * red.shape[0]:
            point = pixel * np.array((left + xs.start + xx.mean(), top + ys.start + yy.mean()))
            components.append((len(xx), point, .5 * pixel * size))
    if components:
        _, coordinate, span = max(components, key=lambda item: item[0])
        return coordinate, span
    return None


class CreditHistory(ViewHistory):
    """Keep the demonstration learner's input and measure anonymous-scale cue support."""

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.optical_pairs = {name: deque(maxlen=4) for name in EYES}
        self.tick = -1
        self.first_support = None
        self.record = []

    def observe(self, frames, joints, issued_position):
        state = super().observe(frames, joints, issued_position)
        self.tick += 1
        if self.tick < HANDOVER or (self.first_support is not None and
                                   self.tick - self.first_support >= CREDIT_FRAMES):
            return state
        readings = []
        for name, ball_track in zip(EYES, self.views, strict=True):
            ball = ball_track[-1]
            blade = blade_sighting(frames[name]) if ball is not None else None
            history = self.optical_pairs[name]
            if blade is None:
                history.clear()
                continue
            actor, halfspan = blade
            history.append((actor, ball, halfspan))
            if len(history) != history.maxlen:
                continue
            a, b, s = tuple(np.asarray(parts) for parts in zip(*history, strict=True))
            fit = self.models['pong'].score_history(a, b, dt=1. / 30., scale=np.median(s), details=True)
            scores = {mode: float(model.score_features(fit['features']) - model.prior)
                      for mode, model in self.models.items()}
            support = float(fit['support'])
            if np.isfinite(fit['features']).all():
                readings.append((support, name, scores, fit['features'].tolist()))
        if readings:
            # Both cameras measure the same event. Use support for one view only.
            support, name, scores, features = max(readings, key=lambda r: r[0])
            if support > 0. and self.first_support is None:
                self.first_support = self.tick
            if self.first_support is not None:
                self.record.append(dict(tick=self.tick, eye=name, support=support, scores=scores, features=features))
        return state

    def average(self, mode):
        return float(np.mean([record['scores'][mode] for record in self.record])) if self.record else 0.


class ReadoutCorrection:
    """Modify each new RGB-feedback joint rate by a learned bounded local calibration."""

    def __init__(self, base, parameters):
        self.base, self.parameters = base, np.asarray(parameters, float)

    def command(self, state, target):
        drive = self.base.command(state, target)
        gain, correction = self.parameters.T
        return np.clip(drive * (1. + .25 * gain) + .11 * correction * (1. - .55 * np.abs(drive)), -1., 1.)


def population():
    rng = np.random.default_rng(SCHEDULE)
    samples = np.tanh(rng.normal(0., .65, size=((CANDIDATES - 1) // 2, 6, 2)))
    return np.concatenate((np.zeros((1, 6, 2)), samples, -samples))


def score(station, start, base, parameters, models):
    credit = CreditHistory(models)
    row = station.play(start, 'source_credit', ReadoutCorrection(base, parameters), new_history=lambda: credit)
    duration = (len(row['actions']) - HANDOVER) / 30.
    fitness = float(np.log1p(duration / .65) + .25 * row['reward'])
    return dict(row, source_cues=credit.record, visual_credit={mode: credit.average(mode) for mode in MODES},
                elapsed_after_opening=duration, public_fitness=fitness)


def chosen(rows):
    tables = {mode: [np.mean([entry['results'][str(i)]['public_fitness'] +
                             CREDIT_WEIGHT * entry['results'][str(i)]['visual_credit'][mode] for entry in rows])
                     for i in range(CANDIDATES)] for mode in MODES}
    picks = {mode: int(np.argmax(table)) for mode, table in tables.items()}
    return picks, {mode: [float(v) for v in table] for mode, table in tables.items()}


def fresh_openings(opening, station, starts, maximum):
    tape = OpeningTape(station)
    eligible, screening = [], []
    for start in starts:
        record = dict(start, tested=[])
        for mode in ('neutral', 'duration'):
            row, good = tape.try_opening(start, opening['macro'], opening['gains'][mode])
            record['tested'].append(dict(opening=mode, eligible=good, continuation=returns(row, -1)))
            if good:
                eligible.append(row)
                break
        screening.append(record)
        if len(eligible) == maximum:
            break
    return eligible, screening


def run(path, data_path, fresh_count, render_dims):
    began = perf_counter()
    archive = archive_schedule()
    # Reserve each practice release by ordinal in the independent archived roster.
    indices = set(np.linspace(0, len(archive) - 1, PRACTICE, dtype=int))
    train = [row for i, row in enumerate(archive) if i in indices]
    held = [row for i, row in enumerate(archive) if i not in indices]
    starts = course(FRESH_COHORTS)
    known = {row['seed'] for row in archive} | set(DEVELOPMENT)
    if {row['seed'] for row in starts} & known:
        raise ValueError('new physical courses overlap the recorded practice')
    draw = population()
    metadata = dict(schema='tt24-pong-rally-credit-1', source_sha256=_hash(CHECKPOINT),
                    source_modes=list(MODES), practice_starts=[row['seed'] for row in train],
                    archived_held_starts=[row['seed'] for row in held], fresh_schedule=starts, fresh_count=fresh_count,
                    demo_data_sha256=_hash(data_path), archive_sha256=_hash(COACHED), emission=draw.tolist(),
                    credit_weight=CREDIT_WEIGHT, shared_opening_frames=HANDOVER, credit_frames=CREDIT_FRAMES,
                    render_dimensions=list(render_dims))
    report = _read(path) if path.exists() else dict(metadata=metadata, practice=[
        dict(seed=row['seed'], results={}) for row in train], held=[])
    if report['metadata'] != metadata:
        raise ValueError('rally credit receipt does not use these frozen sources and schedules')
    opening = _read(COACHED)['metadata']['opening']
    station = Station(opening['far_profile'], render_dims=render_dims)
    with np.load(data_path) as data:
        base, exposure = make_policy(data, 'demo_full', **SETTINGS)
    report['demonstration_exposure'] = exposure
    models = {mode: VisualConsequence.load(mode) for mode in MODES}
    try:
        for start, row in zip(train, report['practice'], strict=True):
            for i, candidate in enumerate(draw):
                key = str(i)
                if key not in row['results']:
                    row['results'][key] = score(station, start, base, candidate, models)
                    played = row['results'][key]
                    print('practice', start['seed'], i, len(played['actions']), played['visual_credit'],
                          len(played['source_cues']), round(perf_counter() - began, 1), flush=True)
                    _write(path, report)
        picks, fitness = chosen(report['practice'])
        report.update(selected=picks, practice_fitness=fitness)
        print('frozen target feedback corrections', picks, flush=True)
        if fresh_count and 'fresh' not in report:
            # The common teacher's old motion-prefix reader was calibrated at
            # its original camera resolution. The pupil uses its own pixels.
            screen = Station(opening['far_profile'])
            try:
                report['fresh'], report['fresh_screening'] = fresh_openings(opening, screen, starts, fresh_count)
            finally:
                screen.arena.close()
            _write(path, report)
        assessment = [(row, 'archived_held') for row in held] + [(row, 'fresh_held') for row in report.get('fresh', ())]
        if not report['held']:
            report['held'] = [dict(seed=row['seed'], cohort=cohort, results={}) for row, cohort in assessment]
        for index, ((start, _), row) in enumerate(zip(assessment, report['held'], strict=True)):
            order = MODES if index % 2 == 0 else MODES[::-1]
            for mode in order:
                if mode in row['results']:
                    continue
                candidate = picks[mode]
                same = next((earlier for earlier, existing in row['results'].items()
                             if existing['target_checkpoint'] == candidate), None)
                played = score(station, start, base, draw[candidate], models) if same is None else row['results'][same].copy()
                played.update(target_checkpoint=candidate, identical_checkpoint_to=same)
                row['results'][mode] = played
                print('held', start['seed'], mode, returns(played, 1), len(played['actions']),
                      round(perf_counter() - began, 1), flush=True)
                _write(path, report)
            if len({r['prefix_rgb'] for r in row['results'].values()}) != 1:
                raise ValueError('matched rally treatments differed before handoff')
            for mode in MODES:
                row['results'][mode]['vs_cold'] = difference_count(row['results'][mode], row['results']['cold'])
        report['verified_tapes'] = verify(report['held'])
        cohorts = dict.fromkeys(cohort for _, cohort in assessment)
        report['summary'] = {cohort: {mode: summary([r for r in report['held'] if r['cohort'] == cohort], mode)
                                      for mode in MODES} for cohort in cohorts}
        report['seconds'] = perf_counter() - began
        _write(path, report)
        print(report['summary'], flush=True)
    finally:
        station.arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=RECEIPT, type=Path)
    parser.add_argument('--data', default=DATA, type=Path)
    parser.add_argument('--fresh-count', type=int, default=3)
    parser.add_argument('--render-dimensions', type=int, nargs=2, default=(240, 180))
    args = parser.parse_args()
    run(args.output, args.data, args.fresh_count, args.render_dimensions)


if __name__ == '__main__':
    main()
