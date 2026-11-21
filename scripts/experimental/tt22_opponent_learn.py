"""Fit a cooperative physical feed profile from optical continuation outcomes."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt20_natural import NaturalViewTransport
from .tt22_opponent import ANCHOR, RallyArena, install_rally_opponent
from .tt22_opponent_trial import FarTape, OUTPUT
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, GentleDraws, Station
from .tt19_view_full_game import digest_json


SOURCES = ('rally-anchor-612423769.json.gz', 'rally-coach-612423769.json.gz', 'tt22_opponent-native.json.gz')
DEPTHS = (-.86, -.90, -.94, -.98)
GRID = list(product(DEPTHS, (.15, .175, .20, .225, .25), (1.8, 2.1, 2.4, 2.7, 3.)))
SCALES = np.array([.114, .055, .52])  # x, height and speed, in metres and seconds
WIDTHS = (.72, 1., 1.42)
HELD_SEED = 1009939329
PHASE_HELD_SEED = 1091203814
SPREAD, PACE = .55, .90


def _frozen_rows(path):
    source = _read(path)
    if 'variants' in source:
        return [row for section in source['variants'].values() for row in section['rows']]
    if source.get('privileged', False):
        raise ValueError('the feed fit requires a nonprivileged optical learner')
    return source['rows']


def _forecast(optical, start, stop, phase):
    """Use the first optical approach estimate common to hits and misses."""
    hist = []
    law = FlightLaw()
    for tick, point in optical:
        if tick <= start or tick >= stop:
            continue
        hist.append((tick, point))
        if not -.72 < point[1] < -.49:
            continue
        seen = optical_start(hist, law.gravity)
        if seen is None or seen[1][1] > -.022:
            continue
        arc, velocity = law.trajectory(*seen, 25, bounds=(.424, .80))
        features = []
        for depth in DEPTHS:
            crossed = np.flatnonzero((arc[:-1, 1] >= depth) & (arc[1:, 1] < depth))
            if not len(crossed):
                features.append(None)
                continue
            step = crossed[0]
            a, b = arc[step:step + 2]
            share = (depth - a[1]) / (b[1] - a[1])
            intercept = a + share * (b - a)
            velocity_here = 30. * (velocity[step] + share * (velocity[step+1] - velocity[step]))
            feature = [float(intercept[0]), float(intercept[2]), float(-velocity_here[1])]
            if phase:
                feature += [float(velocity_here[0]), float(velocity_here[2])]
            features.append(feature)
        if any(value is not None for value in features):
            return tick, features
    return None, [None] * len(DEPTHS)


def examples(sources, *, phase=False):
    samples = []
    for source in sources:
        for row in _frozen_rows(source):
            events = row['events']
            for index, bounce in enumerate(events):
                if bounce['kind'] != 'legal_return' or bounce['side'] != -1:
                    continue
                contact = next(e for e in reversed(events[:index]) if e['kind'] == 'contact' and e.get('side') == -1)
                later = events[index + 1:]
                end = next((i for i,e in enumerate(later) if e['kind'] == 'contact' and e.get('side') == -1), len(later))
                later = later[:end]
                own = next((e for e in later if e['kind'] == 'contact' and e.get('side') == 1), None)
                stop = own['step'] if own else len(row['actions'])
                tick, features = _forecast(row['optical'], contact['step'], stop, phase)
                legal = any(e['kind'] == 'legal_return' and e.get('side') == 1 for e in later)
                samples.append(dict(source=source.name, seed=row['seed'], reply_tick=contact['step'],
                                    landing_tick=bounce['step'], optical_tick=tick, features=features, legal=legal))
    return samples


class Response:
    """A seed-balanced local estimate of legal continuation probability."""

    def __init__(self, samples, depth, width):
        slot = DEPTHS.index(depth)
        rows = [row for row in samples if row['features'][slot] is not None]
        seeds = [row['seed'] for row in rows]
        self.x = np.asarray([row['features'][slot] for row in rows])
        self.y = np.asarray([row['legal'] for row in rows], float)
        self.weights = np.asarray([1. / seeds.count(seed) for seed in seeds])
        self.scales = width * (np.r_[SCALES, .47, .84] if self.x.shape[1] == 5 else SCALES)
        self.prior = float(np.average(self.y, weights=self.weights))

    def predict(self, feature):
        distance = np.linalg.norm((self.x - feature) / self.scales, axis=1)
        weight = self.weights * np.exp(-.5 * distance**2)
        support = float(weight.sum())
        probability = float((weight @ self.y + 1.4 * self.prior) / (support + 1.4))
        return probability, support


def _folds(samples):
    seeds = sorted({row['seed'] for row in samples})
    return {seed: ([row for row in samples if row['seed'] != seed],
                   [row for row in samples if row['seed'] == seed]) for seed in seeds}


def _validate(folds, depths):
    validation = {}
    for width in WIDTHS:
        brier = []
        for train, test in folds.values():
            errors = []
            for depth in depths:
                slot = DEPTHS.index(depth)
                model = Response(train, depth, width)
                errors.extend((model.predict(row['features'][slot])[0] - row['legal'])**2
                              for row in test if row['features'][slot] is not None)
            if errors:
                brier.append(float(np.mean(errors)))
        validation[width] = float(np.mean(brier))
    width = min(validation, key=validation.get)
    return width, validation


def _record(profile, samples, folds, width, validation, rankings):
    return dict(profile=profile, training_points=len(folds),
                legal_episodes=sum(row['legal'] for row in samples), input_episodes=len(samples),
                fitted_episodes=sum(row['optical_tick'] is not None for row in samples),
                brier_by_bandwidth=validation, width=width, ranking=rankings)


def fit(samples):
    folds = _folds(samples)
    width, validation = _validate(folds, DEPTHS)
    responses = {depth: [Response(train, depth, width) for train, _ in folds.values()] for depth in DEPTHS}
    rankings = []
    for depth, height, speed in GRID:
        profile = [0., depth, height, speed]
        estimates = [model.predict([0., height, speed]) for model in responses[depth]]
        probabilities, support = np.asarray(estimates).T
        # Prefer a repeatable improvement with observed support across whole starts.
        conservative = float(np.mean(probabilities) - .55 * np.std(probabilities))
        score = conservative + .022 * np.log1p(float(np.mean(support))) - .12 * abs(depth - ANCHOR[1])
        rankings.append(dict(profile=profile, score=float(score), expected=float(np.mean(probabilities)),
                             support=float(np.mean(support))))
    rankings.sort(key=lambda item: item['score'], reverse=True)
    return _record(rankings[0]['profile'], samples, folds, width, validation, rankings)


def fit_phase(samples):
    # The phase intervention keeps the original target point and pace.
    depth = min(DEPTHS, key=lambda item: abs(item - ANCHOR[1]))
    folds = _folds(samples)
    width, validation = _validate(folds, (depth,))
    models = [Response(train, depth, width) for train, _ in folds.values()]
    rankings = []
    for vx, vz in product((-.24, 0., .24), (-1.6, -1.2, -.8, -.4, 0., .4, .8, 1.2, 1.6, 2.0)):
        estimates, support = np.asarray([model.predict([ANCHOR[0], ANCHOR[2], ANCHOR[3], vx, vz])
                                        for model in models]).T
        score = float(np.mean(estimates) - .55 * np.std(estimates) + .022 * np.log1p(np.mean(support)))
        rankings.append(dict(profile=[*ANCHOR, vx, vz], score=score, expected=float(np.mean(estimates)),
                             support=float(np.mean(support))))
    rankings.sort(key=lambda item: item['score'], reverse=True)
    return _record(rankings[0]['profile'], samples, folds, width, validation, rankings)


class JitteredOpening(RallyArena):
    def _opening(self):
        if self.server == 1:
            return super()._opening()
        rng = self._rng
        self._rng = GentleDraws(rng, SPREAD, PACE)
        try:
            # Keep the live release generator and change only the predeclared spread.
            super()._opening()
        finally:
            self._rng = rng


def _station(profile):
    station = Station.__new__(Station)
    law = FlightLaw()
    station.arena = JitteredOpening(**CONFIG)
    station.rival = install_rally_opponent(station.arena, target_profile=profile)
    station.player = RallyPlayer(law)
    station.optics = NaturalViewTransport('fixed', lens_fov=station.arena.lens_intrinsics(), law=law,
                                           flight=False, associate=True)
    return station


def _audit_motion(arena):
    samples = []
    physical_step = arena.step

    def step(action, *, render=True):
        samples.append([arena.steps, arena.rival_returns, arena._last_hitter, arena._received_bounce,
                        *arena.ball.tolist(), *arena.velocity.tolist()])
        return physical_step(action, render=render)

    arena.step = step
    return samples


def _observed_phase(rows):
    result = []
    for row in rows:
        samples = np.asarray(row['evaluator_motion'])
        if not len(samples):
            continue
        usable = samples[(samples[:, 1] > 0.) & (samples[:, 2] == -1.) & (samples[:, 3] == 1.)
                         & (samples[:, 5] < -.77) & (samples[:, 5] > -1.12) & (samples[:, 8] < -.45)]
        for reply in np.unique(usable[:, 1]):
            part = usable[usable[:, 1] == reply]
            point = part[np.argmin(abs(part[:, 5] + .91))]
            result.append(point.tolist())
    return result


def _replay(rows):
    arena = JitteredOpening(**CONFIG)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=-1, render=False)
            for learner, rival in zip(row['actions'], row['rival_actions'], strict=True):
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(rival)
                _, reward, _, _, _ = arena.step(learner, render=False)
            if (arena.point_result() != row['outcome'] or reward != row['reward']
                    or digest_json(arena.events) != digest_json(row['events'])):
                raise ValueError('the independent physical replay changed the result')
    finally:
        arena.close()
    return len(rows)


def held(path, evidence, model, count, *, schedule_seed=HELD_SEED):
    seeds = np.random.default_rng(schedule_seed).integers(2**31, size=count).tolist()
    if set(seeds).intersection(row['seed'] for row in evidence):
        raise ValueError('held seeds overlap fit source')
    profiles = {'fixed': tuple(ANCHOR), 'learned': tuple(model['profile'])}
    histories = {name: [] for name in profiles}
    stations = {}
    started = perf_counter()
    data = dict(schema='tt22-opponent-learned-target-1', fit=model,
                sources={name: _hash(OUTPUT/name) for name in SOURCES}, evidence=evidence,
                profiles=profiles, held_seed=schedule_seed, held_starts=seeds,
                spread=SPREAD, pace=PACE, rows=histories, verified=0)
    _write(path, data)
    try:
        for name, profile in profiles.items():
            station = _station(profile)
            stations[name] = station, FarTape(station.arena), _audit_motion(station.arena)
        for index, seed in enumerate(seeds):
            order = tuple(stations) if index % 2 == 0 else tuple(reversed(stations))
            for name in order:
                station, tape, samples = stations[name]
                tape.steps.clear()
                samples.clear()
                row = station.point(seed)
                row['rival_actions'] = tape.tape(len(row['actions']))
                row['evaluator_motion'] = samples.copy()
                histories[name].append(row)
                _write(path, data)
                print(name, seed, row['longest'], row['outcome']['reason'], flush=True)
    finally:
        for station, tape, samples in stations.values():
            station.close()
    data['verified'] = sum(_replay(rows) for rows in histories.values())
    data['strike_plane_samples'] = {name: _observed_phase(rows) for name, rows in histories.items()}
    data['wall_seconds'] = perf_counter() - started
    _write(path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--phase', action='store_true')
    args = parser.parse_args()
    output = args.output or OUTPUT / ('tt22_opponent-phase.json.gz' if args.phase else 'tt22_opponent-learned.json.gz')
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = examples([OUTPUT/name for name in SOURCES], phase=args.phase)
    model = fit_phase(evidence) if args.phase else fit(evidence)
    print({key: value for key, value in model.items() if key != 'ranking'}, flush=True)
    held(output, evidence, model, args.count or (4 if args.phase else 6),
         schedule_seed=PHASE_HELD_SEED if args.phase else HELD_SEED)
