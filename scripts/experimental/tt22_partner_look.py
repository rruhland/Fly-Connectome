"""Rank physical far shots by an archive model of the actual RGB pupil's reply."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt20_natural import NaturalViewTransport
from .tt22_opponent import EXACT_CHECKS, install_rally_opponent
from .tt22_opponent_learn import JitteredOpening
from .tt22_opponent_trial import OUTPUT, FarTape, SteeredOpening, metrics
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, Station


SOURCES = ('tt22_opponent-native.json.gz', 'tt22_opponent-learned.json.gz', 'tt22_opponent-phase.json.gz')
SCALES = np.array([.095, .080, .62, .92, .40, 48., 34., 42., .20, .15, .15])
FEATURES = ('lateral', 'height', 'forward_speed', 'vertical_speed', 'lateral_speed', 'spin_x', 'spin_y',
            'spin_z', 'near_bounce_depth', 'net_height', 'travel_to_sample_seconds')
WIDTHS = (.65, .90, 1.20, 1.60)
HELD = 718832005
GATE = -.72


def _cross(samples, depth):
    indices = np.flatnonzero((samples[:-1, 2] >= depth) & (samples[1:, 2] < depth))
    if not len(indices):
        return None
    start, end = samples[indices[0]:indices[0] + 2]
    fraction = (depth - start[2]) / (end[2] - start[2])
    return start + fraction * (end - start)


def descriptor(samples, bounce):
    """Measure the incoming corridor before the pupil's likely contact."""
    if len(samples) < 2:
        return None
    gate, net = _cross(samples, GATE), _cross(samples, 0.)
    if gate is None or net is None or gate[3] < TABLE_HEIGHT - .014:
        return None
    return np.r_[gate[1], gate[3] - TABLE_HEIGHT, -gate[5], gate[6], gate[4], gate[7:10],
                 bounce, net[3] - TABLE_HEIGHT, gate[0] - net[0]].tolist()


def sample(arena):
    return [arena.data.time, *arena.ball.tolist(), *arena.velocity.tolist(), *arena.spin.tolist()]


def _examples(arena, row, source, condition):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    samples = [sample(arena)]
    for near, far in zip(row['actions'], row['rival_actions'], strict=True):
        arena._rival_attempt = True
        arena._aim = lambda *args: np.asarray(far)
        arena.step(near, render=False)
        samples.append(sample(arena))
    if (arena.point_result() != row['outcome'] or digest_json(arena.events) != digest_json(row['events'])):
        raise ValueError('an archived native point did not physically reproduce')
    samples = np.asarray(samples)
    evidence = []
    events = row['events']
    for index, bounce in enumerate(events):
        if bounce['kind'] != 'legal_return' or bounce['side'] != -1:
            continue
        contact = next(e for e in reversed(events[:index]) if e['kind'] == 'contact' and e.get('side') == -1)
        following = events[index + 1:]
        near = next((e for e in following if e['kind'] == 'contact' and e.get('side') == 1), None)
        stop = len(samples) if near is None else near['step']
        feature = descriptor(samples[max(0, contact['step'] - 1):stop], bounce['location'][1])
        next_credit = any(e['kind'] == 'legal_return' and e['side'] == 1 for e in following)
        future = sum(e['kind'] == 'legal_return' for e in following)
        evidence.append(dict(source=source, condition=condition, seed=row['seed'], rival_contact=contact['step'],
                             bounce_tick=bounce['step'], feature=feature, next_credit=next_credit,
                             following_returns=future))
    return evidence


def archived(paths):
    evidence, seeds = [], set()
    for path in paths:
        source = _read(path)
        if 'variants' in source:
            arena = SteeredOpening(**CONFIG)
            branches = {'steered': source['variants']['steered']['rows']}
        else:
            arena = JitteredOpening(**CONFIG)
            branches = source['rows']
        try:
            for condition, rows in branches.items():
                for row in rows:
                    seeds.add(row['seed'])
                    evidence.extend(_examples(arena, row, path.name, condition))
        finally:
            arena.close()
    return evidence, seeds


class Survival:
    """Seed-balanced, regularized local estimates of the next reply and rally survival."""

    def __init__(self, evidence, width):
        rows = [row for row in evidence if row['feature'] is not None]
        self.x = np.asarray([row['feature'] for row in rows])
        self.y = np.asarray([[row['next_credit'], min(10, row['following_returns']) / 10.] for row in rows])
        seeds = [row['seed'] for row in rows]
        self.weights = np.asarray([1. / seeds.count(seed) for seed in seeds])
        self.prior = np.average(self.y, weights=self.weights, axis=0)
        self.width = width

    def predict(self, feature):
        if feature is None:
            return 0., 0., 0.
        distance = np.sum(((self.x - feature) / (SCALES * self.width))**2, axis=1)
        weights = np.exp(-.5 * distance) * self.weights
        support = float(np.sum(weights))
        expected = (weights @ self.y + .58 * self.prior) / (support + .58)
        return float(expected[0]), float(expected[1]), support


def fit(evidence):
    seeds = sorted({row['seed'] for row in evidence})
    validation = {}
    for width in WIDTHS:
        errors, reference, survival_error = [], [], []
        for seed in seeds:
            model = Survival([row for row in evidence if row['seed'] != seed], width)
            test = [row for row in evidence if row['seed'] == seed and row['feature'] is not None]
            if not test:
                continue
            predicted = np.asarray([model.predict(row['feature']) for row in test])
            target = np.asarray([[row['next_credit'], min(10, row['following_returns'])/10.] for row in test])
            errors.append(float(np.mean((predicted[:, 0] - target[:, 0])**2)))
            reference.append(float(np.mean((model.prior[0] - target[:, 0])**2)))
            survival_error.append(float(np.mean((predicted[:, 1] - target[:, 1])**2)))
        validation[width] = dict(reply_brier=float(np.mean(errors)), intercept_brier=float(np.mean(reference)),
                                 survival_mse=float(np.mean(survival_error)))
    selected = min(validation, key=lambda width: validation[width]['reply_brier'] + .20 * validation[width]['survival_mse'])
    return Survival(evidence, selected), validation


class Lookahead:
    """Wrap the existing far planner. Traffic still passes its exact strict referee."""

    def __init__(self, rival, model):
        self.rival, self.model = rival, model
        self.original_rank = rival._rank_paths
        self.original_confirm = rival.confirm
        rival._rank_paths, rival.confirm = self.rank, self.confirm

    def _record(self, feature, geometric):
        probability, survival, support = self.model.predict(feature)
        value = .74 * probability + .26 * survival
        score = value - .20 * np.log1p(geometric) - .04 / (.10 + support)
        return dict(predicted_near_return=probability, predicted_following_returns=10. * survival,
                    archive_support=support, contextual_score=float(score), trajectory_feature=feature)

    def rank(self, paths, lane):
        rows = self.original_rank(paths, lane)
        rival = self.rival
        for row in rows:
            first = row['strike_micro']
            section = paths[row['index'], first:]
            center = section[:, rival.position:rival.position+3]
            velocity = section[:, rival.velocity:rival.velocity+3]
            q = section[:, rival.position+3:rival.position+7]
            local = section[:, rival.velocity+3:rival.velocity+6]
            spin = np.cross(q[:, 1:], local)
            spin = local + 2.*(q[:, :1]*spin + np.cross(q[:, 1:], spin))
            estimate = np.column_stack((section[:, 0], center, velocity, spin))
            row.update(self._record(descriptor(estimate, row['landing'][1]), row['cost']))
        return sorted(rows, key=lambda row: row['contextual_score'], reverse=True)

    def confirm(self, rows, tapes, record):
        check = self.rival.check
        step = check.step
        successes = []
        for row in rows[:EXACT_CHECKS]:
            samples = []

            def observed(action, *, render=True):
                samples.append(sample(check))
                return step(action, render=render)

            check.step = observed
            try:
                candidate = self.original_confirm([row], tapes, record)
            finally:
                check.step = step
            if candidate is None:
                continue
            strike = next(e for e in check.events if e['kind'] == 'contact' and e.get('side') == -1)
            landing = next(e for e in check.events if e['kind'] == 'legal_return' and e.get('side') == -1)
            array = np.asarray(samples)
            exact = descriptor(array[array[:, 0] >= strike['time'] - .035], landing['location'][1])
            candidate['dry_prediction'] = row['predicted_near_return']
            candidate.update(self._record(exact, candidate['exact_cost']))
            successes.append(candidate)
        record['confirmed'] = bool(successes)
        record['confirmed_candidates'] = len(successes)
        if not successes:
            return None
        return max(successes, key=lambda item: item['contextual_score'])


def station(model):
    result = Station.__new__(Station)
    law = FlightLaw()
    result.arena = JitteredOpening(**CONFIG)
    result.rival = install_rally_opponent(result.arena)
    if model is not None:
        Lookahead(result.rival, model)
    result.player = RallyPlayer(law)
    result.optics = NaturalViewTransport('fixed', lens_fov=result.arena.lens_intrinsics(), law=law,
                                         flight=False, associate=True)
    return result, FarTape(result.arena)


def verify(rows):
    arena = JitteredOpening(**CONFIG)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=-1, render=False)
            for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(far)
                _, reward, done, cap, _ = arena.step(near, render=False)
                if (done or cap) != (tick + 1 == len(row['actions'])):
                    raise ValueError('the exact motor replay changed duration')
            if (arena.point_result() != row['outcome'] or reward != row['reward']
                    or digest_json(arena.events) != digest_json(row['events'])):
                raise ValueError('the exact motor replay changed events or scoring')
    finally:
        arena.close()
    return len(rows)


def run(path, count):
    sources = [OUTPUT / name for name in SOURCES]
    evidence, source_seeds = archived(sources)
    model, validation = fit(evidence)
    seeds = np.random.default_rng(HELD).integers(2**31, size=count).tolist()
    if set(seeds) & source_seeds or len(seeds) != len(set(seeds)):
        raise ValueError('the frozen assessment overlaps prior training points')
    data = dict(schema='tt22-physical-partner-lookahead-1', feature_schema=FEATURES, gate=GATE, widths=WIDTHS,
                evidence=evidence, sources={source.name: _hash(source) for source in sources},
                source_seeds=sorted(source_seeds), held_generator=HELD, held=seeds, width=model.width,
                validation=validation, rows={name: [] for name in ('fixed', 'contextual')})
    _write(path, data)
    print('frozen model', len(evidence), sum(row['next_credit'] for row in evidence), sum(
          row['feature'] is not None for row in evidence), model.width, validation, flush=True)
    started = perf_counter()
    stations = {'fixed': station(None), 'contextual': station(model)}
    try:
        for index, seed in enumerate(seeds):
            names = tuple(stations)
            for condition in names if index % 2 == 0 else names[::-1]:
                live, recorder = stations[condition]
                recorder.steps.clear()
                row = live.point(seed)
                row['rival_actions'] = recorder.tape(len(row['actions']))
                data['rows'][condition].append(row)
                _write(path, data)
                print(condition, seed, row['longest'], row['outcome'], flush=True)
    finally:
        for live, _ in stations.values():
            live.close()
    data['physical_replays'] = sum(verify(rows) for rows in data['rows'].values())
    data['summary'] = {condition: metrics(rows) for condition, rows in data['rows'].items()}
    data['wall_seconds'] = perf_counter() - started
    _write(path, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/'tt22_partner_look.json.gz')
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    print(run(args.output, args.count)['summary'])
