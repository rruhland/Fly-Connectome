"""Fit an unlabeled spatial action target from visible trajectory reversals."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .tt3_arena import heldout_cases
from .tt3_learning_trial import play_course, summarize
from .tt3_numeric import NumericMemory
from .tt3_pong import cases_from_seeds, evaluate_pong, practice_pong, source_episodes, summarize_pong
from .tt3_visual import RoleBinder, RoleModel


FIT_SEED = 21_074_665
SCORE_SEED = 21_074_666
VERIFY_SEED = 21_074_667
NAMES = ('absent', 'fitted', 'shuffled')


class FrozenTarget(NumericMemory):
    """Keep the deployed coefficients fixed while a new body learns its motors."""

    def observe(self, option, reward):
        pass


def _line(time, path):
    time = np.asarray(time)
    coords = np.asarray(path)
    local = time - time.mean()
    velocity = local @ coords / (local @ local)
    center = coords.mean(axis=0)
    residual = coords - (center + local[:, None] * velocity)
    return center - time.mean() * velocity, velocity, float(np.square(residual).mean())


def _segment(relevant, first, last, origin):
    segment = [(tick, relevant[tick]) for tick in range(max(0, first), min(last, len(relevant)))
               if relevant[tick] is not None]
    if not segment:
        return None
    label, number = Counter(track.id for _, track in segment).most_common(1)[0]
    if number < 4:
        return None
    time, path = zip(*((tick - origin, track.position) for tick, track in segment if track.id == label), strict=True)
    center, speed, residual = _line(time, path)
    return (label, center, speed) if residual < .00005 else None


def visible_examples(samples, roles):
    """Infer candidates from optical track windows and self-excitation alone."""
    binder = RoleBinder(roles)
    acted = []
    relevant = []
    paths = {}
    for tick, sample in enumerate(samples):
        binding = binder.update(sample.frames, sample.preceding_action)
        acted.append(binding.controlled.id if binding.controlled is not None else None)
        mover = binding.relevant
        relevant.append(mover if mover is not None and mover.observed else None)
        for track in binding.tracks:
            if track.observed:
                paths.setdefault(track.id, {})[tick] = np.asarray(track.position)
    actor_ids = Counter(label for label in acted if label is not None)
    if not actor_ids:
        return [], False
    actor = max(actor_ids, key=actor_ids.get)
    path = paths[actor]
    support = np.ptp(list(path.values()), axis=0)
    axis = int(support.argmax())
    if support[axis] < .02 or support[axis] < 3 * support[1 - axis]:
        return [], False
    normal = 1 - axis
    ready = next(iter(path.values()))
    proposals = []
    for tick in range(12, len(relevant) - 12):
        if not any(acted[t] == actor for t in range(tick - 8, tick + 9)):
            continue
        incoming = _segment(relevant, tick - 12, tick - 3, tick)
        outgoing = _segment(relevant, tick + 4, tick + 13, tick)
        if incoming is None or outgoing is None:
            continue
        _, a, va = incoming
        _, b, vb = outgoing
        if abs(va[normal]) < .008 or abs(vb[normal]) < .008 or va[normal] * vb[normal] >= 0:
            continue
        delay = (b[normal] - a[normal]) / (va[normal] - vb[normal])
        if abs(delay) > 7:
            continue
        onset = round(tick + delay)
        before, after = a + delay * va, b + delay * vb
        closest = min(path, key=lambda t: abs(t - onset))
        occupant = path[closest]
        if (abs(closest - onset) > 5 or (a[normal] - 8 * va[normal] - ready[normal]) * va[normal] >= 0
                or abs(before[normal] - ready[normal]) > .060 or abs(before[axis] - occupant[axis]) > .140
                or abs(before[axis] - after[axis]) > .037):
            continue
        proposals.append((abs(delay) + 25 * abs(before[axis] - after[axis]), onset, before, occupant))
    accepted = []
    for _, onset, intersection, actor_at_turn in sorted(proposals, key=lambda row: row[0]):
        if any(abs(onset - prior[0]) < 14 for prior in accepted):
            continue
        accepted.append((onset, intersection, actor_at_turn))
    examples = []
    for onset, intersection, actor_at_turn in accepted:
        for lead in (6, 12, 20):
            now = onset - lead
            observed = _segment(relevant, now - 5, now + 1, now)
            if observed is None:
                continue
            _, mover, rate = observed
            gap = mover[normal] - ready[normal]
            if gap * rate[normal] >= 0 or abs(rate[normal]) < .007:
                continue
            horizon = np.clip(-gap / rate[normal], 0., 90.)
            extrapolated = mover[axis] + rate[axis] * horizon
            reflected = .5 - abs((extrapolated + .5) % 2 - 1)
            feature = [1., float(mover[axis] - ready[axis]), float(reflected - mover[axis])]
            examples.append(dict(tick=onset, lead=lead, feature=feature, target=float(actor_at_turn[axis] - ready[axis]),
                                 optical_crossing_offset=float(intersection[axis] - actor_at_turn[axis])))
    return examples, True


def _fit(x, y):
    return np.linalg.solve(x.T @ x + np.diag([4., .06, .06]), x.T @ y)


def fitting_rows(roles, count):
    rng = np.random.default_rng(FIT_SEED)
    seeds = (rng.choice(140_000_000, count, replace=False) + 601_000_000).tolist()
    all_rows = []
    bound = 0
    for number, (seed, samples) in enumerate(zip(seeds, source_episodes(seeds, max_ticks=1080, batch_size=12), strict=True), 1):
        examples, did_bind = visible_examples(samples, roles)
        all_rows.extend(dict(seed=seed, **row) for row in examples)
        bound += int(did_bind)
        if number % 12 == 0 or number == count:
            print(f'Collected {number} of {count} source episodes, {len(all_rows)} fit rows', flush=True)
    if len(all_rows) < 12:
        raise ValueError(f'only {len(all_rows)} usable visual regression rows')
    ids = list(dict.fromkeys((row['seed'], row['tick']) for row in all_rows))
    x = np.asarray([row['feature'] for row in all_rows])
    y = np.asarray([row['target'] for row in all_rows])
    event_targets = {(row['seed'], row['tick']): row['target'] for row in all_rows}
    remap = dict(zip(ids, rng.permutation([event_targets[key] for key in ids]), strict=True))
    scrambled = np.array([remap[(row['seed'], row['tick'])] for row in all_rows])
    coefficients = np.stack((np.zeros(3), _fit(x, y), _fit(x, scrambled)))
    return coefficients, dict(seeds=seeds, bound=bound, visible_events=len(ids), rows=all_rows,
                              regression_rmse=[float(np.sqrt(np.mean(np.square(x @ c - y)))) for c in coefficients])


def _schedule(seed, count, low):
    rng = np.random.default_rng(seed)
    return cases_from_seeds(rng.choice(90_000_000, count, replace=False) + low)


def evaluate(args):
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    inputs = [directory / f'tt3_{name}.py' for name in
              ('arena', 'event_fit', 'learning', 'learning_trial', 'numeric', 'pong', 'visual')]
    inputs.append(directory.parents[1] / 'src/fly_connectome/pong.py')
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}
    state = json.loads(args.roles.read_text())
    roles = RoleModel(**state.get('model', state.get('role_model', state)))
    coefficients, fit = fitting_rows(roles, args.fit)
    print('Fitted coefficients', coefficients.tolist(), flush=True)
    memory = NumericMemory(coefficients)
    score_starts = _schedule(SCORE_SEED, args.score, 1_140_000_000)
    options = np.tile(np.arange(len(coefficients)), len(score_starts)).tolist()
    scored = practice_pong([case for case in score_starts for _ in coefficients], roles, memory,
                           options=options, max_ticks=1080, batch_size=18)
    chosen = memory.choose(np.random.default_rng(SCORE_SEED))
    print('Public score state', memory.state(), 'selected', NAMES[chosen], flush=True)
    test_starts = _schedule(VERIFY_SEED, args.verify, 1_250_000_000)
    pong = {}
    table = {}
    for index, name in enumerate(NAMES):
        fixed = FrozenTarget([coefficients[index]])
        rows = evaluate_pong(test_starts, roles, fixed, max_ticks=1080, batch_size=12)
        pong[name] = dict(totals=summarize_pong(rows), points=rows)
        print(name, 'Pong', pong[name]['totals'], flush=True)
        if args.table:
            courses = []
            for course_index in range(2):
                cases = heldout_cases(f'{VERIFY_SEED}/event/{course_index}', args.table, variant='standard')
                course = play_course(cases, roles, fixed, seed=course_index, learn_roles=False)
                # The shared runner labels its legacy option slot. This run uses a single numeric target.
                for row in course['points']:
                    row['option'] = 0
                courses.append(course)
            table[name] = dict(totals=summarize(courses), courses=courses)
            print(name, 'table', table[name]['totals'], flush=True)
    if hashes != {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}:
        raise ValueError('shared experiment code changed during this run')
    with gzip.open(args.output, 'xt') as output:
        json.dump(dict(role_model=asdict(roles), fit_seed=FIT_SEED, score_seed=SCORE_SEED, verify_seed=VERIFY_SEED,
                       public_score_selected=NAMES[chosen], fit=fit, memory=memory.state(), inputs_sha256=hashes,
                       score_points=scored, pong=pong, table=table),
                  output, allow_nan=False, separators=(',', ':'))
        output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fit', type=int, default=60)
    parser.add_argument('--score', type=int, default=16)
    parser.add_argument('--verify', type=int, default=16)
    parser.add_argument('--table', type=int, default=8, help='points per receiving course, or zero')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
