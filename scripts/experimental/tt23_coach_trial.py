"""Play a frozen optical receiver against geometric and learned physical partners."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt21_coach_oracle import PHASE
from .tt22_after_service import verify_rows
from .tt22_match import CHECKPOINT, HANDOVER, summarize
from .tt22_opponent import EXACT_CHECKS
from .tt22_ordinary_game import FrozenMemory, StandardMatch
from .tt23_coach import FEATURES, OUTPUT, PRACTICE, design, episodes


RECEIPT = OUTPUT/'coach-held.json.gz'
SCHEDULE = 2099526147
COUNT = 8
RANK_WEIGHT = 1.20
CONDITIONS = ('stock', 'learned')


class ReplyRanking:
    """Use the stock menu, exact checks and geometry guard with learned continuation cost."""

    def __init__(self, rival, practice):
        self.rival = rival
        self.model = practice['model']['features']
        self.weights = np.asarray(practice['model']['coefficients'])
        self.reference = design(practice['rows'], self.model)[:, 1:]

    def probability(self, features):
        feature = np.clip(features[:FEATURES[self.model]], -5., 5.)
        distance = np.linalg.norm(self.reference-feature, axis=1).min()
        if distance > 2.5:
            return None
        return float(expit(self.weights[0]+feature@self.weights[1:]))

    def confirm(self, ranked, tapes, record):
        rival = self.rival
        arena, check = rival.arena, rival.check
        fields = {key: getattr(arena, key) for key in PHASE if hasattr(arena, key)}
        target = np.asarray(rival.target())
        penalty = np.array([5.8, 4.9, 4., .42])
        best = None
        near_idle = np.zeros(arena.action_size)
        for candidate in ranked[:EXACT_CHECKS]:
            record['exact_trials'] += 1
            mujoco.mj_copyData(check.data, arena.model, arena.data)
            for key, value in fields.items():
                setattr(check, key, deepcopy(value))
            check.events = []
            check._rival_attempt = True
            tape = tapes[candidate['index']]
            check._aim = lambda *args: tape[check.steps-fields['steps']]
            check._choose_rival_response = lambda: None
            landing, landing_tick, corridor, geometric = None, None, None, 9.
            for _ in tape:
                _, _, done, capped, _ = check.step(near_idle, render=False)
                if check.rival_returns > arena.rival_returns:
                    if landing is None:
                        landing, landing_tick = check.ball.tolist(), check.steps
                    x, y, z = check.ball-[0., 0., TABLE_HEIGHT]
                    vy = -check.velocity[1]
                    if -.1+target[1] < y < -.72 and .055 < z < .46 and abs(x) < .40 and vy > .45:
                        sample = np.array((x, y, z, vy))
                        error = float(penalty@np.abs(sample-target))
                        if error < geometric:
                            corridor, geometric = sample.tolist(), error
                    if check._last_hitter == 1 or y <= -1.10:
                        break
                if done or capped:
                    break
            if landing is None:
                continue
            prediction = next(episodes(dict(seed=arena.seed, events=check.events), target), None)
            probability = self.probability(prediction['features']) if prediction is not None else None
            score = geometric if probability is None else geometric+RANK_WEIGHT*(.5-probability)
            if best is None or score < best['learned_cost']:
                best = dict(candidate, exact_landing=landing, exact_tick=landing_tick,
                            exact_corridor=corridor, exact_cost=geometric, learned_cost=score,
                            predicted_response=probability)
            if geometric < .34:
                break
        record['confirmed'] = best is not None
        return best


def summarize_points(rows):
    result = summarize(rows)
    result['twenty_or_more'] = sum(row['longest'] >= 20 for row in rows)
    result['point_wins'] = sum(row['outcome']['points_for'] for row in rows)
    return result


def verify_service(stock, learned):
    reference, treatment = stock['actions'][:HANDOVER], learned['actions'][:HANDOVER]
    pairs = (stock, learned)
    planning = [row['rival'][0]['tick'] if row['rival'] else None for row in pairs]
    rival_start = min(tick if tick is not None else len(row['actions']) for tick, row in zip(planning, pairs, strict=True))
    if reference != treatment or stock['rival_actions'][:rival_start] != learned['rival_actions'][:rival_start]:
        raise ValueError('coach ranking changed the opening before far-arm planning')
    if any(stock[key] != learned[key] for key in ('toss_x', 'service_rgb')):
        raise ValueError('the optical service setup changed across coaches')

    def events(row):
        stop = next((i+1 for i, event in enumerate(row['events']) if event['kind'] == 'legal_serve'), len(row['events']))
        return row['events'][:stop]

    if events(stock) != events(learned):
        raise ValueError('coach ranking changed the common physical serve')
    service = next((e for e in stock['events'] if e['kind'] == 'legal_serve'), None)
    far_contacts = [next((e for e in row['events'] if e['kind'] == 'contact' and e['side'] == -1), None)
                    for row in pairs]
    if service is not None and any(contact is not None and contact['time'] <= service['time'] for contact in far_contacts):
        raise ValueError('far contact preceded the certified legal serve')

    def first_difference(key):
        return next((tick for tick, (a, b) in enumerate(zip(stock[key], learned[key])) if a != b), None)

    return dict(seed=stock['seed'], toss_rgb=stock['service_rgb'], near_command_prefix=len(reference),
                first_far_plan=planning, legal_service=service,
                first_far_contact=[e['step'] if e is not None else None for e in far_contacts],
                first_near_command_difference=first_difference('actions'),
                first_far_command_difference=first_difference('rival_actions'))


def experiment(output, practice_path, count):
    practice = _read(practice_path)
    if not practice['live_gate']:
        raise ValueError('the fitted continuation score did not pass grouped validation')
    if count < 1:
        raise ValueError('the held schedule requires at least one point')
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=count).tolist()
    fitted = {row['seed'] for row in practice['rows']}
    calibration = _read(CHECKPOINT)
    fitted.update(row['seed'] for row in calibration['practice'])
    if len(seeds) != len(set(seeds)) or fitted.intersection(seeds):
        raise ValueError('held seeds must be fresh and unique')
    profile = practice['profile']
    if len(profile) != 4:
        raise ValueError('the stock geometric corridor uses four coordinates')
    metadata = dict(schema='tt23-coach-held-1', seeds=seeds, schedule=SCHEDULE, server=1,
                    practice_sha256=_hash(practice_path), near_sha256=_hash(CHECKPOINT), profile=profile,
                    rank_weight=RANK_WEIGHT, exact_trials=EXACT_CHECKS)
    report = _read(output) if output.exists() else dict(metadata=metadata, rows={name: [] for name in CONDITIONS})
    if report['metadata'] != metadata:
        raise ValueError('held receipt uses a different fit or schedule')
    for rows in report['rows'].values():
        if [row['seed'] for row in rows] != seeds[:len(rows)]:
            raise ValueError('held receipt differs from its schedule')
    if report.get('verified') == len(CONDITIONS)*count and len(report.get('service_pairs', [])) == count:
        return report
    station = StandardMatch(profile)
    stock = station.rival.confirm
    learned = ReplyRanking(station.rival, practice).confirm
    memory = FrozenMemory(calibration['evidence'])
    started = perf_counter()
    try:
        for index, seed in enumerate(seeds):
            order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
            for name in order:
                rows = report['rows'][name]
                if len(rows) > index:
                    continue
                station.rival.confirm = stock if name == 'stock' else learned
                row = station.play(seed, 1, memory)
                rows.append(row)
                _write(output, report)
                print(index+1, name, row['longest'], row['outcome']['legal_returns'],
                      round(perf_counter()-started, 1), flush=True)
    finally:
        station.close()
    service_pairs = [verify_service(stock_row, learned_row) for stock_row, learned_row in
                     zip(report['rows']['stock'], report['rows']['learned'], strict=True)]
    flat = [row for rows in report['rows'].values() for row in rows]
    verify_rows(flat)
    report.update(verified=len(flat), service_pairs=service_pairs,
                  summary={name: summarize_points(rows) for name, rows in report['rows'].items()})
    _write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--practice', type=Path, default=PRACTICE)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--count', type=int, default=COUNT)
    args = parser.parse_args()
    print(experiment(args.output, args.practice, args.count)['summary'], flush=True)


if __name__ == '__main__':
    main()
