"""Compare equal mini-table practice before hidden native-hall racket changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import median
from unittest.mock import patch

import numpy as np

from .tt3_arena import HZ, IMAGE_WIDTH, PIXELS_PER_METER, heldout_cases, PADDLE_RESTITUTION, _strike
from .tt3_learning import play_point
from .tt4_arena import TT4Arena
from .tt4_mini_pov import load_camera_player, make_arena
from .tt4_transfer_trial import ROOT, Wiring, _counts, _hash, _point, _read, _write
from .tt5_checkpoint import load
from .tt6_racket import (
    RACKET_DAMPING,
    OpticalImpact,
    RacketAdaptation,
    RacketPlayer,
    optical_impact,
    practiced_gain,
    set_racket_response,
)


MINI = ROOT / 'artifacts/tt4/mini-first-person-player.json.gz'
STATION = ROOT / 'artifacts/tt5/first-hall-player.npz'
CALIBRATION = ROOT / 'artifacts/tt5/short-calibration.json.gz'
CONDITIONS = ('varied_mini', 'ordinary_mini', 'naive', 'frozen')
MATCH = (('stock', 3), ('medium', 5), ('stock', 3), ('medium', 5), ('soft', 5))


def _mini_point(source, case, optical_seed, response):
    arena = make_arena(case, optical_seed, 'first_person', 192, 144, .28)
    player = load_camera_player(source, arena)
    observe = player.binder.update
    ball = []

    def track_ball(*args, **kwargs):
        binding = observe(*args, **kwargs)
        relevant = binding.relevant
        if relevant is not None and relevant.observed:
            depth, lateral, neg_height = relevant.position
            ball.append(np.array((lateral, depth, -neg_height)) * IMAGE_WIDTH / PIXELS_PER_METER)
        else:
            ball.append(None)
        return binding

    def strike(old_ball, point, incoming, old_center, center, normal, dt):
        result = _strike(old_ball, point, incoming, old_center, center, normal, dt)
        if result is None or normal[1] < 0.:
            return result
        contact, original = result
        scale = (1. + response) / (1. + PADDLE_RESTITUTION)
        outgoing = tuple(v + scale * (q - v) for v, q in zip(incoming, original, strict=True))
        return contact, outgoing

    player.binder.update = track_ball
    with patch('scripts.experimental.tt3_arena._strike', strike):
        row = play_point(arena, player)
    impact = optical_impact(ball, 1. / HZ)
    return dict(seed=int(case.seed), view_seed=int(optical_seed), response=response, impact=asdict(impact) if impact else None,
                steps=row['frames'], contacts=row['contacts'], returns=row['legal_returns'], score=row['public_reward'],
                action_sha256=row['action_sha256'])


def _mini_pair(item):
    index, case, other_view, mini_path = item
    source = _read(mini_path)
    response = (.52, .67, 1.25)[index % 3]
    return dict(reference=_mini_point(source, case, case.view_seed, PADDLE_RESTITUTION),
                ordinary=_mini_point(source, case, other_view, PADDLE_RESTITUTION),
                varied=_mini_point(source, case, other_view, response))


def mini_practice(seed, pairs, workers=1):
    cases = heldout_cases(seed, pairs * 2)
    items = [(i, case, cases[pairs + i].view_seed, MINI) for i, case in enumerate(cases[:pairs])]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_mini_pair, items))

    def samples(group):
        def unpack(row):
            return OpticalImpact(**row['impact']) if row['impact'] is not None else None
        # Exclude a pair from both estimates when any of its traces is occluded.
        comparable = [row for row in rows if all(item['impact'] is not None for item in row.values())]
        return [(unpack(row['reference']), unpack(row[group])) for row in comparable]

    return dict(seed=seed, checkpoint_sha256=_hash(MINI), points_per_group=2 * pairs,
                ordinary=practiced_gain(samples('ordinary')), varied=practiced_gain(samples('varied')), rows=rows)


def _station(max_steps):
    doc = _read(CALIBRATION)
    schedule = doc['courses']['first_person'][0]['schedule']
    arena = TT4Arena(schedule['arena_seed'], view_seed=schedule['view_seed'], view='first_person',
                     **{**doc['geometry']['arena'], 'max_steps': max_steps})
    return arena, Wiring(arena, schedule['arena_seed'])


def physical_audit():
    """Retrace fixed commands to verify that changed ball motion comes from collision physics."""
    receipt = _read(CALIBRATION)
    course = receipt['courses']['first_person'][0]
    rows = course['runs']['rally:legacy_mini']['points']
    arena, wiring = _station(receipt['geometry']['arena']['max_steps'])
    results = {}
    try:
        for response in RACKET_DAMPING:
            set_racket_response(arena, response)
            samples = []
            for row in rows:
                wiring.reset(seed=row['seed'], render=False)
                for action in row['actions']:
                    _, _, done, truncated, _ = wiring.step(action, render=False)
                    if done or truncated:
                        break
                contact = next(e for e in arena.events if e['kind'] == 'contact' and e['side'] == 1)
                departure = next(e for e in arena.events if e['kind'] == 'separation' and e['collider'] == 'learner_racket')
                samples.append(dict(seed=row['seed'], return_found=bool(arena.legal_returns), reason=arena.reason,
                                    action_sha256=row['action_sha256'], incoming=contact['incoming'],
                                    departure=departure['velocity']))
            results[response] = dict(points=len(samples), points_with_return=sum(r['return_found'] for r in samples),
                                     median_exit_speed=median(r['departure'][1] for r in samples), rows=samples)
    finally:
        arena.close()
    return results


def _run_match(job):
    condition, gain, schedule, max_steps, renderer = job
    arena, wiring = _station(max_steps)
    try:
        if renderer == 'cached':
            from .tt6_render import cache_hall_stereo
            cache_hall_stereo(arena)
        station = load(STATION)
        learner = RacketPlayer(station, RacketAdaptation(gain=gain, enabled=condition != 'frozen',
                                                         default_speed=station.stroke_speed))
        rows = []
        for phase, offset, response, seed in schedule:
            set_racket_response(arena, response)
            row = _point(wiring, learner, seed)
            rows.append(dict(phase=phase, offset=offset, racket=response, **row, **_counts(row)))
        return condition, rows
    finally:
        arena.close()


def summarize(runs):
    summary = {}
    for condition, rows in runs.items():
        phases = []
        for phase in dict.fromkeys(row['phase'] for row in rows):
            part = [row for row in rows if row['phase'] == phase]
            returned = [bool(row['player_returns']) for row in part]
            phases.append(dict(phase=phase, response=part[0]['racket'], points=len(part),
                               returned=returned, points_with_return=sum(returned), early_returns=sum(returned[:3]),
                               first_return_point=next((i + 1 for i, yes in enumerate(returned) if yes), None),
                               optics=sum(row['racket_adaptation']['visual_impact'] is not None for row in part),
                               score=[sum(row[key] for row in part) for key in ('points_for', 'points_against')]))
        summary[condition] = phases
    return summary


def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.practice is None:
        practice = mini_practice(args.practice_seed, args.practice_pairs, args.workers)
    else:
        practice = _read(args.practice)['mini_practice']
    report = dict(schema='tt6-native-racket-response-v1', mini_practice=practice, physical_audit=physical_audit())
    _write(args.output, report)
    print({key: practice[key] for key in ('ordinary', 'varied')}, flush=True)
    if args.practice_only:
        return
    rng = np.random.default_rng(args.seed)
    schedule = [[phase, offset, response, int(rng.integers(0, 2**31 - 1))]
                for phase, (response, points) in enumerate(MATCH) for offset in range(points)]
    gains = dict(varied_mini=practice['varied']['gain'], ordinary_mini=practice['ordinary']['gain'],
                 naive=.20, frozen=0.)
    report.update(seed=args.seed, schedule=schedule, scores={}, renderer=args.renderer,
                  max_steps=args.max_steps, gains=gains, checkpoint_sha256=_hash(STATION),
                  calibration_sha256=_hash(CALIBRATION),
                  implementation_sha256={path.name: _hash(path) for path in
                                         (Path(__file__), Path(__file__).with_name('tt6_racket.py'))})
    jobs = [(condition, gains[condition], schedule, args.max_steps, args.renderer) for condition in args.conditions]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for condition, rows in executor.map(_run_match, jobs):
            report['scores'][condition] = rows
            report['summary'] = summarize(report['scores'])
            _write(args.output, report)
            print(condition, report['summary'][condition], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--practice', type=Path)
    parser.add_argument('--practice-only', action='store_true')
    parser.add_argument('--practice-seed', type=int, default=487732872)
    parser.add_argument('--practice-pairs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026112203)
    parser.add_argument('--max-steps', type=int, default=140)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--renderer', choices=('stock', 'cached'), default='stock')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    main(parser.parse_args())
