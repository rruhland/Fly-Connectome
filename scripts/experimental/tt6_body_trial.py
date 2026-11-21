"""Practice on alternate physical bodies and test what transfers to a new one."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _seeds, _write, load_sources
from .tt5_player import RallyPlayer, fit_arm
from .tt5_vision import hall_tracker_factory, motion_tracker_factory
from .tt6_body_arena import BODY_AXES, TT6Arena, TT6Wiring
from .tt6_body_control import CameraBodyPlayer, fit_camera_body
from .tt6_body_probe import probe_histories
from .tt6_body_vision import body_tracker_factory
from .tt6_render import cache_hall_stereo


MINI = ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz'
ALTERNATES = ('gantry', 'offset')


def _setup(body, schedule, geometry, cache):
    arena = TT6Arena(schedule['arena_seed'], view_seed=schedule['view_seed'], body=body, **geometry['arena'])
    try:
        if geometry['cached_hall']:
            cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, schedule['arena_seed'])
        histories, receipt = probe_histories(wiring, schedule, {**geometry, 'body': body}, cache)
    except Exception:
        arena.close()
        raise
    return arena, wiring, histories, receipt


def _play(wiring, player, seeds):
    initial = player.prior.state()
    rows = []
    for seed in seeds:
        point = _point(wiring, player, seed)
        rows.append(dict(**point, **_counts(point)))
    return dict(initial_prior=initial, final_prior=player.prior.state(), points=rows, summary=_metrics(rows))


def _stock_fit(arena, histories):
    factory = hall_tracker_factory(arena.cameras, histories)
    try:
        return fit_arm(histories, factory), factory
    except ValueError:
        factory = motion_tracker_factory(arena.cameras, histories)
        return fit_arm(histories, factory), factory


def practice(body, schedule, geometry, cache, prior):
    arena, wiring, histories, probes = _setup(body, schedule, geometry, cache)
    try:
        if body == 'stock':
            fit, factory = _stock_fit(arena, histories)
            player = RallyPlayer(fit, prior, factory)
        else:
            factory = body_tracker_factory(arena.cameras, histories)
            fit = fit_camera_body(histories, factory)
            player = CameraBodyPlayer(fit, prior, factory)
        report = _play(wiring, player, schedule['tests'])
        return dict(body=body, schedule=schedule, calibration=probes, adapter=fit.stats(),
                    evaluator_wiring=wiring.state(), **report), fit.lags
    finally:
        arena.close()


def qualify(body, schedule, geometry, cache, *, shared, mini, fresh, donor_lags, donors):
    arena, wiring, histories, probes = _setup(body, schedule, geometry, cache)
    try:
        factory = body_tracker_factory(arena.cameras, histories)
        shared_fit = fit_camera_body(histories, factory, lag_prior=donor_lags)
        limited_fit = fit_camera_body(histories, factory)
        fits = {'shared': shared_fit, 'mini-only': limited_fit, 'fresh': limited_fit,
                'reversed-map': shared_fit.reversed_channels()}
        priors = {'shared': shared, 'mini-only': mini, 'fresh': fresh, 'reversed-map': shared}
        sweep_public_axis = None
        if body == 'offset':
            physical = BODY_AXES[body][0].index('sweep')
            sweep_public_axis = int(np.flatnonzero(wiring.order == physical)[0])
            fits['without-sweep'] = shared_fit.without_channel(sweep_public_axis)
            priors['without-sweep'] = shared
        runs = {name: _play(wiring, CameraBodyPlayer(fit, deepcopy(priors[name]), factory), schedule['tests'])
                for name, fit in fits.items()}
        if sweep_public_axis is not None:
            for name in ('shared', 'without-sweep'):
                applied = np.concatenate([point['actions'] for point in runs[name]['points']])[:, sweep_public_axis]
                runs[name]['sweep_action'] = dict(mean_abs=float(np.mean(np.abs(applied))),
                                                  frames_above_005=int(np.count_nonzero(np.abs(applied) > .05)))
        return dict(body=body, schedule=schedule, calibration=probes, donor_bodies=donors,
                    donor_lag_prior=donor_lags, adapter={name: fit.stats() for name, fit in fits.items()},
                    evaluator_wiring=wiring.state(), evaluator_sweep_public_axis=sweep_public_axis, runs=runs)
    finally:
        arena.close()


def _summary(courses):
    rows = defaultdict(list)
    for course in courses:
        for condition, result in course.get('runs', {}).items():
            rows[course['body'], condition].extend(result['points'])
    summary = {}
    for (body, condition), points in rows.items():
        summary.setdefault(body, {})[condition] = _metrics(points)
    return summary


def run(args):
    geometry = dict(probe_horizon=args.probe_steps, cached_hall=not args.native,
                    arena=dict(venue='hall', width=192, height=144, baseline=.28, max_steps=args.max_steps))
    native, states, hashes = load_sources(MINI)

    def new_prior(name, seed):
        return TransferPrior(name, native, seed, mini=states['first_person_mini'])

    def new_schedule(seed, i, points):
        schedule = _seeds(seed, i, max(8, args.probes), points)
        schedule['calibration'] = schedule['calibration'][:args.probes]
        return schedule

    files = sorted((ROOT / 'scripts' / 'experimental').glob('tt[456]*'))
    report = dict(schema='tt6-physical-body-transfer-v1', seed=args.seed, practice_seed=args.practice_seed,
                  geometry=geometry, inputs_sha256=hashes, practice=[], courses=[],
                  implementation_sha256={path.name: _hash(path) for path in files if path.is_file()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    stock_schedule = new_schedule(args.practice_seed, 0, args.practice_points)
    stock = new_prior('first_person_mini', stock_schedule['player_seed'])
    stock_report, stock_lags = practice('stock', stock_schedule, geometry, args.cache, stock)
    report['practice'].append(stock_report)
    print('stock practice', stock_report['summary'], flush=True)
    donors = {}
    for index, body in enumerate(ALTERNATES, start=1):
        donor = deepcopy(stock)
        schedule = new_schedule(args.practice_seed, index, args.practice_points)
        donor_report, lags = practice(body, schedule, geometry, args.cache, donor)
        report['practice'].append(donor_report)
        donors[body] = donor, float(np.median(np.r_[stock_lags, lags]))
        print(body, 'practice', donor_report['summary'], flush=True)
        _write(args.output, report)

    for installation in range(args.installations):
        for body_index, body in enumerate(ALTERNATES):
            schedule = new_schedule(args.seed, len(ALTERNATES) * installation + body_index, args.points)
            alternative = next(candidate for candidate in ALTERNATES if candidate != body)
            prior, lags = donors[alternative]
            try:
                result = qualify(body, schedule, geometry, args.cache,
                                 shared=prior, mini=new_prior('first_person_mini', schedule['player_seed']),
                                 fresh=new_prior('fresh', schedule['player_seed']),
                                 donor_lags=lags, donors=['stock', alternative])
            except ValueError as error:
                result = dict(body=body, schedule=schedule, error=str(error), runs={})
            report['courses'].append(result)
            report['summary'] = _summary(report['courses'])
            report['wall_seconds'] = perf_counter() - started
            _write(args.output, report)
            print(body, 'installation', installation, result.get('error', ''),
                  {name: (row['summary']['with_return'], row['summary']['player_returns'], row['summary']['sustained'])
                   for name, row in result['runs'].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--practice-seed', type=int, required=True)
    parser.add_argument('--installations', type=int, default=2)
    parser.add_argument('--points', type=int, default=8)
    parser.add_argument('--practice-points', type=int, default=8)
    parser.add_argument('--probes', type=int, default=6)
    parser.add_argument('--probe-steps', type=int, default=34)
    parser.add_argument('--max-steps', type=int, default=360)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--native', action='store_true', help='render live using MuJoCo’s standard shaded pass')
    args = parser.parse_args()
    if args.seed == args.practice_seed:
        parser.error('use disjoint practice and evaluation seeds')
    if min(args.installations, args.points, args.practice_points, args.probes, args.probe_steps) < 1:
        parser.error('expected positive tape and point counts')
    run(args)


if __name__ == '__main__':
    main()
