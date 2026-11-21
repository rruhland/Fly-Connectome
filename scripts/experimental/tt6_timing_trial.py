"""Play paired table-tennis points under private changes in actuator latency."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TT4Arena
from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, Wiring, _counts, _hash, _metrics, _point, _read, _write, load_sources
from .tt5_hall_trial import prepare, probe_histories, simplify_hall_lighting
from .tt5_player import RallyPlayer
from .tt5_player_body import fit_arm
from .tt6_timing import DelayedWiring, Disturbance, TimingMemory, TimingRallyPlayer
from .tt6_timing_gated import GatedTimingRallyPlayer


CONDITIONS = ('stock', 'frozen', 'cold', 'continual', 'reset_each_point')
SOURCE = ROOT / 'artifacts/tt4/mini-first-person-player.json.gz'
STATIONS = ROOT / 'artifacts/tt5/first-person-confirmation.json.gz'
INITIAL = Disturbance(delays=((0, 2),), noise=((0, .035),))
SHIFTED = Disturbance(delays=((0, 4),), noise=((0, .075),))
SWITCHING = Disturbance(delays=((0, 2), (10, 4), (24, 1)), noise=((0, .035), (10, .075), (24, .050)))
PHASES = {'retention': INITIAL, 'shift': SHIFTED, 'within_point': SWITCHING, 'recurrence': INITIAL}
LOW = Disturbance(noise=((0, .025),))
HIGH = Disturbance(delays=((0, 5),), noise=((0, .060),))
SEPARATED_PHASES = {'retention': LOW, 'shift': HIGH, 'recurrence': Disturbance(noise=((0, .050),))}
PROTOCOLS = {'switching': (INITIAL, PHASES, TimingRallyPlayer),
             'separated': (LOW, SEPARATED_PHASES, GatedTimingRallyPlayer)}


def played_point(wiring, player, seed, phase, disturbance, noise_seed):
    delayed = DelayedWiring(wiring, disturbance, seed=noise_seed)
    row = _point(delayed, player, seed)
    return dict(**row, **_counts(row), phase=phase, disturbance=asdict(disturbance),
                noise_seed=noise_seed, actuator_actions=delayed.applied)


def course(job):
    schedule, geometry, seed, training_points, phase_points, conditions, cache, protocol = job
    initial, phases, timing_player = PROTOCOLS[protocol]
    start = perf_counter()
    native, sources, _ = load_sources(SOURCE)
    arena = TT4Arena(schedule['arena_seed'], view_seed=schedule['view_seed'], view='first_person', **geometry['arena'])
    if geometry['fast_hall']:
        simplify_hall_lighting(arena)
    wiring = Wiring(arena, schedule['arena_seed'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, schedule['index']]))
    count = training_points + len(phases) * phase_points
    serves, noise_seeds = rng.integers(0, 2**31 - 1, (2, count)).tolist()
    plans = [('practice', initial)] * training_points
    plans.extend((phase, disturbance) for phase, disturbance in phases.items() for _ in range(phase_points))
    points = list(zip(serves, plans, noise_seeds, strict=True))
    result = dict(schedule=schedule, evaluator_wiring=wiring.state(), runs={})
    try:
        probe_geometry = {key: value for key, value in geometry.items() if key != 'cache_stereo'}
        observations, probes, _ = probe_histories(wiring, schedule, probe_geometry, cache)
        result['calibration'] = probes
        try:
            if arena.render_mode == 'projected':
                from .tt4_camera import PerspectiveTracker

                def tracker_factory():
                    return PerspectiveTracker(arena.cameras, coordinates='world')

                adapter = fit_arm(observations, tracker_factory)

                def stock(prior):
                    return RallyPlayer(adapter, prior, tracker_factory)
            else:
                stock, adapter = prepare('fallback', arena.cameras, observations)
        except ValueError as error:
            result['calibration_error'] = str(error)
            return result
        if geometry['cache_stereo']:
            from .tt6_render import cache_hall_stereo

            arena.close()
            cache_hall_stereo(arena)
        result['adapter'] = adapter.stats()
        prior = TransferPrior('first_person_mini', native, schedule['player_seed'], mini=sources['first_person_mini'])
        tracker_factory = stock(deepcopy(prior)).tracker_factory
        practice = timing_player(adapter, prior, tracker_factory)
        result['practice'] = [played_point(wiring, practice, serve, *plan, noise_seed)
                              for serve, plan, noise_seed in points[:training_points]]
        result['prior'] = practice.prior.state()
        result['practiced_timing'] = practice.timing_memory.state()
        for condition in conditions:
            timing_prior = result['practiced_timing'] if condition in ('continual', 'frozen') else None
            memory = TimingMemory(timing_prior)
            point_prior = deepcopy(practice.prior)
            if condition == 'stock':
                player = stock(point_prior)
            else:
                player = timing_player(adapter, point_prior, tracker_factory, timing_memory=memory,
                                       timing_learn=condition != 'frozen', timing_carry=condition != 'reset_each_point')
            player.role_rows = deepcopy(practice.role_rows)
            initial_memory = memory.state() if condition != 'stock' else None
            rows = [played_point(wiring, player, serve, *plan, noise_seed)
                    for serve, plan, noise_seed in points[training_points:]]
            result['runs'][condition] = dict(points=rows, timing_start=initial_memory,
                                             timing_end=memory.state() if condition != 'stock' else None)
    finally:
        arena.close()
        result['wall_seconds'] = perf_counter() - start
    return result


def summarize(courses, conditions, phases=PHASES):
    summary = {}
    for condition in conditions:
        rows = [row for c in courses for row in c['runs'].get(condition, {}).get('points', [])]
        summary[condition] = _metrics(rows)
        summary[condition]['phases'] = {phase: phase_summary([row for row in rows if row['phase'] == phase])
                                        for phase in phases}
    return summary


def phase_summary(rows):
    metrics = _metrics(rows)
    early_errors, late_errors = [], []
    for row in rows:
        if 'timing' not in row:
            continue
        delays = row['disturbance']['delays']
        trajectory = np.asarray(row['timing']['trajectory'])
        # Camera frame zero precedes the first action. Each later observation
        # follows the actuator response to the previous control interval.
        ticks = np.maximum(0, np.arange(len(trajectory)) - 1)
        timeline = np.asarray(delays)
        expected = timeline[np.searchsorted(timeline[:, 0], ticks, side='right') - 1, 1]
        residual = np.abs(trajectory - expected)
        early_errors.extend(residual[:12])
        late_errors.extend(residual[12:])
    if early_errors:
        metrics['early_delay_mae_frames'] = float(np.mean(early_errors))
    if late_errors:
        metrics['later_delay_mae_frames'] = float(np.mean(late_errors))
    return metrics


def run(args):
    station_doc = _read(args.stations)
    geometry = dict(station_doc['geometry'])
    geometry['arena'] = dict(geometry['arena'], render_mode=args.renderer, max_steps=args.max_steps)
    geometry['fast_hall'] = args.fast_hall
    geometry['cache_stereo'] = args.cache_stereo
    if args.probes is not None:
        schedules = [dict(s, calibration=s['calibration'][:args.probes]) for s in station_doc['schedules']]
    else:
        schedules = station_doc['schedules']
    schedules = [schedules[index] for index in args.installations]
    jobs = [(s, geometry, args.seed, args.training, args.phase_points, args.conditions, args.cache, args.protocol)
            for s in schedules]
    _, _, inputs = load_sources(SOURCE)
    inputs['station_receipt'] = _hash(args.stations)
    modules = [Path(__file__).with_name(name) for name in ('tt6_timing_trial.py', 'tt6_timing.py',
               'tt5_player.py', 'tt5_player_body.py', 'tt5_vision.py', 'tt4_arena.py', 'tt4_hall.py')]
    if args.cache_stereo:
        modules.append(Path(__file__).with_name('tt6_render.py'))
    if args.protocol == 'separated':
        modules.append(Path(__file__).with_name('tt6_timing_gated.py'))
    phases = PROTOCOLS[args.protocol][1]
    report = dict(schema='tt6-visual-actuator-timing-v1', seed=args.seed, geometry=geometry, protocol=args.protocol,
                  training_points=args.training, points_per_phase=args.phase_points,
                  station_schedules=schedules, conditions=args.conditions,
                  inputs_sha256=inputs, implementation_sha256={p.name: _hash(p) for p in modules}, courses=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(course, jobs):
            report['courses'].append(result)
            report['summary'] = summarize(report['courses'], args.conditions, phases)
            _write(args.output, report)
            counts = {key: {name: value for name, value in metrics.items() if name in ('points_for', 'with_return', 'contacts')}
                      for key, metrics in summarize([result], args.conditions, phases).items()}
            print(json.dumps(dict(installation=result['schedule']['index'], error=result.get('calibration_error'),
                                  learned=result.get('practiced_timing'), counts=counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stations', type=Path, default=STATIONS)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--installations', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--training', type=int, default=3)
    parser.add_argument('--phase-points', type=int, default=3)
    parser.add_argument('--conditions', choices=CONDITIONS, nargs='+', default=list(CONDITIONS[:4]))
    parser.add_argument('--protocol', choices=tuple(PROTOCOLS), default='switching')
    parser.add_argument('--max-steps', type=int, default=140)
    parser.add_argument('--probes', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--renderer', choices=('projected', 'mujoco'), default='mujoco')
    parser.add_argument('--fast-hall', action='store_true')
    parser.add_argument('--cache-stereo', action='store_true', help='accelerate native hall play with a cached static image')
    args = parser.parse_args()
    if min(args.training, args.phase_points, args.max_steps, args.workers) < 1 or (args.probes is not None and args.probes < 2):
        parser.error('expected positive point budgets and at least two calibration probes')
    if args.cache_stereo and args.renderer != 'mujoco':
        parser.error('cached stereo requires native MuJoCo rendering')
    run(args)


if __name__ == '__main__':
    main()
