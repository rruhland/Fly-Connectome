"""Compare player variants on identical shaded RGB, point seeds and charged calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import tempfile

import mujoco
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TT4Arena
from .tt4_native_temporal import modal_image, temporal_tracker
from .tt4_native_temporal_trial import _fit, _probe
from .tt4_transfer import TransferPlayer, TransferPrior
from .tt4_transfer_trial import ROOT, Wiring, _counts, _hash, _metrics, _point, _read, _seeds, _write, load_sources


SOURCES = ('legacy_mini', 'first_person_mini', 'pong_only', 'fresh')
PIPELINES = ('temporal', 'tracking', 'rally', 'fallback', 'unmapped', 'impact', 'residual', 'adaptive')


def tracker_factory(pipeline, rig, observations):
    if pipeline == 'temporal':
        reference = modal_image([frames for history in observations for _, frames in history[::2]])
        return lambda: temporal_tracker(rig, reference)
    from .tt5_vision import hall_tracker_factory
    return hall_tracker_factory(rig, observations)


def prepare(pipeline, rig, observations):
    factory = tracker_factory(pipeline, rig, observations)
    if pipeline in ('rally', 'fallback', 'unmapped', 'impact', 'residual'):
        from .tt5_player import fit_arm, RallyPlayer
        try:
            adapter = fit_arm(observations, factory)
        except ValueError:
            if pipeline != 'fallback':
                raise
            from .tt5_vision import motion_tracker_factory
            factory = motion_tracker_factory(rig, observations)
            adapter = fit_arm(observations, factory)
        if pipeline == 'unmapped':
            from dataclasses import replace
            # Counterfactual: preserve estimated geometry while erasing channel order and direction.
            adapter = replace(adapter, scripts=np.arange(4), gains=np.abs(adapter.gains))
        if pipeline == 'impact':
            measured = _read(ROOT/'artifacts/tt5/mini-impact.json')
            return lambda prior: RallyPlayer(adapter, prior, factory, impact_restitution=measured['external']['restitution']), adapter
        if pipeline == 'residual':
            from .tt5_adjust import ContactAdjuster
            return lambda prior: RallyPlayer(adapter, prior, factory, adjuster=ContactAdjuster(adapter.ready)), adapter
        return lambda prior: RallyPlayer(adapter, prior, factory), adapter
    adapter, error = _fit(observations, factory)
    if adapter is None:
        raise ValueError(error)
    return lambda prior: TransferPlayer(adapter, prior, factory), adapter



def probe_histories(wiring, schedule, geometry, cache):
    horizon = geometry['probe_horizon']
    if cache is None:
        return (*_probe(wiring, schedule, horizon), False)
    # A cache entry contains public actions and the RGB stream seen by those commands.
    folder = ROOT/'scripts'/'experimental'
    files = ['tt4_arena.py', 'tt4_arena.xml', 'tt4_hall.py', 'tt4_camera.py', 'tt4_transfer.py',
             'tt4_transfer_trial.py', 'tt4_native_temporal_trial.py']
    manifest = dict(geometry=geometry, schedule={k:v for k,v in schedule.items() if k != 'tests'},
                    mujoco=mujoco.__version__, code=[_hash(folder/f) for f in files], version=1)
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = cache / (key + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            offsets = archive['offsets']
            frames, actions = archive['frames'], archive['actions']
            observations = [list(zip(actions[a:b], frames[a:b], strict=True))
                            for a,b in zip(offsets[:-1], offsets[1:], strict=True)]
            calibration = json.loads(archive['calibration'].item())
        return observations, calibration, True
    observations, calibration = _probe(wiring, schedule, horizon)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        np.savez_compressed(temp_path, frames=np.stack([f for hist in observations for _,f in hist]),
                            actions=np.stack([a for hist in observations for a,_ in hist]),
                            offsets=np.cumsum([0]+[len(hist) for hist in observations]),
                            calibration=json.dumps(calibration))
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return observations, calibration, False


def simplify_hall_lighting(arena):
    """Reduce visual-only tessellation for development; use standard shading for qualification."""
    arena.model.vis.global_.offwidth = arena.width
    arena.model.vis.global_.offheight = arena.height
    arena.model.vis.quality.numquads = 1
    arena.model.geom('floor').size[2] = .5


def course(job):
    schedule, geometry, conditions, sources_file, cache = job
    native, states, _ = load_sources(sources_file)
    start = perf_counter()
    arena = TT4Arena(schedule['arena_seed'], view_seed=schedule['view_seed'],
                     view='first_person', **geometry['arena'])
    if geometry.get('fast_hall'):
        simplify_hall_lighting(arena)
    wiring = Wiring(arena, schedule['arena_seed'])
    try:
        t = perf_counter()
        if conditions[0].startswith('adaptive:'):
            from .tt5_station import commission
            adapter, factory, probes, attempts, error = commission(wiring, arena.cameras, schedule, geometry['probe_horizon'])
            cache_hit = False
        else:
            observations, probes, cache_hit = probe_histories(wiring, schedule, geometry, cache)
            attempts = []
        result = dict(schedule=schedule, view='first_person', calibration=probes, evaluator_wiring=wiring.state(),
                      cache_hit=cache_hit, attempts=attempts, probe_seconds=perf_counter()-t, calibration_errors={}, runs={}, adapters={})
        for pipeline in dict.fromkeys(k.split(':')[0] for k in conditions):
            t = perf_counter()
            try:
                if pipeline == 'adaptive':
                    from .tt5_player import RallyPlayer
                    if error is not None:
                        raise ValueError(error)
                    def new_player(prior):
                        return RallyPlayer(adapter,prior,factory)
                else:
                    new_player, adapter = prepare(pipeline, arena.cameras, observations)
            except ValueError as e:
                result['calibration_errors'][pipeline] = str(e)
                continue
            result['adapters'][pipeline] = dict(stats=adapter.stats(), fit_seconds=perf_counter()-t)
            for name in conditions:
                if not name.startswith(pipeline+':'):
                    continue
                source = name.split(':')[1]
                prior = TransferPrior(source, native, schedule['player_seed'], mini=states.get(source))
                player = new_player(prior)
                rows = []
                before = prior.state()
                t = perf_counter()
                wins = losses = 0
                target = geometry.get('race_to', 0)
                for seed in schedule['tests']:
                    row = _point(wiring, player, seed)
                    rows.append(dict(**row, **_counts(row)))
                    wins += rows[-1]['points_for']
                    losses += rows[-1]['points_against']
                    if target and max(wins, losses) >= target and abs(wins-losses) >= 2:
                        break
                result['runs'][name] = dict(points=rows, initial_prior=before, final_prior=prior.state(),
                                             play_seconds=perf_counter()-t,
                                             score=[wins, losses])
    finally:
        arena.close()
    result['wall_seconds'] = perf_counter()-start
    return result


def summarize(courses, conditions, race_to=0):
    totals = {}
    for condition in conditions:
        scheduled = sum(len(c['schedule']['tests']) for c in courses)
        rows = [p for c in courses for p in c['runs'].get(condition, {}).get('points', [])]
        totals[condition] = dict(_metrics(rows), scheduled_max_points=scheduled)
        if race_to:
            totals[condition]['match_scores'] = [c['runs'][condition]['score'] for c in courses if condition in c['runs']]
        else:
            totals[condition]['missing_points'] = scheduled-len(rows)
    return totals


def run(args):
    conditions = args.conditions
    if len(set(conditions)) != len(conditions):
        raise ValueError('duplicate condition')
    if any(k.startswith('adaptive:') for k in conditions) and not all(k.startswith('adaptive:') for k in conditions):
        raise ValueError('run adaptive-budget and fixed-budget trials separately')
    for name in conditions:
        pipeline, source = name.split(':')
        if pipeline == 'impact' and source != 'legacy_mini':
            raise ValueError('impact comparison requires the original mini source')
        if pipeline not in PIPELINES or source not in SOURCES:
            raise ValueError(f'unknown condition {name}')
    if min(args.installations, args.points, args.workers, args.probe_steps) < 1 or args.probes < 2:
        raise ValueError('expected positive experiment size and at least two probes')
    _, _, hashes = load_sources(args.first_person_mini)
    if any(k.startswith('impact:') for k in conditions):
        hashes['optical_impact'] = _hash(ROOT/'artifacts/tt5/mini-impact.json')
    schedules = [_seeds(args.seed, i, max(8, args.probes), args.points) for i in range(args.installations)]
    for schedule in schedules:
        schedule['calibration'] = schedule['calibration'][:args.probes]
    geometry = dict(probe_horizon=args.probe_steps, fast_hall=args.fast_hall, arena=dict(width=args.width, height=args.height, baseline=args.baseline,
                    venue=args.venue, max_steps=args.max_steps, frameskip=8, render_mode='mujoco', distractor=False))
    if args.race_to:
        geometry['race_to'] = args.race_to
    jobs = [(s, geometry, conditions, args.first_person_mini, args.cache) for s in schedules]
    modules = sorted((ROOT/'scripts'/'experimental').glob('tt5*.py'))
    report = dict(schema='tt5-native-hall-v1', seed=args.seed, schedules=schedules, geometry=geometry,
                  conditions={'first_person':conditions}, inputs_sha256=hashes,
                  implementation_sha256={p.name:_hash(p) for p in modules}, courses={'first_person':[]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for c in pool.map(course, jobs):
            rows = report['courses']['first_person']
            rows.append(c)
            report['summary'] = summarize(rows, conditions, args.race_to)
            report['wall_seconds'] = perf_counter()-start
            _write(args.output, report)
            print(json.dumps(dict(installation=c['schedule']['index'], errors=c['calibration_errors'],
                                 counts={k:_metrics(v['points']) for k,v in c['runs'].items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--first-person-mini', type=Path, default=ROOT/'artifacts/tt4/mini-first-person-player.json.gz')
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--height', type=int, default=144)
    p.add_argument('--baseline', type=float, default=.28)
    p.add_argument('--race-to', type=int, default=0, help='stop the course at this score with a two-point lead')
    p.add_argument('--max-steps', type=int, default=140)
    p.add_argument('--venue', choices=['studio','hall'], default='hall')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--installations', type=int, default=5)
    p.add_argument('--points', type=int, default=10)
    p.add_argument('--probes', type=int, default=8)
    p.add_argument('--probe-steps', type=int, default=34)
    p.add_argument('--cache', type=Path, help='reuse exact motor probe RGB within this local environment')
    p.add_argument('--fast-hall', action='store_true', help='coarser floor and box lighting for development')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--conditions', nargs='+', required=True)
    run(p.parse_args())
