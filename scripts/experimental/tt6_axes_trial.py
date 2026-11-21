"""Run equal-probe transfer comparisons on previously unseen physical axes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _read, _seeds, _write, load_sources
from .tt5_vision import motion_tracker_factory
from .tt6_axes import TiltedAxisArena, pulse_receipt
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import fit_camera_body
from .tt6_body_probe import probe_histories
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous import ContinualPlayer
from .tt6_continuous_trial import point_condition
from .tt6_memory import load
from .tt6_racket import RacketAdaptation, set_racket_response
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring, TimingMemory


PHASES = ('ordinary', 'soft', 'mixed', 'ordinary', 'mixed', 'delay', 'ordinary')
PRACTICE_PHASES = ('ordinary', 'mixed', 'delay', 'ordinary')
GEOMETRY = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=140)
MEMORY = ROOT / 'artifacts' / 'tt6' / 'continual-player.json.gz'
MINI = ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz'


def _play(layout, phase_seed, index, phases, conditions, cache):
    schedule = _seeds(phase_seed, index, 8, len(phases))
    arena = TiltedAxisArena(schedule['arena_seed'], layout=layout, view_seed=schedule['view_seed'], **GEOMETRY)
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, schedule['arena_seed'])
        geometry = dict(arena=GEOMETRY, probe_horizon=34, layout=layout, reflected_hall=True,
                        axes_sha256=_hash(Path(__file__).with_name('tt6_axes.py')))
        histories, probes = probe_histories(wiring, schedule, geometry, cache)
        fit = None
        failures = []
        for tracker_factory in (body_tracker_factory, motion_tracker_factory):
            try:
                factory = tracker_factory(arena.cameras, histories)
                fit = fit_camera_body(histories, factory)
                break
            except ValueError as error:
                failures.append(str(error))
        result = dict(layout=layout, schedule=schedule, phases=phases, wiring=wiring.state(), calibration=probes,
                      fit_failures=failures, runs={})
        if fit is None:
            return result
        result['fit'] = fit.stats()
        for name, (prior, response, timing) in conditions.items():
            player = ContinualPlayer(fit, prior, factory, response, timing, predictive=True)
            initial_prior = prior.state()
            rows = []
            for phase, seed in zip(phases, schedule['tests'], strict=True):
                rubber, disturbance = point_condition(phase)
                set_racket_response(arena, rubber)
                interface = DelayedWiring(wiring, disturbance, seed=phase_seed)
                row = _point(interface, player, seed)
                rows.append(dict(phase=phase, applied=interface.applied, **row, **_counts(row)))
            result['runs'][name] = dict(points=rows, summary=_metrics(rows), initial_prior=initial_prior,
                                        final_prior=prior.state())
        return result
    finally:
        arena.close()


def _practice(job):
    tag, layouts, seed, fresh, cache = job
    practice = [_play(layout, seed, i, PRACTICE_PHASES, {tag: fresh}, cache) for i, layout in enumerate(layouts)]
    return practice, fresh


def _qualify(job):
    return _play(*job)


def verify(path):
    report = _read(path)
    courses = [(report['seed'], visit) for visit in report['held']]
    for group in report['practice'].values():
        courses.extend((report['seed'] + 26129, visit) for visit in group)
    points = returns = 0
    for upset_seed, visit in courses:
        schedule = visit['schedule']
        arena = TiltedAxisArena(schedule['arena_seed'], layout=visit['layout'], view_seed=schedule['view_seed'],
                               **report['geometry'])
        try:
            wiring = TT6Wiring(arena, schedule['arena_seed'])
            for run in visit['runs'].values():
                for row in run['points']:
                    rubber, disturbance = point_condition(row['phase'])
                    set_racket_response(arena, rubber)
                    interface = DelayedWiring(wiring, disturbance, seed=upset_seed)
                    interface.reset(seed=row['seed'], render=False)
                    actions = np.asarray(row['actions'], dtype='<f8')
                    if sha256(actions.tobytes()).hexdigest() != row['action_sha256']:
                        raise ValueError(f'command checksum differs at {row["seed"]}')
                    for action in actions:
                        interface.step(action, render=False)
                        mujoco.mj_kinematics(arena.model, arena.data)
                    np.testing.assert_array_equal(interface.applied, row['applied'])
                    if arena.point_result() != row['evaluator'] or json.dumps(arena.events) != json.dumps(row['events']):
                        raise ValueError(f'referee events differ at {row["seed"]}')
                    points += 1
                    returns += arena.legal_returns
        finally:
            arena.close()
    print(f'Verified {points} points and {returns} legal returns.')


def run(args):
    _, experienced, prior_response, prior_timing = load(MEMORY)
    native, states, hashes = load_sources(MINI)
    hashes['varied_42_point_checkpoint'] = _hash(MEMORY)
    fresh = (TransferPrior('first_person_mini', native, args.seed, mini=states['first_person_mini']),
             RacketAdaptation(gain=prior_response.gain), TimingMemory())
    layouts = ('held-a', 'held-b')
    report = dict(schema='tt6-held-physical-axes-v1', seed=args.seed, geometry=GEOMETRY,
                  reflected_hall=True, inputs_sha256=hashes, practice={}, held=[],
                  pulse_audit=[pulse_receipt(layout, args.seed + i) for i, layout in enumerate(layouts)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.axis_practice:
            names = ('axis-varied', 'axis-standard')
            plans = (('practice-a', 'practice-b'), ('standard', 'standard'))
            practice_seed = args.seed + 26129
            jobs = [(name, plan, practice_seed, deepcopy(fresh), args.cache) for name, plan in zip(names, plans)]
            conditions = {}
            for name, (play, learned) in zip(names, pool.map(_practice, jobs), strict=True):
                report['practice'][name] = play
                conditions[name] = learned
                _write(args.output, report)
        else:
            conditions = {'varied-42': (experienced, prior_response, prior_timing), 'mini': fresh}
        jobs = [(layout, args.seed, i, PHASES, deepcopy(conditions), args.cache) for i, layout in enumerate(layouts)]
        for held in pool.map(_qualify, jobs):
            report['held'].append(held)
            report['summary'] = {name: _metrics([row for station in report['held']
                for row in station['runs'].get(name, {}).get('points', [])]) for name in conditions}
            _write(args.output, report)
            print(held['layout'], {name: (row['summary']['with_return'], row['summary']['player_returns'],
                  row['summary']['sustained']) for name, row in held['runs'].items()}, held['fit_failures'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--seed', type=int, default=14590277)
    cli.add_argument('--workers', type=int, default=2)
    cli.add_argument('--cache', type=Path)
    cli.add_argument('--axis-practice', action='store_true')
    cli.add_argument('--verify', action='store_true', help='replay the receipt and check applied actions and event records')
    args = cli.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args)
