"""Compare follow-up strokes while preserving the frozen RGB service policy."""

import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import DrivePlayer, ServiceDrive
from .tt8_service_trial import PointInterface, check_tapes


def point_summary(row):
    result = row['evaluator']
    events = row['events']
    contacts = [event for event in events if event['kind'] == 'contact' and event['side'] == 1]
    own_serve = result['server'] == 1
    if own_serve:
        contacts = contacts[1:]
    return dict(server=result['server'], legal_serve=result['legal_serve'],
                opponent_returns=result['rival_returns'], legal_returns=result['legal_returns'],
                first_hit=contacts[0]['step'] if contacts else None,
                sustained=result['legal_returns'] >= (1 if own_serve else 2),
                points_for=result['points_for'], points_against=result['points_against'],
                reason=result['reason'])


def development_cases(source, section, include_all):
    previous = _read(source)
    rows = previous['points'] if section is None else previous['sections'][section]['rows']
    return previous, [(row['seed'], row['phase'], row['server']) for row in rows
                      if include_all or (row['evaluator']['legal_serve'] and row['evaluator']['rival_returns'] > 0)]


def run(args):
    library, prior, response, timing = load(args.resume)
    schedule = _seeds(args.seed, 0, 8, args.limit)
    install = _seeds(args.seed + 109, 0, 1, 1)
    arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=args.body, venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    cases = None
    if args.development is not None:
        previous, cases = development_cases(args.development, args.section, args.include_all)
        if not cases:
            raise ValueError('development receipt has no legal reply opportunities')
        if previous['seed'] != args.seed:
            raise ValueError('use the same seed as the development receipt')
        arena_args = previous['arena']
        schedule = previous.get('schedule', schedule)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    report = dict(schema='tt9-serve-follow-up-v1', seed=args.seed, arena=arena_args, schedule=schedule,
                  installation_seed=arena_args['seed'], source_sha256=_hash(args.resume), serve_rate=args.serve_rate,
                  development=None if cases is None else dict(source_sha256=_hash(args.development),
                                                               section=args.section, cases=cases),
                  sections={}, implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                                       ('tt9_recovery_trial.py', 'tt9_recovery.py')})
    phases = match_phases(args.phases, args.limit, [args.seed, 7919])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())
        for name in args.policies:
            kwargs = dict(predictive=True, flight_memory=FlightMemory('fixed'),
                          drive=ServiceDrive(rate=args.serve_rate), learn=False)
            common = fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing)
            if name == 'fixed':
                player = DrivePlayer(*common, **kwargs)
            else:
                from .tt9_recovery import RecoveryPlayer

                player = RecoveryPlayer(*common, **kwargs, follow_up=name)
            section = dict(points=[])
            report['sections'][name] = section
            if cases is None:
                section.update(race_to=11, first_server=args.first_server, practice=False)
            score = ServiceScore(first_server=args.first_server)
            starts = cases if cases is not None else zip(schedule['tests'], phases, [None] * args.limit, strict=True)
            for seed, phase, server in starts:
                server = score.server if server is None else server
                interface = PointInterface(wiring, prepare_point(arena, phase), server=server, seed=args.seed)
                row = _point(interface, player, seed)
                row.update(_counts(row), phase=phase, server=server, applied=interface.applied,
                           summary=point_summary(row))
                if args.trace:
                    row['optical_trace'] = player.trace
                section['points'].append(row)
                if cases is None:
                    score.record(row['evaluator'])
                    section.update(score=score.score, finished=score.finished)
                _write(args.output, report)
                print(name, phase, seed, section.get('score'), row['summary'], flush=True)
                if score.finished:
                    break
    finally:
        arena.close()
    verify(_read(args.output))


def verify(report):
    for section in report['sections'].values():
        check_tapes(dict(report, **section))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--body', choices=('offset', 'gantry'), default='offset')
    parser.add_argument('--first-server', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--limit', type=int, default=65)
    parser.add_argument('--phases', nargs='+', default=['ordinary', 'mixed', 'soft', 'delay'])
    parser.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt8/span-after-four.json.gz')
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--serve-rate', type=float, default=.132)
    parser.add_argument('--policies', nargs='+', default=['fixed', 'velocity'])
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--development', type=Path)
    parser.add_argument('--section')
    parser.add_argument('--include-all', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(_read(args.output))
    elif args.limit <= 0:
        parser.error('the point limit must be positive')
    else:
        run(args)
