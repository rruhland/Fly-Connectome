"""Pair a practiced service drive against fixed pace on unseen tosses."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_drive import DrivePlayer, ServiceDrive
from .tt8_service_player import ServiceMemory, ServicePlayer
from .tt8_service_trial import PointInterface, check_tapes


def replay(report):
    check_tapes(dict(report, points=[row for section in report['sections'].values() for row in section['rows']]))


def run(args):
    library, prior, response, timing = load(args.resume)
    install = _seeds(args.seed + 109, 0, 1, 1)
    schedule = _seeds(args.seed, 0, 8, 0)
    point_seeds = _seeds(args.seed, 807, 0, args.practice + args.test)['tests']
    phases = match_phases(args.phases, len(point_seeds), [args.seed, 807])
    starts = list(zip(point_seeds, phases, strict=True))
    arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=args.body, venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, install['arena_seed'])
    report = dict(schema='tt8-optical-service-drive-v1', seed=args.seed, arena=arena_args,
                  implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                         ('tt8_service_arena.py', 'tt8_service_drive.py', 'tt8_service_drive_trial.py',
                                          'tt8_service_player.py')},
                  source_sha256=_hash(args.resume), installation_seed=install['arena_seed'],
                  practice_points=args.practice, held_points=args.test, sections={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())

        def play(name, cases, drive=None, learning=False):
            common = (fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing))
            settings = dict(predictive=True, flight_memory=FlightMemory('fixed'))
            if drive is None:
                player = ServicePlayer(*common, **settings, service_memory=ServiceMemory(), fixed=3)
            else:
                player = DrivePlayer(*common, **settings, drive=drive, learn=learning)
            section = dict(rows=[], drive_before=None if drive is None else asdict(drive))
            report['sections'][name] = section
            for seed, phase in cases:
                disturbance = prepare_point(arena, phase)
                interface = PointInterface(wiring, disturbance, server=1, seed=args.seed)
                row = _point(interface, player, seed)
                row.update(_counts(row), phase=phase, server=1, applied=interface.applied)
                section['rows'].append(row)
                stats = [item['evaluator'] for item in section['rows']]
                section.update(drive_after=None if drive is None else asdict(drive),
                               legal=sum(s['legal_serve'] for s in stats), lets=sum(s['serve_let'] for s in stats),
                               points_for=sum(s['points_for'] for s in stats))
                _write(args.output, report)
                result = row['evaluator']
                print(name, len(stats), phase, result['legal_serve'], result['reason'],
                      row.get('serve_drive'), flush=True)

        learned = ServiceDrive()
        play('practice', starts[:args.practice], learned, learning=True)
        held = starts[args.practice:]
        play('learned_drive', held, learned)
        play('fixed_hard', held)
        play('fixed_drive', held, ServiceDrive())
    finally:
        arena.close()
    replay(_read(args.output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--body', choices=('gantry', 'offset'), required=True)
    p.add_argument('--phases', nargs='+', default=['soft', 'mixed'])
    p.add_argument('--practice', type=int, default=4)
    p.add_argument('--test', type=int, default=8)
    p.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt6/continual-player.json.gz')
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        replay(_read(args.output))
    elif min(args.practice, args.test) < 1:
        p.error('practice and test point counts must be positive')
    else:
        run(args)
