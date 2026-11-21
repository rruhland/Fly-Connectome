"""Play physical serve practice or alternating-service games from first-person stereo."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path


from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load, save
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory, FlightPlayer
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_player import ServiceMemory, ServicePlayer, STROKES


class PointInterface(DelayedWiring):
    def __init__(self, *args, server, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = server

    def reset(self, **kwargs):
        return super().reset(server=self.server, **kwargs)


def check_tapes(report):
    arena = TT8ServiceArena(**report['arena'])
    wiring = TT6Wiring(arena, report['installation_seed'])
    score = (ServiceScore(report['race_to'], report['first_server'])
             if 'race_to' in report and not report.get('practice') else None)
    try:
        for row in report['points']:
            disturbance = prepare_point(arena, row['phase'])
            interface = PointInterface(wiring, disturbance, server=row['server'], seed=report['seed'])
            interface.reset(seed=row['seed'], render=False)
            for command in row['actions']:
                interface.step(command, render=False)
            if arena.point_result() != row['evaluator'] or interface.applied != row['applied']:
                raise ValueError(f"service replay diverged on point {row['seed']}")
            if json.loads(json.dumps(arena.events)) != row['events']:
                raise ValueError(f"service events diverged on point {row['seed']}")
            if score is not None:
                score.record(arena.point_result())
        if score is not None and (list(score.score) != report['score'] or score.finished != report['finished']):
            raise ValueError('recorded game score differs from the replayed points')
    finally:
        arena.close()


def run(args):
    library, prior, response, timing = load(args.resume)
    schedule = _seeds(args.seed, 0, 8, args.limit)
    install = _seeds(args.installation_seed if args.installation_seed is not None else args.seed + 109, 0, 1, 1)
    arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=args.body, venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    cache_hall_stereo(arena, reflections=True, compiled_geometry=args.compiled_geometry)
    wiring = TT6Wiring(arena, install['arena_seed'])
    checkpoint = _read(args.resume)
    memory = ServiceMemory(_read(args.service_in) if args.service_in else checkpoint.get('service_memory'))
    drive = None
    drive_state = checkpoint.get('service_drive')
    if args.drive_from is not None:
        if args.drive is None:
            raise ValueError('select online or frozen drive when supplying optical drive practice')
        drive_state = _read(args.drive_from)['sections']['practice']['drive_after']
    if args.drive is not None or drive_state is not None:
        from .tt8_service_drive import DrivePlayer, ServiceDrive
        if args.receiver_only or args.fixed is not None or args.service_in is not None:
            raise ValueError('a drive checkpoint uses its stored serving policy')
        drive = ServiceDrive(**(drive_state or {}))
    lift_state = checkpoint.get('receiving_lift')
    lift = None
    lift_mode = args.receive_lift or (lift_state['mode'] if lift_state else None)
    if lift_mode is not None:
        from .tt8_contact import PairedTimingMemory, practice_samples
        from .tt9_receiving_lift import LiftServicePlayer

        if drive is not None or args.receiver_only:
            raise ValueError('receiving lift uses the toss-recognizing service player')
        if args.receive_practice:
            lift = PairedTimingMemory(practice_samples(_read(args.receive_practice)))
        elif lift_state:
            lift = PairedTimingMemory(**lift_state['memory'])
        else:
            lift = PairedTimingMemory()
    scoreboard = ServiceScore(target=args.race_to, first_server=args.first_server)
    phases = match_phases(args.phases, args.limit, [args.seed, 7919])
    try:
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        flight = FlightMemory('fixed')
        if drive is not None:
            player = DrivePlayer(fit, prior, factory, response, timing, predictive=True, flight_memory=flight,
                                 drive=drive, learn=args.drive != 'frozen')
        elif lift is not None:
            player = LiftServicePlayer(fit, prior, factory, response, timing, predictive=True, flight_memory=flight,
                                       service_memory=memory, fixed=args.fixed, contact_memory=lift,
                                       fixed_lift=1 if lift_mode == 'early' else None)
        elif args.receiver_only:
            player = FlightPlayer(fit, prior, factory, response, timing, predictive=True, flight_memory=flight)
        else:
            player = ServicePlayer(fit, prior, factory, response, timing, predictive=True, flight_memory=flight,
                                   service_memory=memory, fixed=args.fixed)
        report = dict(schema='tt8-serve-1', seed=args.seed, arena=arena_args, installation_seed=install['arena_seed'],
                      source=_hash(args.resume), service_source=_hash(args.service_in) if args.service_in else None,
                      practice=args.practice, race_to=args.race_to, first_server=args.first_server,
                      receiver_only=args.receiver_only, fixed=args.fixed, points=[], schedule=schedule,
                      probes=probes, attempts=attempts, before=memory.state())
        if lift is not None:
            report.update(receiving_lift_mode=lift_mode, receiving_lift_before=lift.state(),
                          receiving_practice_sha256=_hash(args.receive_practice) if args.receive_practice else None)
        if drive is not None:
            report.update(drive_mode=args.drive or 'online', drive_before=asdict(drive),
                          drive_practice_sha256=_hash(args.drive_from) if args.drive_from else None)
        args.output.parent.mkdir(exist_ok=True, parents=True)
        for seed, phase in zip(schedule['tests'], phases, strict=True):
            server = 1 if args.practice else scoreboard.server
            interface = PointInterface(wiring, prepare_point(arena, phase), server=server, seed=args.seed)
            row = _point(interface, player, seed)
            row.update(_counts(row), server=server, phase=phase, applied=interface.applied)
            if args.trace:
                row['optical_trace'] = player.trace
            report['points'].append(row)
            if not args.practice:
                scoreboard.record(row['evaluator'])
            report.update(score=scoreboard.score, finished=scoreboard.finished, service_state=memory.state())
            if lift is not None:
                report['receiving_lift_after'] = lift.state()
            if drive is not None:
                report['drive_after'] = asdict(drive)
            _write(args.output, report)
            print(len(report['points']), server, row.get('serve_stroke'), row['evaluator'], flush=True)
            if scoreboard.finished:
                break
        if args.checkpoint:
            save(args.checkpoint, library, prior, response, timing, service_memory=memory, service_drive=drive,
                 receiving_lift=None if lift is None else dict(mode=lift_mode, memory=lift.state()))
        if args.service_out:
            _write(args.service_out, memory.state())
    finally:
        arena.close()
    check_tapes(_read(args.output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--installation-seed', type=int)
    p.add_argument('--body', choices=('gantry', 'offset', 'rotary'), default='gantry')
    p.add_argument('--limit', type=int, default=65)
    p.add_argument('--race-to', type=int, default=11)
    p.add_argument('--first-server', type=int, choices=(-1, 1), default=1)
    p.add_argument('--practice', action='store_true', help='learner serves every point')
    p.add_argument('--phases', nargs='+', default=['ordinary'])
    p.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt6/continual-player.json.gz')
    p.add_argument('--service-in', type=Path)
    p.add_argument('--service-out', type=Path)
    p.add_argument('--checkpoint', type=Path, help='save shared body, reward, timing, and service state together')
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--compiled-geometry', action=argparse.BooleanOptionalAction, default=False,
                   help='reuse native GL geometry lists in the reflective renderer')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--receive-lift', choices=('online', 'early'), help='use matched visual lift timing on incoming strokes')
    p.add_argument('--receive-practice', type=Path, help='initialize lift timing from separate matched practice')
    p.add_argument('--drive-from', type=Path, help='initialize rate from a scored optical drive practice receipt')
    policy = p.add_mutually_exclusive_group()
    policy.add_argument('--drive', choices=('online', 'frozen'))
    policy.add_argument('--receiver-only', action='store_true')
    policy.add_argument('--fixed', type=int, choices=range(len(STROKES)))
    args = p.parse_args()
    if args.receive_practice and args.receive_lift != 'online':
        p.error('use --receive-lift online to initialize from practice')
    if args.limit <= 0:
        p.error('the point limit must be positive')
    run(args)
