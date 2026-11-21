"""Compare short rotary map recall and cold fit with a shared receiving servo."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples, fit_camera_body
from .tt6_body_probe import collect_probe, probe_commands
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous_trial import match_phases
from .tt6_memory import load, save
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory, outgoing_speed
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_player import ServiceMemory
from .tt8_service_trial import check_tapes
from .tt10_series_trial import _capture_tracker, _restore_tracker
from .tt12_body_trial import PHASES, play_point, summary, verify_game
from .tt12_motor_recall import align, loss
from .tt13_rotary_player import DriftCorrectedReturnPlayer, RotaryReturnPlayer


BASE = ROOT / 'artifacts/tt12/body-rotary-start.json.gz'
SERVE = ROOT / 'artifacts/tt12/body-rotary-practice.json.gz'
DONORS = [ROOT / f'artifacts/tt9/rotary-lift-{side}-player.json.gz' for side in ('a', 'b')]
MAPS = ('cold', 'recalled', 'commissioned')
BRANCHES = (*MAPS, 'coherent')
PROBE_SEED = 1737276661
HELD_SEED = 1762590803
FIXTURES = (88266083, 796518049)
HORIZON = 34
SUPPORT = 3


def path(folder, name):
    return folder / f'rotary-{name}.json.gz'


def prepare(folder):
    state = _read(BASE)
    cfg = state['cross_body_timing']
    schedule = _seeds(PROBE_SEED, 5732, 7, 0)
    arena = TT8ServiceArena(**cfg['arena'])
    hardware = DelayedWiring(TT6Wiring(arena, cfg['arena']['seed']))
    try:
        cache_hall_stereo(arena, reflections=True)
        histories, probes = [], []
        for index, seed in enumerate(schedule['calibration']):
            commands = probe_commands(schedule['player_seed'] + index, HORIZON, hardware.action_size)
            history = collect_probe(hardware, seed, commands)
            histories.append(history)
            probes.append(dict(seed=seed, actions=[action.tolist() for action, _ in history[1:]]))
        factory = body_tracker_factory(arena.cameras, histories[:SUPPORT])
        support = _samples(histories[:SUPPORT], factory)
        held = _samples(histories[SUPPORT:], factory)
        cold = fit_camera_body(histories[:SUPPORT], factory)
        candidates, alignments = [], []
        for donor in DONORS:
            source, = [model for model in load(donor)[0].models if model.action_size == hardware.action_size]
            recalled, alignment = align(source, support)
            candidates.append(recalled)
            alignments.append(dict(donor=str(donor.relative_to(ROOT)), donor_sha256=_hash(donor),
                                   unaligned_support_m=loss(source, support), **alignment))
        selected = min(range(len(candidates)), key=lambda i: alignments[i]['error_m'])
        library, _, _, _ = load(BASE)
        models = dict(cold=cold, recalled=candidates[selected], commissioned=library.models[cfg['model']])
        timing = {key: np.asarray(value) for key, value in cfg['timing']['cursor'].items()}
        errors = {}
        for name, model in models.items():
            shared = replace(model, **timing)
            errors[name] = dict(support_m=loss(model, support), held_m=loss(model, held),
                                shared_support_m=loss(shared, support), shared_held_m=loss(shared, held))
            library.models.append(shared)
        optics = dict(support=_capture_tracker(factory), commissioned=cfg['optics'])
    finally:
        arena.close()
    prepared = dict(arena=cfg['arena'], service_model=cfg['model'], optics=optics,
                    maps={name: len(library.models) - len(models) + index for index, name in enumerate(models)},
                    service_belief=cfg['service_belief'], profile=_read(SERVE)['chosen'])
    start = path(folder, 'start')
    _, prior, response, timing_memory = load(BASE)
    save(start, library, prior, response, timing_memory, service_memory=ServiceMemory(state.get('service_memory')))
    portable = _read(start)
    portable['rotary_return'] = prepared
    _write(start, portable)
    _write(path(folder, 'probes'), dict(schema='tt13-rotary-probes-v1', base_sha256=_hash(BASE),
                                       serve_sha256=_hash(SERVE), start_sha256=_hash(start), schedule=schedule,
                                       support_episodes=SUPPORT, probes=probes, alignments=alignments,
                                       selected=selected, errors=errors, support_poses=len(support[2]),
                                       held_poses=len(held[2])))
    print('RGB map errors', errors, 'alignment', alignments[selected], flush=True)


def restore(start, arena, branch):
    state = _read(start)
    cfg = state['rotary_return']
    library, prior, response, timing = load(start)
    fit = library.models[cfg['service_model']]
    receiving = library.models[cfg['maps']['recalled' if branch == 'coherent' else branch]]
    factory = _restore_tracker(arena.cameras, cfg['optics']['commissioned'])
    between_points = state.get('rotary_between_points', {})
    if between_points and between_points['branch'] != branch:
        raise ValueError('a continued game must use its saved receiving branch')
    flight = FlightMemory('fixed', between_points.get('flight', ()), between_points.get('impulses', ()))
    player_type = DriftCorrectedReturnPlayer if branch == 'coherent' else RotaryReturnPlayer
    player = player_type(fit, prior, factory, response, timing, receiving_model=receiving,
                         service_belief=cfg['service_belief'], flight_memory=flight, predictive=True,
                         service_memory=ServiceMemory(state.get('service_memory')),
                         fixed=int(cfg['profile'].partition('-')[2]))
    if between_points:
        prior.rng.bit_generator.state = deepcopy(state['rng'])
        if prior.span_memory is not None:
            span = state['motor_span']
            prior.span_memory.active = span['active']
            prior.span_memory.exposures = {int(key): count for key, count in span['exposures'].items()}
        player.role_rows = [np.asarray(row) for row in between_points['roles']]
    return library, player


def checkpoint(destination, start, library, player, branch):
    save(destination, library, player.prior, player.response, player.timing_memory,
         service_memory=player.service_memory)
    state = _read(destination)
    state['rotary_return'] = _read(start)['rotary_return']
    state['rotary_between_points'] = dict(branch=branch, flight=list(player.memory.history),
                                         impulses=list(player.memory.impulses),
                                         roles=np.asarray(player.role_rows).tolist())
    _write(destination, state)
    load(destination)


def optical_return(trace):
    """Measure pre-impact geometry and coherent departure only from stereo."""
    approaches = [frame for frame in trace if frame['horizon'] is not None and -2 <= frame['horizon'] <= 4]
    if not approaches:
        return dict(contact_prediction=None, closest=None, departure=None)
    first = approaches[0]
    approaches = [frame for frame in approaches if frame['tick'] <= first['tick'] + 12]
    anchor = min(approaches, key=lambda frame: abs(frame['horizon']))
    contact = anchor['tick'] + anchor['horizon']
    paired = [frame for frame in trace if abs(frame['tick'] - contact) <= 4 and frame['blade'] is not None
              and frame['ball'] is not None and frame['ball'][1] < -1.05]
    closest = None
    if paired:
        distances = [np.linalg.norm(np.asarray(f['ball']) - f['blade']) for f in paired]
        index = int(np.argmin(distances))
        frame = paired[index]
        difference = np.asarray(frame['ball']) - frame['blade']
        closest = dict(tick=frame['tick'], offset=difference.tolist(), distance_m=float(distances[index]),
                       blade_pose=frame['blade'], blade_pitch=frame['blade_pitch'],
                       desired_pitch=frame['desired_pitch'])
    departure = outgoing_speed(trace, contact)
    if departure is not None:
        velocity, position, ticks = departure
        departure = dict(velocity_frame=velocity.tolist(), position=position.tolist(), ticks=ticks)
    return dict(contact_prediction=contact, paired_frames=len(paired), closest=closest, departure=departure)


def add_diagnostics(row, player):
    row['optical'] = optical_return(player.trace) if row['server'] == -1 else None
    row['optical_trace'] = player.trace


def schedule(seed, index, count):
    ids = _seeds(seed, index, 0, count)['tests']
    return list(zip(ids, match_phases(PHASES, count, [seed, index]), strict=True))


def run(folder, branch, fixture, start=None, seed=None):
    start = start or path(folder, 'start')
    cfg = _read(start)['rotary_return']
    arena = TT8ServiceArena(**cfg['arena'])
    wiring = TT6Wiring(arena, cfg['arena']['seed'])
    held = fixture == 0
    if seed is None:
        seed = HELD_SEED if held else FIXTURES[fixture - 1]
    cases = {'-1': schedule(seed, 5733, 24 if held else 56)}
    first_server = 1 if fixture == 2 else -1
    if not held:
        cases['1'] = schedule(seed, 5734, 56)
    key = f'{branch}-held' if held else f'{branch}-game-{fixture}'
    output = path(folder, key)
    report = dict(schema='tt13-rotary-return-v1', arena=cfg['arena'], seed=seed,
                  installation_seed=cfg['arena']['seed'], branch=branch, fixture=fixture,
                  start_sha256=_hash(start), schedule=cases, points=[], practice=held)
    if not held:
        report.update(race_to=11, first_server=first_server, score=(0, 0), finished=False)
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player = restore(start, arena, branch)
        score = ServiceScore(first_server=first_server)
        used = {'-1': 0, '1': 0}
        for _ in range(24 if held else 56):
            if held:
                # Compare the same first shots with no cross-point updates.
                _, player = restore(start, arena, branch)
            server = -1 if held else score.server
            side = str(server)
            point_seed, phase = cases[side][used[side]]
            row = play_point(player, arena, wiring, seed, point_seed, phase, server)
            add_diagnostics(row, player)
            report['points'].append(row)
            used[side] += 1
            if not held:
                score.record(row['evaluator'])
                report.update(score=score.score, finished=score.finished)
                if score.finished:
                    break
        if not held:
            if not score.finished:
                raise RuntimeError(f'{key} did not finish')
            destination = path(folder, f'{key}-player')
            checkpoint(destination, start, library, player, branch)
            report['checkpoint_sha256'] = _hash(destination)
    finally:
        arena.close()
    report['metrics'] = {side: summary([row for row in report['points'] if str(row['server']) == side])
                         for side in cases}
    _write(output, report)
    written = _read(output)
    if held:
        check_tapes(written)
        replayed = len(written['points'])
    else:
        replayed = verify_game(written)
    report['replayed_points'] = replayed
    _write(output, report)
    print(key, report.get('score'), report['metrics'], flush=True)


def audit_resume(folder, branch, fixture):
    start = path(folder, 'start')
    source = path(folder, f'{branch}-game-{fixture}')
    saved = path(folder, f'{branch}-game-{fixture}-player')
    receipt = _read(source)
    if (_hash(start) != receipt['start_sha256'] or _hash(saved) != receipt['checkpoint_sha256']):
        raise ValueError('the audited game does not match the supplied checkpoints')
    cfg = _read(start)['rotary_return']
    arena = TT8ServiceArena(**cfg['arena'])
    wiring = TT6Wiring(arena, cfg['arena']['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        _, live = restore(start, arena, branch)
        for expected in receipt['points']:
            row = play_point(live, arena, wiring, receipt['seed'], expected['seed'],
                             expected['phase'], expected['server'])
            if row['action_sha256'] != expected['action_sha256'] or row['evaluator'] != expected['evaluator']:
                raise ValueError(f"RGB policy replay diverged on {expected['seed']}")
        _, restored = restore(saved, arena, branch)
        seed = 2122987616
        runs = []
        for (point_seed, phase), server in zip(schedule(seed, 6156, 4), (-1, 1, -1, 1), strict=True):
            direct = play_point(live, arena, wiring, seed, point_seed, phase, server)
            continued = play_point(restored, arena, wiring, seed, point_seed, phase, server)
            if direct != continued:
                raise ValueError(f'restored player diverged from uninterrupted play on {point_seed}')
            runs.append(dict(seed=point_seed, phase=phase, server=server, action_sha256=direct['action_sha256'],
                             public_reward=direct['public_reward']))
    finally:
        arena.close()
    report = dict(game_sha256=_hash(source), checkpoint_sha256=_hash(saved),
                  policy_replayed_points=len(receipt['points']), continuation_seed=seed, continuation=runs)
    _write(path(folder, f'{branch}-resume-audit'), report)
    print('replayed RGB policy and checkpoint', branch, len(receipt['points']), len(runs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    command = parser.add_subparsers(dest='command', required=True)
    command.add_parser('prepare')
    play = command.add_parser('play')
    play.add_argument('branch', choices=BRANCHES)
    play.add_argument('fixture', type=int, choices=(0, 1, 2), help='0 is the matched held cohort')
    play.add_argument('--resume', type=Path)
    play.add_argument('--seed', type=int, help='play a fresh schedule while retaining the fixture service order')
    audit = command.add_parser('audit-resume')
    audit.add_argument('branch', choices=BRANCHES)
    audit.add_argument('fixture', type=int, choices=(1, 2))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'play':
        run(args.output, args.branch, args.fixture, args.resume, args.seed)
    else:
        audit_resume(args.output, args.branch, args.fixture)
