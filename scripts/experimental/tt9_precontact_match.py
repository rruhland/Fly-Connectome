"""Continue optical service registration in full alternating table games."""
import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load, save
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_player import ServiceMemory
from .tt9_precontact_registration import RegistrationMemory, RegistrationPlayer
from .tt9_precontact_registration_trial import _cases
from .tt9_precontact_trial import _play_point, _record, replay


def save_player(path, previous, library, player):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        temporary = Path(folder) / 'rally-state.json.gz'
        save(temporary, library, player.prior, player.response, player.timing_memory,
             service_memory=ServiceMemory(previous.get('service_memory')), service_drive=player.drive)
        state = dict(previous, **_read(temporary))
        state['service_registration'] = player.approach.state()
        _write(path, state)


def run(args):
    library, prior, response, timing = load(args.resume)
    previous = _read(args.resume)
    state = previous['service_registration']
    installation = _seeds(args.installation_seed, 0, 1, 1)
    arena_args = dict(seed=installation['arena_seed'], view_seed=installation['view_seed'],
                      body='offset', venue='hall', view='first_person', render_mode='mujoco',
                      width=192, height=144, baseline=.28, max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, installation['arena_seed'])
    calibration = _seeds(args.seed, 0, 8, 0)
    phases = ('ordinary', 'mixed', 'soft', 'delay')
    cases = _cases(args.seed, 2931, args.limit, phases)
    report = dict(schema='tt9-optical-service-match-v1', seed=args.seed, arena=arena_args,
                  installation_seed=installation['arena_seed'], source_sha256=_hash(args.resume),
                  implementation_sha256={
                      name: _hash(Path(__file__).with_name(name))
                      for name in ('tt9_precontact.py', 'tt9_precontact_registration.py',
                                   'tt9_precontact_match.py', 'tt9_precontact_trial.py')
                  }, schedule=cases, sections={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())

        for mode in ('full', 'fixed'):
            memory = RegistrationMemory(**deepcopy(state))
            active = RegistrationPlayer(
                fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                predictive=True, flight_memory=FlightMemory('fixed'), drive=ServiceDrive(),
                learn=False, registration=memory, intervention=mode, learn_registration=mode == 'full')
            section = report['sections'][mode + '_match'] = dict(rows=[], memory_before=memory.state())
            score = ServiceScore(first_server=-1)
            for seed, phase in cases:
                row = _play_point(active, arena, wiring, args.seed, seed, phase, score.server)
                section['rows'].append(row)
                score.record(row['evaluator'])
                section.update(score=score.score, finished=score.finished, winner=score.winner,
                               memory_after=memory.state(),
                               player_returns=sum(t['player_returns'] for t in section['rows']),
                               rival_returns=sum(t['rival_returns'] for t in section['rows']),
                               sustained_points=sum(t['player_returns'] >= 2 for t in section['rows']))
                _record(report, section, args.output, mode)
                if score.finished:
                    break
            if mode == 'full':
                save_player(args.checkpoint, previous, library, active)
                load(args.checkpoint)
                saved = _read(args.checkpoint)
                if saved['service_registration'] != memory.state() or saved['service_drive']['rate'] != .132:
                    raise RuntimeError('checkpoint did not preserve the current registration and drive')
                report['checkpoint_sha256'] = _hash(args.checkpoint)
                _write(args.output, report)
            elif section['memory_after'] != section['memory_before']:
                raise RuntimeError('fixed drive changed registration memory')
    finally:
        arena.close()
    print(f'{replay(_read(args.output))} action tapes replayed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--installation-seed', type=int)
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'{replay(_read(args.output))} action tapes replayed')
    elif None in (args.seed, args.installation_seed, args.resume, args.checkpoint):
        parser.error('supply the point and installation seeds plus the input and output checkpoints')
    elif args.limit < 1:
        parser.error('point limit must be positive')
    else:
        run(args)
