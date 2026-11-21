"""Challenge timing-only optical service with a learned lateral correction."""
import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_memory import _serializable_body, load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt9_precontact_registration import RegistrationMemory, RegistrationPlayer
from .tt9_precontact_trial import _play_point, _record, replay


def _cases(seed, stream, count, phases):
    return list(zip(_seeds(seed, stream, 0, count)['tests'],
                    match_phases(phases, count, [seed, stream]), strict=True))


def run(args):
    library, prior, response, timing = load(args.resume)
    state = _read(args.resume)
    memory = RegistrationMemory(**state.get('service_registration', {}))
    install = _seeds(args.installation_seed, 0, 1, 1)
    calibration = _seeds(args.calibration_seed, 0, 8, 0)
    settings = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body='offset',
                    venue='hall', view='first_person', render_mode='mujoco',
                    width=192, height=144, baseline=.28, max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**settings)
    wiring = TT6Wiring(arena, install['arena_seed'])
    report = dict(schema='tt9-optical-service-registration-v1', seed=args.seed, arena=settings,
                  installation_seed=install['arena_seed'], calibration_seed=args.calibration_seed,
                  source_sha256=_hash(args.resume), implementation_sha256={
                      name: _hash(Path(__file__).with_name(name))
                      for name in ('tt9_precontact.py', 'tt9_precontact_trial.py',
                                   'tt9_precontact_registration.py', 'tt9_precontact_registration_trial.py',
                                   'tt8_service_drive.py', 'tt8_service_trial.py')
                  }, sections={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())

        def player(mode, learning=False):
            return RegistrationPlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                                      predictive=True, flight_memory=FlightMemory('fixed'),
                                      drive=ServiceDrive(), learn=False, registration=memory,
                                      intervention=mode, learn_registration=learning)

        if args.match_only:
            cases = _cases(args.seed, 2844, args.limit, ('ordinary', 'mixed', 'soft', 'delay'))
            for mode in ('full', 'fixed'):
                section = report['sections'][mode + '_match'] = dict(rows=[])
                score = ServiceScore(first_server=-1)
                active = player(mode)
                for seed, phase in cases:
                    row = _play_point(active, arena, wiring, args.seed, seed, phase, score.server)
                    section['rows'].append(row)
                    score.record(row['evaluator'])
                    section.update(score=score.score, finished=score.finished, winner=score.winner)
                    _record(report, section, args.output, mode)
                    if score.finished:
                        break
        else:
            phases = ('soft', 'mixed')
            practice = _cases(args.seed, 2841, args.practice, phases)
            held = _cases(args.seed, 2842, args.held, phases)
            if len(set(seed for seed, _ in practice + held)) != len(practice) + len(held):
                raise ValueError('practice and held tosses must be distinct')
            report.update(practice_schedule=practice, held_schedule=held)
            for name, mode, cases in [('practice', 'full', practice), ('frozen_registration', 'full', held),
                                      ('frozen_timing', 'timing', held), ('fixed_drive', 'fixed', held)]:
                section = report['sections'][name] = dict(rows=[], memory_before=memory.state())
                for seed, phase in cases:
                    row = _play_point(player(mode, learning=name == 'practice'), arena, wiring,
                                      args.seed, seed, phase)
                    section['rows'].append(row)
                    section['memory_after'] = memory.state()
                    _record(report, section, args.output, name)
                if name != 'practice' and section['memory_after'] != section['memory_before']:
                    raise RuntimeError('held service registration was updated')
            if args.checkpoint:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                state.update(models=[_serializable_body(model) for model in library.models],
                             service_registration=memory.state())
                _write(args.checkpoint, state)
    finally:
        arena.close()
    print(f'{replay(_read(args.output))} action tapes replayed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--installation-seed', type=int)
    p.add_argument('--calibration-seed', type=int)
    p.add_argument('--practice', type=int, default=12)
    p.add_argument('--held', type=int, default=12)
    p.add_argument('--limit', type=int, default=55)
    p.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt8/span-after-four.json.gz')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--match-only', action='store_true')
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        print(f'{replay(_read(args.output))} action tapes replayed')
    elif None in (args.seed, args.installation_seed, args.calibration_seed):
        p.error('supply the independent toss, installation and commissioning seeds')
    elif min(args.practice, args.held, args.limit) < 1:
        p.error('practice, held and match budgets must be positive')
    else:
        run(args)
