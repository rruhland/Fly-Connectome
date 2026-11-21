"""Compare optically learned service timing with fixed and pooled timing."""
import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import _serializable_body, load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import PointInterface, check_tapes
from .tt9_precontact import ApproachMemory, ApproachPlayer


def replay(report):
    total = 0
    for section in report['sections'].values():
        rows = section['rows']
        settings = dict(report, points=rows)
        if 'score' in section:
            settings.update(race_to=11, first_server=-1, score=section['score'], finished=section['finished'])
        check_tapes(settings)
        total += len(rows)
    return total


def run(args):
    library, prior, response, timing = load(args.resume)
    state = _read(args.resume)
    memory = ApproachMemory(**state.get('service_approach', {}))
    installation = _seeds(args.installation_seed or args.seed + 109, 0, 1, 1)
    schedule = _seeds(args.seed, 0, 8, 0)
    arena_args = dict(seed=installation['arena_seed'], view_seed=installation['view_seed'],
                      body='offset', venue='hall', view='first_person', render_mode='mujoco',
                      width=192, height=144, baseline=.28, max_steps=420,
                      frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, installation['arena_seed'])
    report = dict(schema='tt9-optical-service-approach-v1', seed=args.seed, arena=arena_args,
                  installation_seed=installation['arena_seed'], source_sha256=_hash(args.resume),
                  implementation_sha256={name: _hash(Path(__file__).with_name(name))
                                         for name in ('tt9_precontact.py', 'tt9_precontact_trial.py',
                                                      'tt8_service_drive.py', 'tt8_service_trial.py')},
                  sections={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())
        if args.match_only:
            _play_matches(args, report, arena, wiring, fit, factory, prior, response, timing, memory)
        else:
            _practice(args, report, arena, wiring, fit, factory, prior, response, timing, memory)
            if args.checkpoint:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                state.update(models=[_serializable_body(model) for model in library.models],
                             service_approach=memory.state())
                _write(args.checkpoint, state)
    finally:
        arena.close()
    print(f'{replay(_read(args.output))} action tapes replayed', flush=True)


def _player(mode, fit, factory, prior, response, timing, memory, *, learning=False):
    return ApproachPlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                          predictive=True, flight_memory=FlightMemory('fixed'), drive=ServiceDrive(),
                          learn=False, approach=memory, mode=mode, learn_approach=learning)


def _play_point(player, arena, wiring, run_seed, seed, phase, server=1):
    interface = PointInterface(wiring, prepare_point(arena, phase), server=server, seed=run_seed)
    row = _point(interface, player, seed)
    return dict(**row, **_counts(row), phase=phase, server=server, applied=interface.applied)


def _record(report, section, output, mode):
    own = [row for row in section['rows'] if row['server'] == 1]
    stats = [row['evaluator'] for row in own]
    section.update(own_serves=len(own), contacts=sum(row['service_contact'] for row in stats),
                   legal=sum(row['legal_serve'] for row in stats), lets=sum(row['serve_let'] for row in stats),
                   wins_on_serve=sum(row['points_for'] for row in stats))
    _write(output, report)
    row = section['rows'][-1]
    approach = row['precontact']
    outcome = row['evaluator']
    print(mode, len(section['rows']), row['phase'], outcome['service_contact'], outcome['legal_serve'],
          outcome['reason'], 'lead', None if approach is None else round(approach['lead'], 3),
          section.get('score'), flush=True)


def _practice(args, report, arena, wiring, fit, factory, prior, response, timing, memory):
    practice = _seeds(args.seed, 1741, 0, args.practice)['tests']
    held = _seeds(args.seed, 1742, 0, args.held)['tests']
    if set(practice) & set(held) or len(set(practice + held)) != len(practice + held):
        raise ValueError('toss seeds must be distinct across practice and held sets')
    phases = ('soft', 'mixed')
    practice_cases = list(zip(practice, match_phases(phases, args.practice, [args.seed, 1741]), strict=True))
    held_cases = list(zip(held, match_phases(phases, args.held, [args.seed, 1742]), strict=True))
    report.update(practice_schedule=practice_cases, held_schedule=held_cases)

    for name, mode, cases in [('practice', 'contextual', practice_cases),
                              ('frozen_contextual', 'contextual', held_cases),
                              ('frozen_mean', 'mean', held_cases),
                              ('fixed_drive', 'fixed', held_cases)]:
        section = report['sections'][name] = dict(rows=[], memory_before=memory.state())
        for seed, phase in cases:
            # Each comparison receives the same commissioned body, shared rally
            # checkpoint, and latency prior. The approach model alone learns
            # across distinct practice points.
            player = _player(mode, fit, factory, prior, response, timing, memory,
                             learning=name == 'practice')
            section['rows'].append(_play_point(player, arena, wiring, args.seed, seed, phase))
            section['memory_after'] = memory.state()
            _record(report, section, args.output, name)
        if name != 'practice' and section['memory_after'] != section['memory_before']:
            raise RuntimeError('held service approach was updated')


def _play_matches(args, report, arena, wiring, fit, factory, prior, response, timing, memory):
    seeds = _seeds(args.seed, 1743, 0, args.limit)['tests']
    phases = match_phases(('ordinary', 'mixed', 'soft', 'delay'), args.limit, [args.seed, 1743])
    for mode in ('contextual', 'fixed'):
        section = report['sections'][mode + '_match'] = dict(rows=[])
        scoreboard = ServiceScore(first_server=-1)
        player = _player(mode, fit, factory, prior, response, timing, memory)
        for seed, phase in zip(seeds, phases, strict=True):
            row = _play_point(player, arena, wiring, args.seed, seed, phase, scoreboard.server)
            section['rows'].append(row)
            scoreboard.record(row['evaluator'])
            section.update(score=scoreboard.score, finished=scoreboard.finished, winner=scoreboard.winner)
            _record(report, section, args.output, mode)
            if scoreboard.finished:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=398512109)
    parser.add_argument('--installation-seed', type=int)
    parser.add_argument('--practice', type=int, default=16)
    parser.add_argument('--held', type=int, default=14)
    parser.add_argument('--limit', type=int, default=55)
    parser.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt8/span-after-four.json.gz')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--match-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'{replay(_read(args.output))} action tapes replayed')
    elif min(args.practice, args.held, args.limit) < 1:
        parser.error('practice, held and match budgets must be positive')
    else:
        run(args)
