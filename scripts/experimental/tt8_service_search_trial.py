"""Learn an offset-arm service stroke through scored RGB practice."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import DrivePlayer, ServiceDrive
from .tt8_service_search import SearchPlayer, ServiceStroke, proposals
from .tt8_service_trial import PointInterface, check_tapes


def replay(report):
    rows = [row for section in report['sections'].values() for row in section['rows']]
    check_tapes(dict(report, points=rows))
    return len(rows)


def run(args):
    library, prior, response, timing = load(args.resume)
    station = _seeds(args.seed + 109, 0, 1, 1)
    schedule = _seeds(args.seed, 0, 8, 0)
    point_budget = args.generations + args.held
    starts = _seeds(args.seed, 1717, 0, point_budget)['tests']
    phases = match_phases(['soft', 'mixed'], point_budget, [args.seed, 1717])
    practice = list(zip(starts[:args.generations], phases[:args.generations], strict=True))
    held = list(zip(starts[args.generations:], phases[args.generations:], strict=True))
    arena_args = dict(seed=station['arena_seed'], view_seed=station['view_seed'], body='offset', venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, station['arena_seed'])
    report = dict(schema='tt8-rgb-service-stroke-search-v1', seed=args.seed, arena=arena_args,
                  installation_seed=station['arena_seed'], source_sha256=_hash(args.resume),
                  implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                         ('tt8_service_search_trial.py', 'tt8_service_search.py',
                                          'tt8_service_drive.py', 'tt8_service_arena.py')},
                  practice_schedule=practice, held_schedule=held, search_updates=[], sections={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())

        def player(stroke):
            common = (fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing))
            kwargs = dict(predictive=True, flight_memory=FlightMemory('fixed'))
            if stroke is None:
                return DrivePlayer(*common, **kwargs, drive=ServiceDrive(), learn=False)
            return SearchPlayer(*common, **kwargs, stroke=stroke)

        def play(active, seed, phase, server):
            point = PointInterface(wiring, prepare_point(arena, phase), seed=args.seed, server=server)
            row = _point(point, active, seed)
            row.update(_counts(row), phase=phase, server=server, applied=point.applied)
            return row

        def save_summary(section):
            own = [row['evaluator'] for row in section['rows'] if row['server'] == 1]
            section.update(attempts=len(own), legal=sum(row['legal_serve'] for row in own),
                           points_won_on_serve=sum(row['points_for'] for row in own),
                           lets=sum(row['serve_let'] for row in own))
            _write(args.output, report)

        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 483]))
        best = ServiceStroke()
        training = dict(rows=[])
        report['sections']['search'] = training
        for index, (seed, phase) in enumerate(practice):
            candidates = proposals(best, rng, index)
            scores = []
            trial_rows = []
            for candidate in candidates:
                row = play(player(candidate), seed, phase, 1)
                row.update(stroke=asdict(candidate), generation=index)
                trial_rows.append(row)
                evidence = row['optical_service']
                scores.append(evidence['score'] if evidence is not None else 0.)
                training['rows'].append(row)
            top = int(np.argmax(scores))
            accepted = top if scores[top] > scores[0] + .035 else 0
            best = candidates[accepted]
            report['search_updates'].append(dict(seed=seed, phase=phase, scores=scores, accepted=accepted))
            report['best_stroke'] = asdict(best)
            save_summary(training)
            print('practice', index + 1, phase, np.round(scores, 3).tolist(),
                  [row['evaluator']['legal_serve'] for row in trial_rows], accepted, asdict(best), flush=True)
            if index >= 2 and max(max(row['scores']) for row in report['search_updates']) < .20:
                report['stalled'] = True
                break

        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write(args.checkpoint, dict(schema='tt8-rgb-service-stroke-v1', stroke=asdict(best),
                                         practice_seed=args.seed, practice_attempts=len(training['rows'])))

        for name, stroke in [('searched', best), ('fixed_velocity', None)]:
            section = dict(rows=[], stroke=None if stroke is None else asdict(stroke))
            report['sections'][name] = section
            active = player(stroke)
            for seed, phase in held:
                row = play(active, seed, phase, 1)
                section['rows'].append(row)
                print(name, len(section['rows']), phase, row['evaluator']['legal_serve'],
                      row['evaluator']['points_for'], row['evaluator']['reason'], flush=True)
                save_summary(section)

        results = report['sections']
        if args.match and results['searched']['legal'] >= max(4, results['fixed_velocity']['legal'] + 1):
            seeds = _seeds(args.seed, 1718, 0, 30)['tests']
            phases = match_phases(['ordinary', 'soft', 'mixed', 'delay'], len(seeds), [args.seed, 1718])
            for name, stroke in [('searched', best), ('fixed_velocity', None)]:
                section = dict(rows=[], stroke=None if stroke is None else asdict(stroke))
                report['sections'][name + '_match'] = section
                active = player(stroke)
                score = ServiceScore()
                for seed, phase in zip(seeds, phases, strict=True):
                    row = play(active, seed, phase, score.server)
                    section['rows'].append(row)
                    score.record(row['evaluator'])
                    section.update(score=score.score, finished=score.finished, winner=score.winner)
                    save_summary(section)
                    print(name, 'match', len(section['rows']), score.score, row['evaluator']['reason'], flush=True)
                    if score.finished:
                        break
    finally:
        arena.close()
    print(f'{replay(_read(args.output))} action tapes replayed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=219706341)
    parser.add_argument('--generations', type=int, default=7)
    parser.add_argument('--held', type=int, default=10)
    parser.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt6/continual-player.json.gz')
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--match', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'{replay(_read(args.output))} action tapes replayed')
    elif min(args.generations, args.held) < 1:
        parser.error('practice generations and held attempts must be positive')
    else:
        run(args)
