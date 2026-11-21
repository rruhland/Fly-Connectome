"""Continue optical timing practice and play disjoint softened six-joint courses."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _read, _write
from .tt6_contact import BASELINE_CONTACT, ContactPlayer
from .tt6_continuous_trial import match_phases, point_condition
from .tt6_memory import load
from .tt6_racket import set_racket_response
from .tt6_rotary_adapt_trial import PHASES, commission
from .tt6_timing import DelayedWiring
from .tt7_contact import TIMING_FAMILIES, ContactTimingMemory, ContactTimingPlayer


DEFAULT_SOURCE = ROOT / 'artifacts/tt6/continual-player.json.gz'
FIXED_CHOICES = {'coast': 0, 'fixed-early': 1, 'fixed': 2, 'fixed-late': 3}
CONTROL_NAMES = ('baseline', 'cold', 'carried', 'frozen', *FIXED_CHOICES)


def _replay(wiring, phases, station_seed, rows):
    for phase, row in zip(phases, rows):
        response, disturbance = point_condition(phase)
        set_racket_response(wiring.arena, response)
        game = DelayedWiring(wiring, disturbance, seed=station_seed)
        game.reset(seed=row['seed'], render=False)
        for rates in row['actions']:
            _, _, done, capped, _ = game.step(np.asarray(rates), render=False)
        if not (done or capped) or game.point_result() != row['evaluator']:
            raise ValueError(f"failed physical replay for {row['seed']}")
        if game.events != row['events'] or game.applied != row['applied']:
            raise ValueError(f"failed command or event replay for {row['seed']}")


def _run_player(wiring, player, seed, schedule, phases, section, output, report, *, race_to):
    rows = section['points']
    modes = Counter()
    wins = losses = 0
    for phase, point_seed in zip(phases, schedule['tests'], strict=True):
        response, disturbance = point_condition(phase)
        set_racket_response(wiring.arena, response)
        disturbed = DelayedWiring(wiring, disturbance, seed=seed)
        row = _point(disturbed, player, point_seed)
        row.update(_counts(row), phase=phase, applied=disturbed.applied)
        rows.append(row)
        modes.update(shot['profile'] for shot in row.get('contact_timing', []))
        wins += row['points_for']
        losses += row['points_against']
        section.update(summary=_metrics(rows), score=[wins, losses], timing_choices=dict(modes))
        _write(output, report)
        print(seed, section['condition'], len(rows), phase, row['player_returns'], row['rival_returns'],
              row['evaluator']['reason'], [shot['profile'] for shot in row.get('contact_timing', [])],
              tuple(section['score']), flush=True)
        if race_to and max(wins, losses) >= race_to and abs(wins - losses) >= 2:
            break
    section['match_finished'] = bool(race_to and max(wins, losses) >= race_to and abs(wins - losses) >= 2)
    _replay(wiring, phases, seed, rows)
    section['replayed_points'] = len(rows)
    _write(output, report)


def _player(condition, fit, prior, factory, motor_timing, contact_memory, family):
    stroke, follow = BASELINE_CONTACT
    if condition == 'baseline':
        return ContactPlayer(fit, deepcopy(prior), factory, stroke=stroke, follow=follow, timing=deepcopy(motor_timing))
    fixed = FIXED_CHOICES.get(condition)
    memory = type(contact_memory)() if condition == 'cold' else deepcopy(contact_memory)
    return ContactTimingPlayer(fit, deepcopy(prior), factory, timing=deepcopy(motor_timing), contact_memory=memory,
                               fixed=fixed, explore=condition != 'frozen', family=family)


def _conditions(phases, count, seed):
    return match_phases(phases, count, [seed, 21491]) if len(phases) > 1 else list(phases) * count


def run(args):
    if set(args.practice_seeds) & set(args.test_seeds):
        raise ValueError('practice and tests must use disjoint installation seeds')
    _, prior, _, motor_timing = load(args.source)
    resumed = {} if args.resume is None else _read(args.resume)
    if resumed.get('family', 'lift') != args.family and resumed:
        raise ValueError('the contact checkpoint must use the selected timing family')
    memory = ContactTimingMemory.from_state(resumed) if resumed else ContactTimingMemory()
    if args.matched_practice is not None:
        from .tt8_contact import PairedTimingMemory, practice_samples

        training = _read(args.matched_practice)
        if set(args.test_seeds) & {site['seed'] for site in training['held']}:
            raise ValueError('matched practice and tests must use disjoint installation seeds')
        memory = PairedTimingMemory(practice_samples(training))
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=args.max_steps)
    report = dict(schema='tt7-stereo-contact-timing-v1', geometry=geometry, phase=args.phase,
                  source_sha256=_hash(args.source), resume_sha256=None if args.resume is None else _hash(args.resume),
                  implementation_sha256={name: _hash(Path(__file__).with_name(name))
                                         for name in ('tt7_contact.py', 'tt7_contact_trial.py', 'tt7_flight.py')},
                  race_to=args.race_to, match_limit=args.match_limit if args.race_to else None,
                  family=args.family,
                  candidate_timings=[None if shape is None else asdict(shape)
                                     for shape in TIMING_FAMILIES[args.family]],
                  practice=[], held=[])
    phases = args.phases or [args.phase]
    if args.matched_practice is not None:
        report['matched_practice_sha256'] = _hash(args.matched_practice)
        report['implementation_sha256']['tt8_contact.py'] = _hash(Path(__file__).with_name('tt8_contact.py'))
    if args.phases is not None:
        report.pop('phase')
        report['phases'] = phases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.practice_seeds:
        arena, wiring, fit, factory, commissioning = commission(seed, args.practice_points, geometry,
                                                                probe_cache=args.probe_cache)
        try:
            section = dict(seed=seed, commissioning=commissioning, condition='practice', points=[])
            report['practice'].append(section)
            player = ContactTimingPlayer(fit, deepcopy(prior), factory, timing=deepcopy(motor_timing),
                                         contact_memory=memory, family=args.family)
            _run_player(wiring, player, seed, commissioning['schedule'], _conditions(phases, args.practice_points, seed),
                        section, args.output, report, race_to=0)
            section['contact_memory'] = memory.state()
        finally:
            arena.close()
    report['contact_memory'] = memory.state()
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write(args.checkpoint, dict(family=args.family, **memory.state()))
    _write(args.output, report)
    count = args.match_limit if args.race_to else args.test_points
    for seed in args.test_seeds:
        arena, wiring, fit, factory, commissioning = commission(seed, count, geometry, probe_cache=args.probe_cache)
        try:
            installation = dict(seed=seed, commissioning=commissioning, conditions={})
            report['held'].append(installation)
            for condition in args.conditions:
                section = dict(condition=condition, points=[])
                installation['conditions'][condition] = section
                player = _player(condition, fit, prior, factory, motor_timing, memory, args.family)
                _run_player(wiring, player, seed, commissioning['schedule'], _conditions(phases, count, seed), section,
                            args.output, report, race_to=args.race_to)
                if isinstance(player, ContactTimingPlayer):
                    section['contact_memory'] = player.contact_memory.state()
                _write(args.output, report)
        finally:
            arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    cli.add_argument('--resume', type=Path)
    cli.add_argument('--matched-practice', type=Path, help='learn from a replayed early/aligned/late lift comparison')
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--practice-seeds', type=int, nargs='*', default=[73301393, 90041121])
    cli.add_argument('--test-seeds', type=int, nargs='*', default=[55217409, 15976437])
    cli.add_argument('--practice-points', type=int, default=12)
    cli.add_argument('--test-points', type=int, default=10)
    cli.add_argument('--family', choices=tuple(TIMING_FAMILIES), default='lift')
    conditions = cli.add_mutually_exclusive_group()
    conditions.add_argument('--phase', choices=PHASES, default='soft')
    conditions.add_argument('--phases', choices=PHASES, nargs='+', help='balance conditions in shuffled blocks')
    cli.add_argument('--probe-cache', type=Path, help='reuse public calibration RGB in this directory')
    cli.add_argument('--conditions', nargs='+', choices=CONTROL_NAMES, default=['baseline', 'coast', 'cold', 'carried'])
    cli.add_argument('--race-to', type=int, default=0)
    cli.add_argument('--match-limit', type=int, default=70)
    cli.add_argument('--max-steps', type=int, default=280)
    args = cli.parse_args()
    if min(args.practice_points, args.test_points, args.match_limit, args.max_steps) < 1 or args.race_to < 0:
        cli.error('point counts and limits must be positive and race target must be nonnegative')
    if args.matched_practice is not None and (args.family != 'lift' or args.resume is not None
                                               or args.practice_seeds or args.checkpoint is not None):
        cli.error('use matched lift practice with an empty --practice-seeds and without --resume or --checkpoint')
    run(args)
