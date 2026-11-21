"""Run matched serve-placement practice and qualify on disjoint throws or games."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt10_strategy import SWEEPS, StrategyMemory, StrategyPlayer, optical_exchange
from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import check_tapes
from .tt9_precontact_match import save_player
from .tt9_precontact_registration import RegistrationMemory
from .tt9_precontact_registration_trial import _cases
from .tt9_precontact_trial import _play_point


PHASES = ('ordinary', 'mixed', 'soft', 'delay')


def summarize(rows):
    stats = [row['evaluator'] for row in rows if row['server'] == 1]
    replies = [result for result in stats if result['legal_serve'] and result['rival_returns'] > 0]
    shots = [row['serve_strategy'] for row in rows if row['serve_strategy'] is not None]
    slopes = [shot['outbound_slope'] for shot in shots if shot['outbound_slope'] is not None]
    return dict(served=len(stats), legal=sum(row['legal_serve'] for row in stats),
                contacts=sum(row['service_contact'] for row in stats), replies=len(replies),
                follow_ups=sum(row['legal_returns'] > 0 for row in replies),
                serve_wins=sum(row['points_for'] for row in stats),
                wins_after_reply=sum(row['points_for'] for row in replies),
                own_later_returns=sum(max(0, row['legal_returns'] - 1) for row in replies),
                total_returns=sum(row['player_returns'] for row in rows),
                optical_slopes=len(slopes), median_slope=None if not slopes else float(np.median(slopes)),
                optical_replies=sum(shot['reply'] is not None for shot in shots),
                optical_follow_ups=sum(shot['follow_up'] for shot in shots),
                choice_counts=[sum(shot['choice'] == choice for shot in shots)
                               for choice in range(len(SWEEPS))])


def append(report, section, row, output):
    section['rows'].append(row)
    section['summary'] = summarize(section['rows'])
    _write(output, report)
    outcome = row['evaluator']
    shot = row['serve_strategy']
    print(section['name'], len(section['rows']), row['server'], row['phase'],
          None if shot is None else shot['choice'], outcome['legal_serve'], outcome['rival_returns'],
          outcome['legal_returns'], row['public_reward'], section.get('score'), flush=True)


def verify(report):
    for section in report['sections'].values():
        settings = dict(report, points=section['rows'], practice='score' not in section)
        if 'score' in section:
            settings.update(race_to=11, first_server=-1, score=section['score'], finished=section['finished'])
        check_tapes(settings)
        if summarize(section['rows']) != section['summary']:
            raise ValueError(f"incorrect summary in {section['name']}")
        for row in section['rows']:
            if 'serve_optical_trace' in row:
                estimate = optical_exchange(row['serve_optical_trace'], row['serve_drive']['tick'])
                if any(row['serve_strategy'][key] != value for key, value in estimate.items()):
                    raise ValueError(f"optical evidence diverged on point {row['seed']}")
    return sum(len(section['rows']) for section in report['sections'].values())


def run(args):
    library, prior, response, timing = load(args.resume)
    previous = _read(args.resume)
    registration = previous['service_registration']
    installation = _seeds(args.installation_seed, 0, 1, 1)
    calibration = _seeds(args.calibration_seed, 0, 8, 0)
    arena_args = dict(seed=installation['arena_seed'], view_seed=installation['view_seed'],
                      body='offset', venue='hall', view='first_person', render_mode='mujoco',
                      width=192, height=144, baseline=.28, max_steps=420, frameskip=8, distractor=False)
    report = dict(schema='tt10-serve-strategy-1', seed=args.seed, arena=arena_args,
                  installation_seed=installation['arena_seed'], calibration_seed=args.calibration_seed,
                  placement_rates=SWEEPS, source_sha256=_hash(args.resume),
                  implementation_sha256={name: _hash(Path(__file__).with_name(name))
                                         for name in ('tt10_strategy.py', 'tt10_strategy_trial.py')},
                  sections={})
    practice = _cases(args.seed, 1701, args.practice, PHASES)
    held = _cases(args.seed, 1702, args.held, PHASES)
    game = _cases(args.seed, 1703, args.game, PHASES)
    report['schedules'] = dict(practice=practice, held=held, game=game)
    if len({seed for seed, _ in practice + held + game}) != len(practice) + len(held) + len(game):
        raise ValueError('point schedules must be disjoint')
    if args.training:
        training = _read(args.training)
        if training['arena'] != arena_args or training['source_sha256'] != report['source_sha256']:
            raise ValueError('placement practice and test must use the same installation and starting checkpoint')
        practiced = {row['seed'] for section in training['sections'].values() for row in section['rows']}
        if practiced.intersection(seed for seed, _ in held + game):
            raise ValueError('held and game starts must be disjoint from supplied practice')
        report['training_sha256'] = _hash(args.training)
        memory = StrategyMemory(**training['memory_after_practice'])
    else:
        memory = StrategyMemory(**previous.get('service_strategy', {}))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, installation['arena_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())

        def player(placement, *, practice=False, live=False):
            return StrategyPlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                                  predictive=True, flight_memory=FlightMemory('fixed'), drive=ServiceDrive(),
                                  learn=False, registration=RegistrationMemory(**deepcopy(registration)),
                                  learn_registration=live, strategy=memory if practice else deepcopy(memory),
                                  placement=placement, learn_strategy=practice)

        def section(name):
            return report['sections'].setdefault(name, dict(name=name, rows=[]))

        def play(active, seed, phase, server=1):
            row = _play_point(active, arena, wiring, args.seed, seed, phase, server)
            if row['serve_strategy'] is not None:
                row['serve_optical_trace'] = [{key: frame[key] for key in ('tick', 'ball')}
                                              for frame in active.trace if frame['ball'] is not None]
            return row

        rng = np.random.default_rng(args.seed)
        for seed, phase in practice:
            for choice in rng.permutation(len(SWEEPS)):
                choice = int(choice)
                active = player(choice, practice=True)
                row = play(active, seed, phase)
                append(report, section(f'practice-{choice}'), row, args.output)
        best_fixed = int(np.argmax(memory.values([1., 0., 0.], contextual=False)))
        report.update(memory_after_practice=memory.state(), best_fixed=best_fixed)
        _write(args.output, report)

        policies = [('learned', None), ('registered', 0)]
        if best_fixed != 0:
            policies.append(('best-fixed-sweep', best_fixed))
        for mode, choice in policies:
            for seed, phase in held:
                active = player(choice)
                row = play(active, seed, phase)
                append(report, section(f'held-{mode}'), row, args.output)
            if game:
                active = player(choice, live=True)
                match = section(f'game-{mode}')
                score = ServiceScore(first_server=-1)
                for seed, phase in game:
                    row = play(active, seed, phase, score.server)
                    score.record(row['evaluator'])
                    match.update(score=list(score.score), finished=score.finished)
                    append(report, match, row, args.output)
                    if score.finished:
                        break
                if mode == 'learned' and args.checkpoint:
                    save_player(args.checkpoint, dict(previous, service_strategy=memory.state()), library, active)
                    load(args.checkpoint)
                    report['checkpoint_sha256'] = _hash(args.checkpoint)
                    _write(args.output, report)
    finally:
        arena.close()
    print(f'{verify(_read(args.output))} point tapes replayed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--installation-seed', type=int)
    parser.add_argument('--calibration-seed', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--practice', type=int, default=0)
    parser.add_argument('--held', type=int, default=0)
    parser.add_argument('--game', type=int, default=0, help='point limit for each first-to-11 game')
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'{verify(_read(args.output))} point tapes replayed')
    elif None in (args.seed, args.installation_seed, args.calibration_seed, args.resume):
        parser.error('supply course, installation and calibration seeds plus the starting checkpoint')
    elif min(args.practice, args.held, args.game) < 0 or not (args.practice or args.held or args.game):
        parser.error('supply at least one positive point count')
    elif args.training and args.practice:
        parser.error('use fresh practice or a saved practice receipt')
    elif args.checkpoint and not args.game:
        parser.error('checkpoint requires a game')
    else:
        run(args)
