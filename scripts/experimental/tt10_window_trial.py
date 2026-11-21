"""Compare RGB-calibrated reply contact windows on matched histories and full games."""
import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import check_tapes
from .tt9_precontact_match import save_player
from .tt9_precontact_registration import RegistrationMemory, RegistrationPlayer
from .tt9_precontact_trial import _play_point
from .tt9_recovery_trial import point_summary
from .tt10_window import WindowPlayer


DEFAULT_SOURCE = ROOT / 'artifacts/tt9/precontact-match-b-player.json.gz'
PLANNERS = {'fixed': RegistrationPlayer, 'window': WindowPlayer}
CARRIED = ('prior', 'response', 'timing_memory', 'memory', 'approach', 'drive', 'role_rows')


def _player(kind, fit, factory, prior, response, timing, state):
    player = PLANNERS[kind]
    return player(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                  predictive=True, flight_memory=FlightMemory('fixed'),
                  drive=ServiceDrive(**state.get('service_drive', {})), learn=False,
                  registration=RegistrationMemory(**deepcopy(state['service_registration'])),
                  learn_registration=True)


def _point_row(player, arena, wiring, run_seed, seed, phase, server):
    row = _play_point(player, arena, wiring, run_seed, seed, phase, server)
    row.update(summary=point_summary(row))
    if server == 1 and row['evaluator']['rival_returns'] > 0:
        row['optical_trace'] = player.trace
    return row


def _summary(rows):
    own = [row for row in rows if row['server'] == 1]
    replies = [row for row in own if row['evaluator']['legal_serve'] and row['rival_returns'] > 0]
    return dict(points=len(rows), legal_own_serves=sum(row['evaluator']['legal_serve'] for row in own),
                own_serves=len(own), replies=len(replies), hit_replies=sum(
                    row['summary']['first_hit'] is not None for row in replies),
                legal_followups=sum(row['player_returns'] > 0 for row in replies),
                later_reply_returns=sum(max(0, row['player_returns'] - 1) for row in replies),
                replied_points_won=sum(row['points_for'] for row in replies),
                total_legal_returns=sum(row['player_returns'] for row in rows),
                points_won=sum(row['points_for'] for row in rows),
                window_points=sum(bool(row.get('reply_window', {}).get('choices')) for row in rows),
                referee_reply_points_using_window=sum(
                    bool(row.get('reply_window', {}).get('choices')) for row in replies))


def _run_pairs(args, report, arena, wiring, players, previous):
    baseline = players['fixed']
    candidate = players['window']
    original = previous['sections']['full_match']['rows']
    sections = report['sections']
    for row in original:
        paired = row['server'] == 1 and row['evaluator']['legal_serve'] and row['rival_returns'] > 0
        if paired:
            for field in CARRIED:
                setattr(candidate, field, deepcopy(getattr(baseline, field)))
        fixed = _point_row(baseline, arena, wiring, args.seed, row['seed'], row['phase'], row['server'])
        if fixed['action_sha256'] != row['action_sha256'] or fixed['evaluator'] != row['evaluator']:
            raise ValueError(f"historical baseline diverged on point {row['seed']}")
        report['historical_points_matched'] += 1
        if paired:
            window = _point_row(candidate, arena, wiring, args.seed, row['seed'], row['phase'], row['server'])
            opponent_step = next(event['step'] for event in fixed['events']
                                 if event['kind'] == 'contact' and event['side'] == -1)
            if fixed['actions'][:opponent_step] != window['actions'][:opponent_step]:
                raise ValueError(f"the controllers differed before the opponent struck on point {row['seed']}")
            if window['rival_returns'] < 1:
                raise ValueError(f"the same rival contact failed to yield a legal reply on point {row['seed']}")
            window['identical_through_rival_contact'] = opponent_step
            sections['fixed']['points'].append(fixed)
            sections['window']['points'].append(window)
            print(row['seed'], fixed['summary'], window['summary'],
                  window['reply_window']['choices'], flush=True)
        for section in sections.values():
            section['metrics'] = _summary(section['points'])
        _write(args.output, report)


def _run_games(args, report, arena, wiring, players, library, state):
    for kind, player in players.items():
        score = ServiceScore(first_server=args.first_server)
        section = report['sections'][kind]
        section.update(race_to=11, first_server=args.first_server)
        for seed, phase in report['point_schedule']:
            row = _point_row(player, arena, wiring, args.seed, seed, phase, score.server)
            section['points'].append(row)
            score.record(row['evaluator'])
            section.update(score=score.score, finished=score.finished, metrics=_summary(section['points']))
            _write(args.output, report)
            print(kind, seed, phase, score.score, row['summary'],
                  bool(row.get('reply_window', {}).get('choices')), flush=True)
            if score.finished:
                break
        if not score.finished:
            raise RuntimeError('game exhausted the configured point limit')
        if kind == 'window' and args.checkpoint is not None:
            save_player(args.checkpoint, state, library, player)
            saved = _read(args.checkpoint)
            saved['reply_window'] = dict(version=1)
            _write(args.checkpoint, saved)
            load(args.checkpoint)
            report['checkpoint_sha256'] = _hash(args.checkpoint)
            _write(args.output, report)


def verify(report):
    count = 0
    for section in report['sections'].values():
        if section['metrics'] != _summary(section['points']):
            raise ValueError('inconsistent reply metrics')
        check_tapes(dict(report, **section))
        count += len(section['points'])
    if report['paired']:
        fixed, window = (report['sections'][mode]['points'] for mode in PLANNERS)
        for left, right in zip(fixed, window, strict=True):
            stop = right['identical_through_rival_contact']
            if left['seed'] != right['seed'] or left['actions'][:stop] != right['actions'][:stop]:
                raise ValueError('paired point history diverged before the rival contact')
    return count


def run(args):
    library, prior, response, timing = load(args.resume)
    state = _read(args.resume)
    previous = _read(args.paired) if args.paired else None
    calibration = _seeds(args.seed, 0, 8, 0)
    if previous:
        if previous['source_sha256'] != _hash(args.resume) or previous['seed'] != args.seed:
            raise ValueError('paired evaluation needs the exact original source checkpoint and seed')
        arena_args = previous['arena']
        point_schedule = previous['schedule']
    else:
        install = _seeds(args.installation_seed if args.installation_seed is not None else args.seed + 109, 0, 1, 1)
        arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=args.body, venue='hall',
                          view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                          max_steps=420, frameskip=8, distractor=False)
        starts = _seeds(args.seed, 4931, 0, args.limit)['tests']
        phases = match_phases(args.phases, args.limit, [args.seed, 4931])
        point_schedule = list(zip(starts, phases, strict=True))
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    report = dict(schema='tt10-reply-window-v1', seed=args.seed, paired=bool(previous), arena=arena_args,
                  installation_seed=arena_args['seed'], source_sha256=_hash(args.resume), calibration=calibration,
                  point_schedule=point_schedule, original_sha256=_hash(args.paired) if previous else None,
                  historical_points_matched=0, sections={kind: dict(points=[]) for kind in PLANNERS},
                  implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                         ('tt10_window.py', 'tt10_window_trial.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        if previous and (probes != previous['probes'] or attempts != previous['attempts']):
            raise ValueError('body commissioning differs from the original game')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())
        players = {kind: _player(kind, fit, factory, prior, response, timing, state) for kind in PLANNERS}
        if previous:
            _run_pairs(args, report, arena, wiring, players, previous)
        else:
            _run_games(args, report, arena, wiring, players, library, state)
    finally:
        arena.close()
    print(f'{verify(_read(args.output))} action tapes replayed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--body', choices=('offset', 'gantry'), default='offset')
    parser.add_argument('--installation-seed', type=int)
    parser.add_argument('--resume', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--paired', type=Path)
    parser.add_argument('--phases', nargs='+', default=['ordinary', 'mixed', 'soft', 'delay'])
    parser.add_argument('--first-server', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--limit', type=int, default=65)
    parser.add_argument('--verify', action='store_true')
    arguments = parser.parse_args()
    if arguments.verify:
        print(f'{verify(_read(arguments.output))} action tapes replayed')
    elif arguments.seed is None or arguments.limit < 1:
        parser.error('supply a seed and a positive point limit')
    else:
        run(arguments)
