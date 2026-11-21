"""Compare reward-selected timing with fixed timing in paired full table-tennis games."""

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import check_tapes
from .tt9_precontact_registration import RegistrationMemory
from .tt10_series_trial import PHASES, _restore_tracker, _save
from .tt10_window_trial import _point_row
from .tt11_timing_select import MODES
from .tt12_online_timing import OPENING_FRAMES, ROLES, SAMPLES_PER_MODEL, OnlineTimingPlayer, TimingChoice


POINT_LIMIT = 65
RACE_TO = 11
WINDOW = 2 * SAMPLES_PER_MODEL * len(ROLES)
SEED = 1153264317
ENTRY = ('arena', 'model', 'optics', 'calibration', 'probes', 'attempts', 'donor_sha256')


def restore(path, arena, policy, anchor=None):
    state = _read(path)
    entry = state['series']
    library, prior, response, timing = load(path)
    fit = library.models[entry['model']]
    selection = state['timing_selection']
    models = {mode: replace(fit, **{key: np.asarray(values) for key, values in selection['candidates'][mode].items()})
              for mode in MODES}
    choice = TimingChoice(**state['online_timing']) if 'online_timing' in state else TimingChoice(
        anchor or selection['model'])
    player = OnlineTimingPlayer(
        models[choice.anchor], prior, _restore_tracker(arena.cameras, entry['optics']), response, timing,
        predictive=True, flight_memory=FlightMemory('fixed', entry['flight_history'], entry['flight_impulses']),
        drive=ServiceDrive(**state.get('service_drive', {})), learn=False,
        registration=RegistrationMemory(**deepcopy(state['service_registration'])), learn_registration=True,
        models=models, choice=choice, policy=policy)
    prior.rng.bit_generator.state = deepcopy(state['rng'])
    if prior.span_memory is not None:
        prior.span_memory.active = state['motor_span']['active']
        prior.span_memory.exposures = {int(key): n for key, n in state['motor_span']['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in entry['role_rows']]
    return state, library, player


def save(path, state, library, player):
    entry = {key: state['series'][key] for key in ENTRY}
    _save(path, state, library, player, entry, state['series']['completed'] + 1)
    state = _read(path)
    state['online_timing'] = player.choice.state()
    _write(path, state)


def _wins(rows):
    return sum(row['points_for'] for row in rows)


def _with_return(rows):
    return sum(row['player_returns'] > 0 for row in rows)


def metrics(rows):
    opponent = [row for row in rows if row['server'] == -1]
    own = [row for row in rows if row['server'] == 1]
    replies = [row for row in own if row['evaluator']['legal_serve'] and row['rival_returns'] > 0]
    recognized = [row for row in rows if row['timing_choice'] is not None]
    return dict(points=len(rows), points_won=_wins(rows),
                actions=sum(row['steps'] for row in rows), stereo_pairs=sum(row['frames'] for row in rows),
                legal_returns=sum(row['player_returns'] for row in rows),
                sustained=sum(row['summary']['sustained'] for row in rows),
                recognized=len(recognized), misrecognized=sum(
                    row['timing_choice']['role'] != ('serve' if row['server'] == 1 else 'receive') for row in recognized),
                receive=dict(points=len(opponent), wins=_wins(opponent), legal_first_returns=_with_return(opponent),
                             sustained=sum(row['summary']['sustained'] for row in opponent)),
                serve=dict(points=len(own), wins=_wins(own), legal=sum(row['evaluator']['legal_serve'] for row in own),
                           rival_replies=len(replies), legal_followups=_with_return(replies),
                           sustained=sum(row['summary']['sustained'] for row in own)))


def play(arena, wiring, player, seed, schedule, stage):
    score = ServiceScore(RACE_TO, first_server=-1)
    game = dict(seed=seed, stage=stage, policy=player.policy, schedule=schedule, points=[],
                race_to=RACE_TO, first_server=score.first_server, choice_before=player.choice.state(),
                preferences_before={role: player.choice.preferred(role) for role in ROLES})
    for point_seed, phase in schedule:
        row = _point_row(player, arena, wiring, seed, point_seed, phase, score.server)
        game['points'].append(row)
        score.record(row['evaluator'])
        if score.finished:
            break
    if not score.finished:
        raise RuntimeError('game did not finish within the common point budget')
    rows = game['points']
    game.update(score=score.score, finished=True, metrics=metrics(rows),
                opening=metrics(rows[:WINDOW]), late=metrics(rows[WINDOW:]), choice_after=player.choice.state(),
                preferences_after={role: player.choice.preferred(role) for role in ROLES})
    print(stage, player.policy, score.score, game['metrics'], game['preferences_after'], flush=True)
    return game


def audit_choice(game):
    learner = TimingChoice(**deepcopy(game['choice_before']))
    policy = game['policy']
    if game['preferences_before'] != {role: learner.preferred(role) for role in ROLES}:
        raise ValueError('incorrect opening preferences')
    for row in game['points']:
        decision = row['timing_choice']
        if decision is None:
            continue
        observed = 'serve' if decision['depth'] < -1.10 else 'receive' if decision['depth'] > .35 else None
        role = decision['role']
        if observed != role or not 0 <= decision['tick'] < OPENING_FRAMES:
            raise ValueError('decision was made outside its RGB recognition gate')
        if decision['values'] != learner.values(role) or decision['attempts'] != learner.attempts[role]:
            raise ValueError('choice state cannot be reconstructed from own rewards')
        if policy == 'adaptive':
            mode, rule = learner.choose(role)
            learner.observe(role, mode, row['public_reward'])
        elif policy == 'retained':
            mode, rule = learner.preferred(role), 'retained'
        else:
            mode, rule = policy, 'fixed'
        if (mode, rule) != (decision['mode'], decision['rule']):
            raise ValueError('choice was not reproduced')
    if learner.state() != game['choice_after']:
        raise ValueError('postgame timing memory differs')
    if game['preferences_after'] != {role: learner.preferred(role) for role in ROLES}:
        raise ValueError('incorrect retained preferences')


def verify(report):
    matches = {stage: [] for stage in ('adaptation', 'qualification')}
    total = 0
    for game in report['games']:
        matches[game['stage']].append(game)
        audit_choice(game)
        if game['metrics'] != metrics(game['points']):
            raise ValueError('game metrics differ from the point records')
        if [(row['seed'], row['phase']) for row in game['points']] != [tuple(case) for case in
                                                                     game['schedule'][:len(game['points'])]]:
            raise ValueError('game departed from the planned schedule')
        check_tapes(dict(report, **game))
        total += len(game['points'])
    for games in matches.values():
        if len(games) != 3 or any(game['schedule'] != games[0]['schedule'] or
                                  game['choice_before'] != games[0]['choice_before'] for game in games):
            raise ValueError('paired branches must start with the same schedule and timing state')
    left, right = (games[0] for games in matches.values())
    if left['policy'] != 'adaptive' or right['policy'] != 'retained' or left['choice_after'] != right['choice_before']:
        raise ValueError('the learned timing state did not enter the held game')
    schedules = [{seed for seed, _ in game['schedule']} for game in (left, right)]
    if any(len(schedule) != POINT_LIMIT for schedule in schedules) or schedules[0] & schedules[1]:
        raise ValueError('the held schedule is not disjoint')
    return total


def live_check(arena, wiring, checkpoint, game):
    _, _, player = restore(checkpoint, arena, game['policy'])
    score = ServiceScore(game['race_to'], game['first_server'])
    hashes = []
    for expected in game['points'][:4]:
        point = _point_row(player, arena, wiring, game['seed'], expected['seed'], expected['phase'], score.server)
        score.record(point['evaluator'])
        if json.loads(json.dumps(point)) != json.loads(json.dumps(expected)):
            raise ValueError('live RGB continuation diverged after reloading the checkpoint')
        hashes.append(point['action_sha256'])
    return hashes


def run(site, folder, seed, anchor):
    source = ROOT / 'artifacts/tt11' / f'timing-{site}-selected-start.json.gz'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'online-{site}-games.json.gz'
    learned = folder / f'online-{site}-learned.json.gz'
    final = folder / f'online-{site}-player.json.gz'
    arena_args = _read(source)['series']['arena']
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    report = dict(schema='tt12-online-timing-v1', site=site, seed=seed, arena=arena_args,
                  installation_seed=arena_args['seed'], source_sha256=_hash(source),
                  sample_attempts_per_role_per_model=SAMPLES_PER_MODEL, opening_frames=OPENING_FRAMES,
                  race_to=RACE_TO, point_limit=POINT_LIMIT, point_action_limit=arena_args['max_steps'],
                  extra_practice_points=0, games=[], implementation_sha256={name: _hash(Path(__file__).with_name(name))
                      for name in ('tt12_online_timing.py', 'tt12_online_timing_trial.py')})
    try:
        cache_hall_stereo(arena, reflections=True)
        seeds = _seeds(seed, 9387, 0, 2)['tests']
        stages = [('adaptation', source, ('adaptive', *MODES), learned),
                  ('qualification', learned, ('retained', *MODES), final)]
        for game_seed, (stage, start, policies, checkpoint) in zip(seeds, stages, strict=True):
            starts = _seeds(game_seed, 6011, 0, POINT_LIMIT)['tests']
            phases = match_phases(PHASES, POINT_LIMIT, [game_seed, 6011])
            schedule = list(zip(starts, phases, strict=True))
            start_sha256 = _hash(start)
            for index, policy in enumerate(policies):
                state, library, player = restore(start, arena, policy, anchor=anchor)
                game = play(arena, wiring, player, game_seed, schedule, stage)
                game['start_sha256'] = start_sha256
                if index == 0:
                    save(checkpoint, state, library, player)
                    game['checkpoint_sha256'] = _hash(checkpoint)
                    if stage == 'qualification':
                        game['live_resume_action_sha256'] = live_check(arena, wiring, start, game)
                report['games'].append(game)
                _write(output, report)
            if _hash(start) != start_sha256:
                raise RuntimeError('source checkpoint was mutated')
    finally:
        arena.close()
    report['verified_physical_point_tapes'] = verify(_read(output))
    _write(output, report)
    print(f"{report['verified_physical_point_tapes']} physical point tapes verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', choices=('c', 'd'))
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--anchor', choices=MODES)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f"{verify(_read(args.folder / f'online-{args.site}-games.json.gz'))} physical point tapes verified")
    else:
        run(args.site, args.folder, args.seed, args.anchor)


if __name__ == '__main__':
    main()
