"""Test a frozen visual timing choice on different physical mechanisms and service roles."""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_arena import TIMESTEP
from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_probe import collect_probe, probe_commands
from .tt6_continuous import prediction_error, supports
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load, save
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_player import ServiceMemory
from .tt8_service_search import ServiceStroke
from .tt8_service_trial import PointInterface, check_tapes
from .tt10_series_trial import _capture_tracker, _restore_tracker
from .tt10_target import TargetArena
from .tt10_target_dynamics import errors, fit_kernels, observe as observe_cursor
from .tt11_timing_select import MODES, PROBE_COUNT, SHOT_FRAMES, observe, select
from .tt12_body_player import ClassicServePlayer, DriveServePlayer


DONOR = ROOT / 'artifacts/tt8/span-after-four.json.gz'
PHASES = ('ordinary', 'soft', 'delay', 'mixed')
SEEDS = {
    'gantry': dict(installation=247179439, calibration=660149961, cursor=77108313, dry=132907519,
                   practice=183719121, fixtures=(1465108873, 720214369)),
    'rotary': dict(installation=1342861537, calibration=1115604919, cursor=75627677, dry=376281881,
                   practice=58182331, fixtures=(1324842371, 163748753)),
}
BRANCHES = (*MODES, 'selected')


def path(folder, body, label):
    return folder / f'body-{body}-{label}.json.gz'


def prepare(folder, body, probe_cache):
    seeds = SEEDS[body]
    install = _seeds(seeds['installation'], 0, 1, 1)
    arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=body, venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    library, prior, response, timing = load(DONOR)
    state = _read(DONOR)
    arena = TT8ServiceArena(**arena_args)
    calibration = _seeds(seeds['calibration'], 0, 8, 0)
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, arena_args['seed'])
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=probe_cache)
        if fit is None:
            raise ValueError(f'{body} calibration yielded no viable motor map: {attempts}')
        model = next(index for index, candidate in enumerate(library.models) if candidate is fit)
        optics = _capture_tracker(factory)
    finally:
        arena.close()
    target = TargetArena(**dict(arena_args, render_mode='projected'))
    cursor_seeds = _seeds(seeds['cursor'], 0, 0, 2)['tests']
    try:
        sessions = [observe_cursor(target, arena_args['seed'], fit, seed) for seed in cursor_seeds]
    finally:
        target.close()
    commands, positions, scores = map(np.asarray, zip(*sessions, strict=True))
    cursor = fit_kernels(fit, commands, positions)
    models = dict(inherited=fit, cursor=cursor)
    dry_seeds = _seeds(seeds['dry'], 0, 0, PROBE_COUNT)['tests']
    rows = observe(arena_args, optics, fit, dry_seeds)
    dt = TIMESTEP * arena_args['frameskip']
    choice = select(models, rows, dt)
    timing_arrays = {name: dict(lags=m.lags.tolist(), kernels=m.kernels.tolist()) for name, m in models.items()}
    start = path(folder, body, 'start')
    save(start, library, prior, response, timing, service_memory=ServiceMemory(state.get('service_memory')))
    state = _read(start)
    state['cross_body_timing'] = dict(arena=arena_args, model=model, optics=optics, timing=timing_arrays,
                                      service_belief=timing.state(), choice=choice, fixtures=seeds['fixtures'],
                                      stage='prefixture')
    _write(start, state)
    report = dict(schema='tt12-cross-body-probes-v1', arena=arena_args, seeds=seeds, calibration=calibration,
                  wiring=wiring.state(), calibration_probes=probes, calibration_attempts=attempts, fit=fit.stats(),
                  donor_sha256=_hash(DONOR), start_sha256=_hash(start), dry_probes=rows, control_dt=dt,
                  choice=choice, timing=timing_arrays,
                  cursor=dict(seeds=cursor_seeds, commands=commands.tolist(), xyz=positions.tolist(),
                              scores=scores.tolist(), errors={name: errors(m, commands, positions)
                                                              for name, m in models.items()}))
    _write(path(folder, body, 'probes'), report)
    print(body, 'dry choice', choice['selected'], choice['error_m'], choice['median_peak_forward_mps'],
          choice['visible_pairs'], flush=True)


def restore(start, arena, receive, profile, *, prior_path=None):
    cfg = _read(start)['cross_body_timing']
    model_library, _, _, _ = load(start)
    fit = model_library.models[cfg['model']]
    saved = prior_path or start
    state = _read(saved)
    library, prior, response, timing = load(saved)
    if prior_path is not None:
        for index, candidate in enumerate(library.models):
            if np.array_equal(candidate.coefficients, fit.coefficients) and np.array_equal(candidate.kernels, fit.kernels):
                break
        else:
            index = len(library.models)
            library.models.append(fit)
        cfg['model'] = index
    name = cfg['choice']['selected'] if receive == 'selected' else receive
    receiving = replace(fit, **{key: np.asarray(value) for key, value in cfg['timing'][name].items()})
    factory = _restore_tracker(arena.cameras, cfg['optics'])
    flight = FlightMemory('fixed', state.get('cross_body_flight_history', ()), state.get('cross_body_flight_impulses', ()))
    common = dict(receiving_model=receiving, service_belief=cfg['service_belief'], predictive=True, flight_memory=flight)
    models = (fit, prior, factory, response, timing)
    if profile.startswith('classic-'):
        player = ClassicServePlayer(*models, **common, fixed=int(profile.partition('-')[2]),
                                    service_memory=ServiceMemory(state.get('service_memory')))
    else:
        player = DriveServePlayer(*models, **common, stroke=ServiceStroke())
    if prior_path is not None:
        prior.rng.bit_generator.state = deepcopy(state['rng'])
        player.role_rows = [np.asarray(row) for row in state.get('cross_body_role_rows', ())]
    return library, player, cfg


def play_point(player, arena, wiring, run_seed, seed, phase, server):
    disturbance = prepare_point(arena, phase)
    interface = PointInterface(wiring, disturbance, server=server, seed=run_seed)
    row = _point(interface, player, seed)
    row.update(_counts(row), server=server, phase=phase, applied=interface.applied)
    return row


def summary(rows):
    points = len(rows)
    good_serves = sum(row['evaluator']['legal_serve'] for row in rows)
    first_ops = first_hits = later_ops = later_hits = sustained = 0
    for row in rows:
        own_serve = row['server'] == 1
        first = row['evaluator']['legal_serve'] and (not own_serve or row['rival_returns'] > 0)
        returns = row['player_returns']
        first_ops += int(first)
        first_hits += int(first and returns > 0)
        later_ops += max(0, row['rival_returns'] - int(own_serve))
        later_hits += max(0, returns - 1)
        sustained += returns >= (1 if own_serve else 2)
    return dict(starts=points, legal_serves=good_serves, first_opportunities=first_ops,
                first_legal_returns=first_hits, later_opportunities=later_ops, later_legal_returns=later_hits,
                sustained=sustained, player_returns=sum(row['player_returns'] for row in rows),
                wins=sum(row['points_for'] for row in rows), losses=sum(row['points_against'] for row in rows),
                lets=sum(row['evaluator']['serve_let'] for row in rows))


def practice(folder, body, profiles, limit):
    start = path(folder, body, 'start')
    cfg = _read(start)['cross_body_timing']
    run_seed = SEEDS[body]['practice']
    starts = _seeds(run_seed, 3919, 0, limit)['tests']
    phases = match_phases(PHASES, limit, [run_seed, 3919])
    arena = TT8ServiceArena(**cfg['arena'])
    wiring = TT6Wiring(arena, cfg['arena']['seed'])
    report = dict(schema='tt12-cross-body-serve-practice-v1', seed=run_seed, arena=cfg['arena'],
                  installation_seed=cfg['arena']['seed'], practice=True, start_sha256=_hash(start),
                  schedule=list(zip(starts, phases, strict=True)), profiles={}, profile_order=profiles)
    output = path(folder, body, 'practice')
    try:
        cache_hall_stereo(arena, reflections=True)
        for profile in profiles:
            _, player, _ = restore(start, arena, 'inherited', profile)
            rows = [play_point(player, arena, wiring, run_seed, seed, phase, 1)
                    for seed, phase in report['schedule']]
            report['profiles'][profile] = dict(points=rows, metrics=summary(rows))
            _write(output, report)
            print(body, profile, report['profiles'][profile]['metrics'], flush=True)
    finally:
        arena.close()
    report['chosen'] = max(profiles, key=lambda label: report['profiles'][label]['metrics']['legal_serves'])
    _write(output, report)
    report['replayed_points'] = verify_practice(_read(output))
    _write(output, report)


def verify_practice(report):
    count = 0
    for section in report['profiles'].values():
        rows = section['points']
        if section['metrics'] != summary(rows) or [[r['seed'], r['phase']] for r in rows] != report['schedule']:
            raise ValueError('serve practice metrics or schedule diverged')
        check_tapes(dict(report, points=rows))
        count += len(rows)
    return count


def save_player(output, library, player, cfg, state, branch, profile, seed):
    save(output, library, player.prior, player.response, player.timing_memory,
         service_memory=ServiceMemory(state.get('service_memory')))
    updated = _read(output)
    updated['cross_body_timing'] = dict(cfg, stage='postgame', branch=branch, profile=profile, last_seed=seed)
    updated['cross_body_flight_history'] = list(player.memory.history)
    updated['cross_body_flight_impulses'] = list(player.memory.impulses)
    updated['cross_body_role_rows'] = np.asarray(player.role_rows).tolist()
    _write(output, updated)
    load(output)


def verify_body_entry(player, arena, wiring, seed):
    """Check an archived body map against unused interventions at the revisited mechanism."""
    cases = _seeds(seed, 9067, 2, 0)
    prepare_point(arena, 'ordinary')
    observations = []
    for index, start in enumerate(cases['calibration']):
        commands = probe_commands(cases['player_seed'] + index, 34, arena.action_size)
        history = collect_probe(wiring, start, commands)
        evidence = prediction_error(player.adapter, history, player.tracker_factory)
        observations.append(dict(seed=start, actions=[action.tolist() for action, _ in history[1:]], evidence=evidence,
                                 supported=supports(player.adapter, evidence)))
    return observations


def play_game(folder, body, branch, fixture, limit, *, prior_path=None, run_seed=None, label=None):
    start = path(folder, body, 'start')
    practice_path = path(folder, body, 'practice')
    cfg = _read(start)['cross_body_timing']
    profile = _read(practice_path)['chosen']
    seed = run_seed if run_seed is not None else cfg['fixtures'][fixture]
    first_server = (-1, 1)[fixture]
    score = ServiceScore(first_server=first_server)
    # Separate cohorts ensure the nth own toss and nth rival opening remain
    # comparable after the score, service rotation and number of lets diverge.
    schedule = {}
    for index, side in enumerate((-1, 1)):
        starts = _seeds(seed, 6011 + index, 0, limit)['tests']
        phases = match_phases(PHASES, limit, [seed, 6011 + index])
        schedule[str(side)] = list(zip(starts, phases, strict=True))
    arena = TT8ServiceArena(**cfg['arena'])
    wiring = TT6Wiring(arena, cfg['arena']['seed'])
    report = dict(schema='tt12-cross-body-timing-game-v1', arena=cfg['arena'], seed=seed,
                  installation_seed=cfg['arena']['seed'], profile=profile, branch=branch, fixture=fixture,
                  start_sha256=_hash(start), practice_sha256=_hash(practice_path),
                  prior_sha256=_hash(prior_path) if prior_path else _hash(start), schedule=schedule,
                  timing=cfg['choice']['selected'] if branch == 'selected' else branch,
                  race_to=11, first_server=first_server, points=[], finished=False)
    name = label or f'{branch}-{fixture + 1}'
    output = path(folder, body, name)
    checkpoint = path(folder, body, f'{name}-player')
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player, cfg = restore(start, arena, branch, profile, prior_path=prior_path)
        if prior_path is not None:
            report['entry_observations'] = verify_body_entry(player, arena, wiring, seed)
        used = {'1': 0, '-1': 0}
        for _ in range(limit):
            side = str(score.server)
            point_seed, phase = schedule[side][used[side]]
            row = play_point(player, arena, wiring, seed, point_seed, phase, score.server)
            report['points'].append(row)
            score.record(row['evaluator'])
            used[side] += 1
            report.update(score=score.score, finished=score.finished)
            _write(output, report)
            if score.finished:
                break
        if not score.finished:
            raise RuntimeError(f'{body} {name} exhausted its schedule without a completed game')
        if branch == 'selected':
            save_player(checkpoint, library, player, cfg, _read(prior_path or start), branch, profile, seed)
            report['checkpoint_sha256'] = _hash(checkpoint)
    finally:
        arena.close()
    report['metrics'] = {side: summary([r for r in report['points'] if r['server'] == int(side)]) for side in schedule}
    report['commands'] = sum(row['steps'] for row in report['points'])
    report['stereo_pairs'] = sum(row['frames'] for row in report['points'])
    _write(output, report)
    report['replayed_points'] = verify_game(_read(output))
    _write(output, report)
    print(body, name, report['score'], report['metrics'], flush=True)
    return checkpoint


def verify_game(report):
    if not report['finished']:
        raise ValueError('only full games qualify')
    for row in report['points']:
        digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
        if digest != row['action_sha256'] or row['steps'] != len(row['actions']):
            raise ValueError(f"invalid requested-command tape for point {row['seed']}")
    for side, cases in report['schedule'].items():
        rows = [row for row in report['points'] if row['server'] == int(side)]
        if [[r['seed'], r['phase']] for r in rows] != cases[:len(rows)]:
            raise ValueError('a side did not use its scheduled openings')
        if report['metrics'][side] != summary(rows):
            raise ValueError('stored game metrics differ from recorded points')
    check_tapes(report)
    return len(report['points'])


def own_service_prefix(row):
    events = [e for e in row['events'] if e['kind'] in ('serve_strike', 'legal_serve', 'let')]
    strikes = [e['step'] for e in events if e['kind'] == 'serve_strike']
    stop = strikes[0] if strikes else row['steps']
    return row['actions'][:stop], row['applied'][:stop], events


def check_decision(replayed, reported):
    for key in ('selected', 'supported', 'visible_pairs', 'median_peak_forward_mps'):
        if replayed[key] != reported[key]:
            raise ValueError(f'dry-stroke replay changed the timing decision field {key}')
    # Array layout of a camera fit changes after the JSON checkpoint roundtrip.
    # BLAS can accumulate in a different order with the same coefficients.
    actual = np.asarray([replayed['error_m'][mode] for mode in MODES], dtype=float)
    expected = np.asarray([reported['error_m'][mode] for mode in MODES], dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
    for observed, stored in zip(replayed['scores'], reported['scores'], strict=True):
        if any(observed[key] != stored[key] for key in ('seed', 'pairs', 'peak_forward_mps')):
            raise ValueError('dry-stroke replay changed single-stroke observations')
        for mode in observed['errors']:
            np.testing.assert_allclose(list(observed['errors'][mode].values()),
                                       list(stored['errors'][mode].values()), rtol=0., atol=1e-12)


def audit(folder, body):
    start = path(folder, body, 'start')
    probes = _read(path(folder, body, 'probes'))
    cfg = _read(start)['cross_body_timing']
    library, _, _, _ = load(start)
    fit = library.models[cfg['model']]
    dry = observe(cfg['arena'], cfg['optics'], fit, [row['seed'] for row in probes['dry_probes']])
    if dry != probes['dry_probes'] or _hash(start) != probes['start_sha256']:
        raise ValueError('dry-stroke replay or checkpoint provenance diverged')
    models = {name: replace(fit, **{k: np.asarray(v) for k, v in params.items()})
              for name, params in cfg['timing'].items()}
    check_decision(select(models, dry, probes['control_dt']), probes['choice'])
    practice = _read(path(folder, body, 'practice'))
    if practice['start_sha256'] != probes['start_sha256']:
        raise ValueError('service practice used a different motor map')
    practice_count = verify_practice(practice)
    sensitivity = dict(
        leave_one_out=[select(models, dry[:i] + dry[i + 1:], probes['control_dt'])['selected']
                       for i in range(len(dry))],
        by_pace={str(pace): select(models, [row for row in dry if row['stroke_frames'] == pace],
                                   probes['control_dt'])['selected'] for pace in SHOT_FRAMES},
    )
    audited = dict(body=body, choice=cfg['choice'], sensitivity=sensitivity,
                   practice_replayed=practice_count, dry_replayed=len(dry), fixtures=[])
    seen = set(probes['calibration']['calibration']) | set(probes['cursor']['seeds']) | {r['seed'] for r in dry}
    seen.update(seed for seed, _ in practice['schedule'])
    for fixture in range(len(cfg['fixtures'])):
        records = {name: _read(path(folder, body, f'{name}-{fixture + 1}')) for name in BRANCHES}
        if any(record['start_sha256'] != probes['start_sha256'] or record['prior_sha256'] != probes['start_sha256']
               for record in records.values()):
            raise ValueError('comparisons used different initial player states')
        inherited, cursor, selected = (records[name] for name in BRANCHES)
        chosen = records[cfg['choice']['selected']]
        if selected['points'] != chosen['points']:
            raise ValueError('selected game differed from its fixed control')
        if any(row['schedule'] != selected['schedule'] for row in records.values()):
            raise ValueError('the three branches played different schedules')
        scheduled = [seed for cases in selected['schedule'].values() for seed, _ in cases]
        if set(scheduled) & seen or len(scheduled) != len(set(scheduled)):
            raise ValueError('shared schedules intersect practice or another fixture')
        seen.update(scheduled)
        own = [{r['seed']: r for r in rec['points'] if r['server'] == 1} for rec in (inherited, cursor)]
        paired = own[0].keys() & own[1].keys()
        identical = sum(own_service_prefix(own[0][s]) == own_service_prefix(own[1][s]) for s in paired)
        if identical != len(paired):
            different = [seed for seed in sorted(paired) if own_service_prefix(own[0][seed]) != own_service_prefix(own[1][seed])]
            raise ValueError(f'fixed own-service strike or legality diverged for matched tosses: {different}')
        fixture_report = dict(own_tosses_matched=identical, branches={})
        for name, record in records.items():
            count = verify_game(record)
            route_metrics = {}
            for side, label in ((1, 'own'), (-1, 'rival')):
                rows = [r for r in record['points'] if r['server'] == side]
                reply_rows = [r for r in rows if r['evaluator']['legal_serve'] and r['rival_returns'] > 0]
                routed_before_reply = routed_by_reply_bounce = 0
                for row in reply_rows:
                    first_rival_contact = next(e['step'] for e in row['events']
                                               if e['kind'] == 'contact' and e['side'] == -1)
                    reply_bounce = next(e['step'] for e in row['events']
                                        if e['kind'] == 'legal_return' and e['side'] == -1)
                    opening = row['timing_route']['first_receiver_frame']
                    routed_before_reply += opening is not None and opening <= first_rival_contact
                    routed_by_reply_bounce += opening is not None and opening <= reply_bounce
                route_metrics[label] = dict(recognized=sum(r['timing_route']['visual_role'] == label for r in rows),
                                            opened_receiver=sum(r['timing_route']['first_receiver_frame'] is not None
                                                                for r in rows),
                                            early_reply_routes=routed_before_reply,
                                            routes_by_reply_bounce=routed_by_reply_bounce,
                                            reply_points=len(reply_rows),
                                            replied_points_won=sum(r['points_for'] for r in reply_rows))
            fixture_report['branches'][name] = {k: record[k] for k in ('score', 'metrics', 'commands', 'stereo_pairs')}
            phase_counts = {phase: sum(r['phase'] == phase for r in record['points']) for phase in PHASES}
            fixture_report['branches'][name].update(replayed_points=count, routing=route_metrics, phases=phase_counts)
        audited['fixtures'].append(fixture_report)
    _write(path(folder, body, 'comparison'), audited)
    print(body, 'audit matched', [r['own_tosses_matched'] for r in audited['fixtures']], flush=True)


def carry(folder, limit):
    seeds = (873643541, 1656343319, 753222359)
    course = ('gantry', 'rotary', 'gantry')
    previous = None
    results = []
    seen = set()
    for index, (body, seed) in enumerate(zip(course, seeds, strict=True)):
        label = f'carry-{index + 1}'
        source = previous if previous is not None else path(folder, body, 'start')
        previous = play_game(folder, body, 'selected', index % 2, limit, prior_path=previous, run_seed=seed, label=label)
        result = _read(path(folder, body, label))
        if result['prior_sha256'] != _hash(source) or result['checkpoint_sha256'] != _hash(previous):
            raise ValueError('a carried game failed checkpoint provenance')
        for cases in result['schedule'].values():
            for start, _ in cases:
                if start in seen:
                    raise ValueError('carried games share a point start')
                seen.add(start)
        results.append({key: result[key] for key in ('seed', 'score', 'metrics', 'commands', 'replayed_points',
                                                     'prior_sha256', 'checkpoint_sha256')})
        results[-1].update(body=body, entry_observations=result.get('entry_observations', []),
                           model_count=len(_read(previous)['models']))
    play_game(folder, course[-1], 'selected', 0, limit, run_seed=seeds[-1], label='revisit-fresh')
    restart = _read(path(folder, course[-1], 'revisit-fresh'))
    revisit = _read(path(folder, course[-1], 'carry-3'))
    if any(restart[key] != revisit[key] for key in ('schedule', 'first_server', 'profile', 'timing')):
        raise ValueError('the restarted and carried revisit used different setup')
    _write(folder / 'body-carry-comparison.json.gz', dict(course=results,
           fresh_revisit={key: restart[key] for key in ('score', 'metrics', 'commands', 'replayed_points')},
           continued_checkpoint=previous.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'practice', 'play', 'audit', 'carry'))
    parser.add_argument('body', choices=SEEDS)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--probe-cache', type=Path)
    parser.add_argument('--profiles', nargs='+', choices=(*(f'classic-{i}' for i in range(6)), 'drive'))
    parser.add_argument('--branch', choices=BRANCHES, default='selected')
    parser.add_argument('--fixture', type=int, choices=(1, 2), default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', type=Path, help='continue the shared player from a postgame cross-body checkpoint')
    parser.add_argument('--seed', type=int, help='schedule an independent first-to-11 game when continuing')
    parser.add_argument('--label', help='output name for an independently scheduled game')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    if args.stage == 'prepare':
        prepare(args.folder, args.body, args.probe_cache)
    elif args.stage == 'practice':
        profiles = args.profiles or (('classic-3', 'drive') if args.body == 'gantry'
                                     else ('classic-3', 'classic-1', 'classic-4', 'classic-2', 'classic-5', 'drive'))
        practice(args.folder, args.body, list(profiles), args.limit or 8)
    elif args.stage == 'play':
        continuation = (args.resume is not None, args.seed is not None, args.label is not None)
        if any(continuation) and not all(continuation):
            parser.error('continuation requires a checkpoint, independent seed and output label')
        play_game(args.folder, args.body, args.branch, args.fixture - 1, args.limit or 65,
                  prior_path=args.resume, run_seed=args.seed, label=args.label)
    elif args.stage == 'audit':
        audit(args.folder, args.body)
    else:
        carry(args.folder, args.limit or 65)


if __name__ == '__main__':
    main()
