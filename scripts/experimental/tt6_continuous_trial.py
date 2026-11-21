"""Test carried experience while the physical arm, contact response and command timing change."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _seeds, _write, load_sources
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_continuous import BodyLibrary, ContinualPlayer
from .tt6_racket import RacketAdaptation, set_racket_response
from .tt6_timing import DelayedWiring, Disturbance, TimingMemory


def point_condition(phase):
    if phase == 'ordinary':
        return 'stock', Disturbance()
    if phase == 'soft':
        return 'medium', Disturbance()
    if phase == 'mixed':
        return 'medium', Disturbance(delays=((0, 1), (18, 3), (67, 1)),
                                     noise=((0, .025), (18, .06), (67, .035)))
    if phase == 'delay':
        return 'stock', Disturbance(delays=((0, 2), (62, 0)),
                                    noise=((0, .045), (62, .025)))
    raise ValueError(f'unknown phase {phase}')


def match_phases(phases, limit, seed):
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < limit:
        order.extend(rng.permutation(phases).tolist())
    return order[:limit]


def prepare_point(arena, phase):
    rubber, disturbance = point_condition(phase)
    set_racket_response(arena, rubber)
    return disturbance


def run(args, *, make_player=None, after_visit=None, prepare=prepare_point, protocol=None):
    if args.race_to < 0 or min(args.max_steps, args.match_limit, args.probes) < 1:
        raise ValueError('the run needs a positive probe budget and valid match lengths')
    if args.rally_response and (not args.predictive or args.mode == 'frozen'):
        raise ValueError('within-rally adjustment requires adaptive predictive control')
    if args.start_visit and args.resume is None:
        raise ValueError('skipping earlier visits requires their checkpoint')
    if not 0 <= args.start_visit < len(args.course):
        raise ValueError('start visit is outside course')
    if args.resume is not None:
        from .tt6_memory import load
        if args.mode != 'carry' or args.gain is not None:
            raise ValueError('resume uses the carried player and its stored gain')
        memory, prior, response, timing = load(args.resume)
        hashes = {'resume': _hash(args.resume)}
        gain = response.gain
    else:
        native, states, hashes = load_sources(ROOT/'artifacts/tt4/mini-first-person-player.json.gz')
        prior = TransferPrior('first_person_mini', native, args.seed, mini=states['first_person_mini'])
        memory = BodyLibrary([])
        gain = args.gain
        if gain is None:
            gain = _read(args.practice)['mini_practice']['varied']['gain']
            hashes['mini_rubber_practice'] = _hash(args.practice)
        response = RacketAdaptation(gain=gain, enabled=args.mode != 'frozen')
        timing = TimingMemory([1., 0., 0., 0., 0., 0.]) if args.mode == 'frozen' else TimingMemory()
    if args.match_rng is not None:
        if args.resume is not None or args.mode != 'carry':
            raise ValueError('the matched random-stream control starts from the mini donor')
        prior.rng.bit_generator.state = _read(args.match_rng)['rng']
        hashes['policy_rng_only'] = _hash(args.match_rng)
    geometry = dict(width=192, height=144, baseline=.28, view='first_person', venue='hall',
                    render_mode='mujoco', max_steps=args.max_steps, frameskip=8, distractor=False)
    report = dict(seed=args.seed, geometry=geometry, surfaces=args.phases, cached_render=args.cached_render, predictive=args.predictive,
                  mode=args.mode, source_sha256=hashes, reflections=args.reflections, visits=[], gain=gain)
    if protocol is not None:
        report['protocol'] = protocol
    if args.rally_response:
        report['rally_response'] = True
    if args.auto_servo:
        report['auto_servo'] = True
    if args.race_to:
        report['match'] = dict(race_to=args.race_to, lead=2, limit=args.match_limit)
    entry_memory = None
    entry_state = _read(args.resume).get('entry_memory') if args.resume else None
    if args.entry_memory is not None:
        if args.mode != 'carry':
            raise ValueError('the learned stroke initializer requires carried training')
        entry_state = _read(args.entry_memory)
        hashes['entry_memory'] = _hash(args.entry_memory)
    if entry_state is not None:
        from .tt6_entry_memory import EntryMemory
        entry_memory = EntryMemory.from_state(entry_state)
    shot_memory = None
    shot_state = _read(args.resume).get('shot_memory') if args.resume else None
    if args.shot_memory is not None:
        shot_state = _read(args.shot_memory)
        hashes['shot_memory'] = _hash(args.shot_memory)
    if shot_state is not None:
        from .tt6_shot_memory import LocalShotPlayer, ShotMemory

        if args.auto_servo or args.rally_response:
            raise ValueError('select a single experimental player extension')
        shot_memory = ShotMemory.from_state(shot_state)
    homes = {}
    if args.stations_from is not None:
        homes.update((v['install'], v['installation']) for v in _read(args.stations_from)['visits'])
    for i, name in enumerate(args.course):
        # Repeated labels select the same physical installation only in this evaluator.
        if name not in homes:
            homes[name] = _seeds(args.seed + 109, len(homes), 1, 1)
        if i < args.start_visit:
            continue
        install = homes[name]
        body = name.partition('-')[0]
        phases = args.phases
        if args.race_to:
            phases = match_phases(phases, args.match_limit, [args.seed, i, 7919])
        schedule = _seeds(args.seed, i, args.probes, len(phases))
        arena = TT6Arena(install['arena_seed'], body=body, view_seed=install['view_seed'], **geometry)
        try:
            if args.mode == 'restart':
                prior = TransferPrior('first_person_mini', native, args.seed, mini=states['first_person_mini'])
            if args.mode in ('reset_skills', 'restart'):
                response = RacketAdaptation(gain=gain)
                timing = TimingMemory()
            if args.cached_render:
                from .tt6_render import cache_hall_stereo
                cache_hall_stereo(arena, reflections=args.reflections)
            wiring = TT6Wiring(arena, install['arena_seed'])
            library = BodyLibrary([]) if args.mode in ('refit', 'restart') else memory
            adapter, factory, probes, attempts = library.commission(
                DelayedWiring(wiring), arena.cameras, schedule, probe_cache=getattr(args, 'probe_cache', None))
            before = prior.state()
            visit = dict(body=body, install=name, schedule=schedule, installation=install,
                         wiring=wiring.state(), probes=probes, attempts=attempts, before=before, points=[])
            if adapter is not None:
                if entry_memory is not None and 'recalled' not in attempts[-1]:
                    visit['learned_entry_speed'] = entry_memory.enter(response)
                if make_player is not None:
                    player = make_player(adapter, prior, factory, response, timing)
                elif shot_memory is not None:
                    player = LocalShotPlayer(adapter, prior, factory, response, timing, adjust=args.mode != 'frozen',
                                             predictive=args.predictive, shot_memory=shot_memory)
                elif args.auto_servo:
                    from .tt6_select import SelectingPlayer

                    player = SelectingPlayer(adapter, prior, factory, response, timing, adjust=args.mode != 'frozen')
                    visit['servo'] = player.servo_evidence
                else:
                    player_class = ContinualPlayer
                    if args.rally_response:
                        from .tt6_rally_response import RallyResponsePlayer
                        player_class = RallyResponsePlayer
                    player = player_class(adapter, prior, factory, response, timing, adjust=args.mode != 'frozen',
                                          predictive=args.predictive)
                visit['fit'] = adapter.stats()
                score = [0, 0]
                for phase, point_seed in zip(phases, schedule['tests'], strict=True):
                    disturbance = prepare(arena, phase)
                    disturbed = DelayedWiring(wiring, disturbance, seed=args.seed)
                    row = _point(disturbed, player, point_seed)
                    point = dict(phase=phase, applied=disturbed.applied, **row, **_counts(row))
                    visit['points'].append(point)
                    if entry_memory is not None:
                        entry_memory.observe(point)
                    score[0] += point['points_for']
                    score[1] += point['points_against']
                    if args.race_to and max(score) >= args.race_to and abs(score[0] - score[1]) >= 2:
                        break
                if args.race_to:
                    visit['score'] = score
                    visit['finished'] = max(score) >= args.race_to and abs(score[0] - score[1]) >= 2
            if entry_memory is not None:
                entry_memory.finish_visit()
                visit['entry_memory'] = entry_memory.state()
            visit['after'] = prior.state()
            if args.checkpoints is not None:
                from .tt6_memory import save
                save(args.checkpoints / f'visit-{i + 1}.json.gz', library, prior, response, timing,
                     entry_memory=entry_memory, shot_memory=shot_memory)
            if after_visit is not None:
                after_visit(visit)
            report['visits'].append(visit)
            _write(args.output, report)
            print(i, name, len(probes), attempts[-1] if attempts else 'no supported fit', [(p['phase'],p['player_returns'],p['points_for'],
                  p.get('timing',{}).get('estimated'),p['impact']['next_speed']) for p in visit['points']], flush=True)
        finally:
            arena.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--course', nargs='+', default=['gantry','offset','gantry-b','offset','gantry','gantry-b'])
    p.add_argument('--phases', nargs='+', default=['ordinary','ordinary','soft','mixed','mixed','delay','ordinary'])
    p.add_argument('--mode', choices=['carry','refit','reset_skills','restart','frozen'], default='carry')
    servo = p.add_mutually_exclusive_group()
    servo.add_argument('--predictive', action='store_true')
    servo.add_argument('--auto-servo', action='store_true', help='select a solver from the fitted camera response')
    p.add_argument('--cached-render', action='store_true')
    p.add_argument('--probe-cache', type=Path, help='reuse public calibration RGB in this directory')
    p.add_argument('--probes', type=int, default=8)
    p.add_argument('--rally-response', action='store_true', help='use an observed rebound to adjust later strikes in the same point')
    p.add_argument('--race-to', type=int, default=0, help='play receiving games to this score, winning by two')
    p.add_argument('--max-steps', type=int, default=140, help='control-frame cap per point')
    p.add_argument('--match-limit', type=int, default=70, help='maximum receiving points in each game')
    p.add_argument('--entry-memory', type=Path, help='start newly fitted arms at a speed learned across previous visits')
    p.add_argument('--shot-memory', type=Path, help='choose shots from rewards on similar fitted motor responses')
    p.add_argument('--match-rng', type=Path, help='start from mini with only the policy RNG copied from this checkpoint')
    p.add_argument('--gain', type=float, help='override the gain from varied mini practice')
    p.add_argument('--practice', type=Path, default=ROOT/'artifacts/tt6/racket-practice.json.gz')
    p.add_argument('--reflections', action='store_true',help='retain live floor reflections when caching')
    p.add_argument('--checkpoints', type=Path, help='save the learned state after each visit')
    p.add_argument('--start-visit', default=0, type=int, help='zero-based first visit to play on resuming the same course')
    p.add_argument('--resume', type=Path, help='resume learned state at an installation boundary')
    p.add_argument('--stations-from', type=Path, help='reuse physical installations from a previous receipt')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
