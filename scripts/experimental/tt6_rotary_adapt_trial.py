"""Practice an RGB stroke policy and carry it to unseen rotary-arm serves."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _metrics, _point, _seeds, _write, load_sources
from .tt5_vision import motion_tracker_factory
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_body_control import fit_camera_body
from .tt6_body_probe import probe_histories
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous import ContinualPlayer
from .tt6_continuous_trial import point_condition
from .tt6_racket import RacketAdaptation, set_racket_response
from .tt6_render import cache_hall_stereo
from .tt6_rotary_adapt import RotaryPlayer, STROKES, StrokeMemory
from .tt6_timing import DelayedWiring, TimingMemory


PHASES = ('ordinary', 'soft', 'mixed', 'delay')


def commission(seed, points, geometry):
    installation = _seeds(seed + 109, 0, 1, 1)
    schedule = _seeds(seed, 0, 8, points)
    arena = TT6Arena(installation['arena_seed'], body='rotary', view_seed=installation['view_seed'], **geometry)
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, installation['arena_seed'])
        histories, probes = probe_histories(wiring, schedule, {'probe_horizon': 34})
        for track in (body_tracker_factory, motion_tracker_factory):
            factory = track(arena.cameras, histories)
            try:
                fit = fit_camera_body(histories, factory)
                break
            except ValueError as err:
                failure = err
        else:
            raise failure
        report = dict(schedule=schedule, installation=installation, probes=probes, fit=fit.stats(),
                      tracker=track.__name__, evaluator_wiring=wiring.state())
        return arena, wiring, fit, factory, report
    except Exception:
        arena.close()
        raise


def _conditions(seed, size, *, balanced=False):
    if balanced:
        order = np.random.default_rng(seed + 143).permutation(PHASES).tolist()
        return [order[i % len(order)] for i in range(size)]
    phases = list(PHASES) * ((size + len(PHASES) - 1) // len(PHASES))
    phases = phases[:size]
    np.random.default_rng(seed + 143).shuffle(phases)
    return phases


def play(wiring, player, seed, phases, schedule, result, output, receipt):
    for phase, start in zip(phases, schedule['tests'], strict=True):
        response, disturbance = point_condition(phase)
        set_racket_response(wiring.arena, response)
        point = _point(DelayedWiring(wiring, disturbance, seed=seed), player, start)
        point.update(_counts(point), phase=phase)
        result['points'].append(point)
        result['summary'] = _metrics(result['points'])
        print(result['condition'], phase, start, point.get('stroke_choice'), point['contacts'],
              point['player_returns'], point['points_for'], point.get('stroke_score'), flush=True)
        _write(output, receipt)


def run(output, practice_seed, held_seed, practice_points, held_points):
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=140)
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    prior = TransferPrior('first_person_mini', native, practice_seed, mini=states['first_person_mini'])
    memory, timing = StrokeMemory(), TimingMemory()
    receipt = dict(schema='tt6-rotary-stroke-carry-v1', geometry=geometry, reflections=True,
                   practice_seed=practice_seed, held_seed=held_seed, inputs_sha256=hashes,
                   strokes=[asdict(stroke) for stroke in STROKES])
    arena, wiring, fit, factory, practice = commission(practice_seed, practice_points, geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt['practice'] = practice
        practice.update(condition='practice', points=[])
        player = RotaryPlayer(fit, prior, factory, memory=memory, explore=True, timing=timing)
        play(wiring, player, practice_seed, _conditions(practice_seed, practice_points, balanced=True), practice['schedule'],
             practice, output, receipt)
        receipt['stroke_memory'] = memory.state()
    finally:
        arena.close()
    arena, wiring, fit, factory, held = commission(held_seed, held_points, geometry)
    try:
        receipt['held'] = held
        held['conditions'] = {}
        phases = _conditions(held_seed, held_points)
        for condition in ('carried', 'frozen', 'predictive'):
            result = dict(condition=condition, points=[])
            held['conditions'][condition] = result
            if condition == 'predictive':
                player = ContinualPlayer(fit, deepcopy(prior), factory, RacketAdaptation(gain=.73),
                                         deepcopy(timing), predictive=True)
            else:
                player = RotaryPlayer(fit, deepcopy(prior), factory, timing=deepcopy(timing),
                                      memory=memory if condition == 'carried' else None)
            play(wiring, player, held_seed, phases, held['schedule'], result, output, receipt)
    finally:
        arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--practice-seed', type=int, default=83169851)
    cli.add_argument('--held-seed', type=int, default=46419231)
    cli.add_argument('--practice-points', type=int, default=12)
    cli.add_argument('--held-points', type=int, default=8)
    args = cli.parse_args()
    if args.practice_seed == args.held_seed or args.practice_points < 6 or args.held_points < 4:
        cli.error('use disjoint seeds, at least six practice points, and at least four held points')
    run(args.output, args.practice_seed, args.held_seed, args.practice_points, args.held_points)
