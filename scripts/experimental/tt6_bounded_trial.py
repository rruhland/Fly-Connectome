"""Compare clip-after-solve and bounded predictive motor plans on paired serves."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _hash, _read, _write, load_sources
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_bounded import BoundedPredictivePlayer
from .tt6_continuous import BodyLibrary
from .tt6_racket import RacketAdaptation
from .tt6_render import cache_hall_stereo
from .tt6_rotary_adapt_trial import commission, play
from .tt6_timing import DelayedWiring, TimingMemory


ORIGINAL = ROOT / 'artifacts' / 'tt6' / 'combined-rotary.json.gz'
HELD_SEEDS = (99603119, 73104537)


def _paired(wiring, fit, tracker, fresh_prior, seed, gain, phases, course, receipt, output):
    for condition in ('clip', 'bounded'):
        result = dict(condition=condition, points=[])
        course['conditions'][condition] = result
        player = BoundedPredictivePlayer(fit, deepcopy(fresh_prior), tracker, RacketAdaptation(gain=gain),
                                         TimingMemory(), bounded=condition == 'bounded', solver=receipt['solver'])
        play(wiring, player, seed, phases, course['schedule'], result, output, receipt)
        result['motors'] = {key: sum(point['motor_plans'][key] for point in result['points'])
                            for key in result['points'][0]['motor_plans']}
        _write(output, receipt)


def _replay(output, receipt, original, native, mini):
    before = _read(original)
    visit = next(row for row in before['visits'] if row['body'] == 'rotary')
    installation = visit['installation']
    schedule = visit['schedule']
    seed = before['seed']
    prior = TransferPrior('first_person_mini', native, seed, mini=mini)
    arena = TT6Arena(installation['arena_seed'], body='rotary', view_seed=installation['view_seed'],
                     **before['geometry'])
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, installation['arena_seed'])
        fit, tracker, probes, attempts = BodyLibrary([]).commission(DelayedWiring(wiring), arena.cameras, schedule)
        if fit is None:
            raise ValueError(f'could not commission original rotary installation: {attempts}')
        course = dict(schedule=schedule, installation=installation, fit=fit.stats(), probes=probes,
                      attempts=attempts, conditions={}, evaluator_wiring=wiring.state())
        receipt['diagnosis'] = course
        phases = [point['phase'] for point in visit['points']]
        _paired(wiring, fit, tracker, prior, seed, before['gain'], phases, course, receipt, output)
        course['clip_replays_original'] = all(
            old['action_sha256'] == new['action_sha256'] and old['evaluator'] == new['evaluator']
            for old, new in zip(visit['points'], course['conditions']['clip']['points'], strict=True))
        _write(output, receipt)
    finally:
        arena.close()


def run(output, original, held_seeds, solver):
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=140)
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    mini = states['first_person_mini']
    gain = _read(original)['gain']
    receipt = dict(schema='tt6-bounded-motor-plan-v1', geometry=geometry, reflections=True, gain=gain, solver=solver,
                   sources={**hashes, original.name: _hash(original)}, held=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    _replay(output, receipt, original, native, mini)
    for seed in held_seeds:
        phases = tuple(np.random.default_rng(seed + 143).permutation(
            ('mixed', 'mixed', 'mixed', 'mixed', 'ordinary', 'ordinary', 'soft', 'delay')).tolist())
        prior = TransferPrior('first_person_mini', native, seed, mini=mini)
        arena, wiring, fit, tracker, course = commission(seed, len(phases), geometry)
        try:
            course.update(seed=seed, conditions={})
            receipt['held'].append(course)
            _paired(wiring, fit, tracker, prior, seed, gain, phases, course, receipt, output)
        finally:
            arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--original', type=Path, default=ORIGINAL)
    cli.add_argument('--held-seeds', type=int, nargs='*', default=HELD_SEEDS)
    cli.add_argument('--solver', choices=('active', 'slsqp'), default='active')
    args = cli.parse_args()
    run(args.output, args.original, args.held_seeds, args.solver)
