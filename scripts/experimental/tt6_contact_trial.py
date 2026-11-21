"""Compare visually planned follow-through on matched six-axis rotary rallies."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _write, load_sources
from .tt6_contact import HIGH_LOFT, ContactPlayer, FollowThrough
from .tt6_continuous_trial import point_condition
from .tt6_racket import set_racket_response
from .tt6_rotary_adapt import STROKES, Stroke
from .tt6_rotary_adapt_trial import commission
from .tt6_timing import DelayedWiring


BOOST = Stroke(.083, .145, .95, .035, .50)
FOLLOW = FollowThrough(5., .07, .045, .016)
CONDITIONS = {
    'baseline': (STROKES[1], FollowThrough(0.)),
    'follow_only': (STROKES[1], FollowThrough()),
    'lift_only': (BOOST, replace(FOLLOW, frames=0.)),
    'lift_and_follow': (BOOST, FOLLOW),
}
DEFAULT_CONDITIONS = tuple(CONDITIONS)
CONDITIONS['high_loft'] = HIGH_LOFT


def summarize(points):
    return {phase: _metrics([point for point in points if point['phase'] == phase])
            for phase in ('mixed', 'soft')}


def run(output, seed, points, conditions):
    if points < 3:
        raise ValueError('expected at least three matched starts')
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=140)
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    arena, wiring, fit, factory, commissioning = commission(seed, points, geometry)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        phases = list((('mixed', 'mixed', 'soft') * ((points + 2) // 3))[:points])
        np.random.default_rng(seed + 253).shuffle(phases)
        specs = {name: dict(stroke=asdict(CONDITIONS[name][0]), follow=asdict(CONDITIONS[name][1]))
                 for name in conditions}
        report = dict(schema='tt6-mixed-rotary-follow-through-v1', seed=seed, geometry=geometry,
                      phases=phases, renderer='cached native with reflections', commissioning=commissioning,
                      conditions=specs, sources_sha256=hashes, runs={}, summary={},
                      implementation_sha256={path.name: _hash(path) for path in
                                             (Path(__file__), Path(__file__).with_name('tt6_contact.py'))})
        for name in conditions:
            stroke, follow = CONDITIONS[name]
            prior = TransferPrior('first_person_mini', native, seed, mini=states['first_person_mini'])
            player = ContactPlayer(fit, prior, factory, stroke=stroke, follow=follow)
            rows = []
            for phase, start in zip(phases, commissioning['schedule']['tests'], strict=True):
                response, disturbance = point_condition(phase)
                set_racket_response(arena, response)
                row = _point(DelayedWiring(wiring, disturbance, seed=seed), player, start)
                row.update(_counts(row), phase=phase)
                rows.append(row)
                report['runs'][name] = rows
                report['summary'][name] = summarize(rows)
                _write(output, report)
                print(name, phase, start, row['contacts'], row['player_returns'], row['rival_returns'],
                      row['follow_through_frames'], row['evaluator']['reason'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--seed', type=int, default=42680951)
    cli.add_argument('--points', type=int, default=12)
    cli.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=DEFAULT_CONDITIONS)
    args = cli.parse_args()
    run(args.output, args.seed, args.points, args.conditions)
