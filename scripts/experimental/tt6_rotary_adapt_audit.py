"""Rerun the documented rotary failure with the camera-feedback controller."""

import argparse
from pathlib import Path

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _hash, _metrics, _read, load_sources
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_continuous import BodyLibrary
from .tt6_racket import RacketAdaptation
from .tt6_render import cache_hall_stereo
from .tt6_rotary_adapt import CoherentContinualPlayer, RotaryPlayer, STROKES
from .tt6_rotary_adapt_trial import play
from .tt6_timing import DelayedWiring, TimingMemory


def audit(original, output, *, predictive=False):
    before = _read(original)
    visit = next(row for row in before['visits'] if row['body'] == 'rotary')
    schedule = visit['schedule']
    installation = visit['installation']
    seed = before['seed']
    native, mini, _ = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    prior = TransferPrior('first_person_mini', native, seed, mini=mini['first_person_mini'])
    arena = TT6Arena(installation['arena_seed'], body='rotary', view_seed=installation['view_seed'],
                     **before['geometry'])
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, installation['arena_seed'])
        fit, factory, probes, attempts = BodyLibrary([]).commission(DelayedWiring(wiring), arena.cameras, schedule)
        if fit is None:
            raise ValueError(f'rotary RGB commissioning failed: {attempts}')
        if predictive:
            response = RacketAdaptation(gain=before['gain'])
            player = CoherentContinualPlayer(fit, prior, factory, response, TimingMemory(), predictive=True)
        else:
            player = RotaryPlayer(fit, prior, factory, stroke=STROKES[1])
        phases = [point['phase'] for point in visit['points']]
        result = dict(condition='coherent-predictive' if predictive else 'coherent-feedback-launch', points=[])
        receipt = dict(schema='tt6-rotary-failure-audit-v1', original_sha256=_hash(original),
                       seed=seed, geometry=before['geometry'], reflections=True, schedule=schedule,
                       installation=installation, probes=probes, attempts=attempts, fit=fit.stats(),
                       original_summary=_metrics(visit['points']), repeated=result)
        output.parent.mkdir(parents=True, exist_ok=True)
        play(wiring, player, seed, phases, schedule, result, output, receipt)
    finally:
        arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--original', type=Path, default=ROOT / 'artifacts' / 'tt6' / 'combined-rotary.json.gz')
    cli.add_argument('--predictive', action='store_true')
    args = cli.parse_args()
    audit(args.original, args.output, predictive=args.predictive)
