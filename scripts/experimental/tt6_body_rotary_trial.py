"""Commission and score the six-axis rotary receiver using public stereo RGB."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _seeds, _write, load_sources
from .tt6_body_arena import BODY_AXES
from .tt6_body_control import CameraBodyPlayer, fit_camera_body
from .tt6_body_trial import _setup
from .tt6_body_vision import body_tracker_factory


def run(output, seed, probes, points, max_steps):
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    schedule = _seeds(seed, 0, probes, points)
    geometry = dict(probe_horizon=34, cached_hall=True,
                    arena=dict(venue='hall', width=192, height=144, baseline=.28, max_steps=max_steps))
    arena, wiring, histories, calibration = _setup('rotary', schedule, geometry, None)
    try:
        tracker = body_tracker_factory(arena.cameras, histories)
        fit = fit_camera_body(histories, tracker)
        prior = TransferPrior('first_person_mini', native, schedule['player_seed'], mini=states['first_person_mini'])
        player = CameraBodyPlayer(fit, prior, tracker)
        axes = BODY_AXES['rotary'][0]
        fixture = Path(__file__).with_name('tt6_body_rotary.xml')
        rows = []
        receipt = dict(schema='tt6-rotary-rgb-play-v1', body='rotary', active_axes=axes,
                       geometry=geometry, schedule=schedule, calibration=calibration, adapter=fit.stats(),
                       evaluator_wiring=wiring.state(), inputs_sha256={**hashes, fixture.name: _hash(fixture)},
                       points=rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        for start in schedule['tests']:
            row = _point(wiring, player, start)
            row.update(_counts(row))
            actions = np.asarray(row['actions'])
            row['mean_abs_public_action'] = np.abs(actions).mean(axis=0).tolist()
            rows.append(row)
            receipt['summary'] = _metrics(rows)
            _write(output, receipt)
            print(start, row['contacts'], row['player_returns'], row['rival_returns'], row['evaluator']['reason'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--seed', type=int, default=48739061)
    cli.add_argument('--probes', type=int, default=8)
    cli.add_argument('--points', type=int, default=6)
    cli.add_argument('--max-steps', type=int, default=180)
    args = cli.parse_args()
    if min(args.probes, args.points, args.max_steps) < 1:
        cli.error('probe count, point count and maximum steps must be positive')
    run(args.output, args.seed, args.probes, args.points, args.max_steps)
