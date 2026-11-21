"""Check the opt-in wrist pose qualifier on the existing native receive starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_factory_serve_trial import load_checkpoint
from .tt20_fixture import TT20Arena
from .tt20_mono_flight_trial import CATALOG, CHECKPOINT, metrics, play, replay
from .tt20_natural import NaturalViewTransport


DATA = ROOT / 'artifacts/tt20'


def _batch(settings, points):
    arena = TT20Arena(**settings, view_mode='wrist')
    player = load_checkpoint(CHECKPOINT, CATALOG, 'factory').player
    rows = []
    try:
        optics = arena.lens_intrinsics()
        for original in points:
            track = NaturalViewTransport('wrist', lens_fov=optics, flight=False,
                                         continuous_wrist=True).tracker()
            row = play(arena, player, track, original['seed'])
            rows.append(dict(index=original['index'], seed=original['seed'], size=original['size'], temporal=row))
    finally:
        player.close()
        arena.close()
    return rows


def run(output, workers):
    source = DATA / 'mono-flight-wrist.json.gz'
    baseline = _read(source)
    if (_hash(CHECKPOINT) != baseline['checkpoint_sha256'] or _hash(CATALOG) != baseline['catalogue_sha256']
            or _hash(Path(__file__).with_name('tt20_fixture.py')) != baseline['implementation']['tt20_fixture.py']):
        raise ValueError('the held wrist fixture or player differs from the saved comparison')
    originals = baseline['pairs']['wrist']
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(_batch, baseline['arena'], originals[i::workers]) for i in range(workers)]
        pairs = sorted((row for job in jobs for row in job.result()), key=lambda row: row['index'])
    # Reuse the physical tape verifier. Its channel names are independent of
    # which optical mechanism produced the two saved action sequences.
    replayed = replay(dict(arena=baseline['arena'], pairs={'wrist': pairs}))
    result = dict(schema='tt20-wrist-continuity-play-v1', source=source.name, source_sha256=_hash(source),
                  arena=baseline['arena'], point_seeds=baseline['point_seeds'], target_practice_points=0,
                  modes={'size': 'size_only', 'temporal': 'size_with_continuity'}, pairs=pairs,
                  metrics=metrics(pairs), replayed_points=replayed)
    _write(output, result)
    print(result['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'natural-wrist-continuity-play.json.gz')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)
