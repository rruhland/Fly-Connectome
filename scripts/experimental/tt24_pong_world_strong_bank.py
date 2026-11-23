"""Replay and export the first six registered eight-update receiving policies."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_pong_world_strong import CUT, OUTPUT, PLAN, PREFIX, RECEIPT
from .tt24_pong_world_train import FRAMES, FROZEN, MODES, SOURCE, Worlds, acquire, certify, save
from .tt24_world_prior import WorldPrior


BANK = f'{PREFIX}-bank'
STREAMS = 6


def export(workers):
    plan, initializer, historical = map(_read, (PLAN, CUT, RECEIPT))
    if (historical['metadata']['plan_sha256'] != _hash(PLAN)
            or historical['metadata']['init_sha256'] != _hash(CUT)
            or plan['physics_sha256'] != _hash(SOURCE)
            or plan['frozen_at'] != FROZEN or tuple(plan['arms']) != MODES):
        raise ValueError('historical acquisition lineage changed')
    initial = certify(initializer)
    world = Worlds(workers=workers, frames=FRAMES)
    stream_rows = []
    try:
        for index, seed in enumerate(plan['cohorts'][:STREAMS]):
            previous = historical['cohorts'][index]
            replay = acquire(world, initial, seed)
            if replay != previous:
                raise ValueError(f'historical world stream {index} failed exact replay')
            files = {}
            for name, weights in zip(MODES, replay['deployed'], strict=True):
                path = OUTPUT / f'{BANK}-{index:02d}-{name}.npz'
                save(path, weights)
                digest = _hash(path)
                if not np.array_equal(WorldPrior.load(path).weights, weights):
                    raise ValueError(f'exported policy {path.name} changed')
                if index == 0 and digest != historical['low_budget_policy_sha256'][name]:
                    raise ValueError(f'first stream policy {name} differs from the frozen hash')
                files[name] = dict(file=path.name, sha256=digest)
            stream_rows.append(dict(index=index, rng=seed, policy_files=files,
                                    exposure=replay['updates'][FROZEN - 1]['exposure']))
            print('exactly replayed and exported', index, seed, flush=True)
    finally:
        world.close()
    _write(OUTPUT / f'{BANK}.json', dict(schema='tt24-pong-world-strong-bank-1', update=FROZEN,
                                        plan_sha256=_hash(PLAN), initial_sha256=_hash(CUT),
                                        receipt_sha256=_hash(RECEIPT), arms=list(MODES), streams=stream_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    export(args.workers)


if __name__ == '__main__':
    main()
