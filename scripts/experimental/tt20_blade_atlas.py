"""Rebuild the receiving shortlist at the optically inferred SO-101 camera load."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from mujoco import rollout
import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import make_players, play, previous_seeds
from .tt19_direct_receive_payload import PayloadArena
from .tt19_flight import FlightLaw
from .tt19_payload_adapt import WeightedSimulator
from .tt20_blade_trial import metrics, replay


FRESH_SEED = 796419497
FRESH_COUNT = 6
ARCHIVE = 'blade-loaded-atlas.npz'
RECEIPT = 'direct-receive-loaded-atlas.json.gz'


def blade_pose(poses, center, normal):
    vector = poses[..., 4:7]
    scalar = poses[..., 3:4]

    def rotate(axis):
        first = 2. * np.cross(vector, axis)
        return axis + scalar * first + np.cross(vector, first)

    face = rotate(normal)
    face = np.where(face[..., 1:2] >= 0, face, -face)
    spot = poses[..., :3] + rotate(center) - [0., 0., TABLE_HEIGHT]
    return np.concatenate((spot, face[..., [0, 2]]), axis=-1)


def rebuild(nominal, target, mass):
    start = perf_counter()
    with np.load(nominal) as archive:
        tapes, old = archive['tapes'], archive['poses']
    sim = WeightedSimulator(FlightLaw(), mass)
    try:
        model = sim.model
        joint = 1 + int(model.joint('learner_paddle_free').qposadr[0])
        geom = model.geom('learner_racket')
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, geom.quat)
        basis = rotation.reshape((3, 3))[:, 2]
        batches = []
        for group in np.array_split(tapes, int(np.ceil(len(tapes) / 24))):
            steps = np.repeat(group, sim.timestep, axis=1)
            control = np.broadcast_to(sim.home, (len(group), steps.shape[1], len(sim.home))).copy()
            integrated = sim.home[sim.axes] + np.cumsum(steps * sim.increments, axis=1)
            control[:, :, sim.axes] = np.clip(integrated, sim.low, sim.high)
            states, _ = rollout.rollout(model, sim.data, sim.base[None], control)
            path = states[:, sim.timestep - 1::sim.timestep, joint:joint + 7]
            first = np.broadcast_to(sim.base[joint:joint + 7], (len(group), 1, 7))
            batches.append(blade_pose(np.concatenate((first, path), axis=1), geom.pos, basis))
        poses = np.concatenate(batches).astype(np.float32)
    finally:
        sim.close()
    np.savez_compressed(target, tapes=tapes, poses=poses)
    shift = np.linalg.norm(poses[..., :3] - old[..., :3], axis=-1)
    return dict(count=len(tapes), model_control_frames=int(np.prod(tapes.shape[:2])), seconds=perf_counter() - start,
                nominal_sha256=_hash(nominal), rebuilt_sha256=_hash(target),
                position_difference_median_m=float(np.median(shift)),
                position_difference_p95_m=float(np.percentile(shift, 95)), position_difference_max_m=float(shift.max()))


def batch(nominal, corrected, source, args, mass, trials, fresh):
    arena = PayloadArena(**args)
    rebuilt, unused = make_players(corrected, source)
    unused.close()
    rebuilt.sim.close()
    rebuilt.sim = WeightedSimulator(rebuilt.law, mass)
    if fresh:
        frozen, spare = make_players(nominal, source)
        spare.close()
    else:
        frozen = None
    players = {'frozen': frozen, 'corrected': rebuilt}
    results = []
    try:
        for index, seed in trials:
            row = dict(seed=seed, index=index)
            if fresh:
                choice = ('frozen', 'corrected') if index % 2 == 0 else ('corrected', 'frozen')
                first, second = choice
                frames = []
                row[first] = play(arena, players[first], seed, frames)
                row[second] = play(arena, players[second], seed, frames, row[first]['actions'])
                shared = row[second]['shared_rgb_frames']
                if row[first]['optical'][:shared] != row[second]['optical'][:shared]:
                    raise ValueError('the matched camera prefixes disagree')
            else:
                row['corrected'] = play(arena, rebuilt, seed, [])
            results.append(row)
            print('rebuilt-fresh' if fresh else 'rebuilt-matched', seed,
                  {k: v['legal_return_frames'] for k, v in row.items() if k in players}, flush=True)
    finally:
        if frozen is not None:
            frozen.close()
        rebuilt.close()
        arena.close()
    return results


def run(previous, folder, workers):
    source = previous / 'flight-source.json.gz'
    before = previous / 'direct-receive-payload.json.gz'
    evidence = previous / 'direct-receive-payload-adapt.json.gz'
    old, learned = _read(before), _read(evidence)
    nominal = previous / 'factory-dry-catalog.npz'
    if _hash(nominal) != old['catalogue_sha256'] or _hash(before) != learned['payload_baseline_sha256']:
        raise ValueError('the rebuilt atlas used a different source or calibration')
    mass = learned['calibration']['choices']['payload']['selected_kg']
    args, seeds = old['arena'], old['point_seeds']
    fresh = np.random.default_rng(FRESH_SEED).integers(2**31, size=FRESH_COUNT).tolist()
    prior = previous_seeds(previous) | previous_seeds(folder) | set(seeds)
    if len(set(fresh)) != FRESH_COUNT or prior.intersection(fresh):
        raise ValueError('the fresh starts overlap an earlier assay')
    corrected = folder / ARCHIVE
    update = rebuild(nominal, corrected, mass)
    print('new atlas', update, flush=True)
    report = dict(schema='tt20-rebuilt-blade-atlas-v1', inferred_payload_kg=mass, arena=args,
                  rgb_calibration_sha256=_hash(evidence), frozen_payload_sha256=_hash(before),
                  prior_flight_sha256=_hash(source), reused_calibration_frames=24, new_target_practice_frames=0,
                  geometry=update, scored_candidates=32, fresh_seeds=fresh, matched_seeds=seeds)
    path = folder / RECEIPT
    _write(path, report)

    def sample(pool, starts, novel):
        schedule = list(enumerate(starts))
        tasks = [pool.submit(batch, nominal, corrected, source, args, mass, schedule[i::workers], novel)
                 for i in range(min(workers, len(starts)))]
        return sorted([row for task in tasks for row in task.result()], key=lambda row: row['index'])

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        matched = sample(pool, seeds, False)
        for row, historical in zip(matched, old['pairs'], strict=True):
            if row['seed'] != historical['seed']:
                raise ValueError('the old starting order changed')
            row['frozen'] = historical['gantry']
        report.update(matched=matched, matched_metrics=metrics(matched),
                      mass_only_returns=learned['loaded_adapted_returns'])
        _write(path, report)
        new = sample(pool, fresh, True)
    report.update(fresh=new, fresh_metrics=metrics(new), replayed_points=replay(args, (matched, new)))
    _write(path, report)
    print(report['matched_metrics'], report['fresh_metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        saved = _read(args.folder / RECEIPT)
        atlas_path = args.folder / ARCHIVE
        if _hash(atlas_path) != saved['geometry']['rebuilt_sha256']:
            raise ValueError('the rebuilt shortlist archive changed')
        print('replayed', replay(saved['arena'], (saved['matched'], saved['fresh'])))
    else:
        args.folder.mkdir(parents=True, exist_ok=True)
        run(args.previous, args.folder, args.workers)
