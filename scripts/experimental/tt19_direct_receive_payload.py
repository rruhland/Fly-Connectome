"""Replay the published direct receiving assay with a camera-mounted mass mismatch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import MODES, make_players, play, summarize, verify as verify_original
from .tt19_so101 import CAMERA_BODY, _vector, mujoco
from .tt19_view_trial import SO101ViewArena


STARTS = 12
MASS = .080
HALF_SIZE = np.array((.018, .011, .009))
OFFSET = np.array((.027, -.008, -.012))
INERTIA = MASS / 3 * (np.sum(HALF_SIZE ** 2) - HALF_SIZE ** 2)
MOUNT = 'learner_wrist_camera_mount'
ACCESSORY = 'tt19_camera_ballast'


class PayloadArena(SO101ViewArena):
    def _model_document(self):
        model = super()._model_document()
        camera = model.find(f".//body[@name='{CAMERA_BODY}']")
        ballast = ET.SubElement(camera, 'body', name=ACCESSORY, pos=_vector(OFFSET))
        ET.SubElement(ballast, 'inertial', pos='0 0 0', mass=str(MASS), diaginertia=_vector(INERTIA))
        return model


def assembly_inertia(arena):
    model, data = arena.model, arena.data
    camera = model.body(CAMERA_BODY).id
    base = data.xpos[camera]
    axes = data.xmat[camera].reshape(3, 3)
    names = (MOUNT, CAMERA_BODY) + ((ACCESSORY,) if isinstance(arena, PayloadArena) else ())
    pieces = [model.body(name).id for name in names]
    masses = model.body_mass[pieces]
    centers = (data.xipos[pieces] - base) @ axes
    com = np.average(centers, axis=0, weights=masses)
    tensors = []
    for index, position, mass in zip(pieces, centers, masses, strict=True):
        orientations = axes.T @ data.ximat[index].reshape(3, 3)
        local = orientations @ np.diag(model.body_inertia[index]) @ orientations.T
        offset = position - com
        tensors.append(local + mass * (np.eye(3) * (offset @ offset) - np.outer(offset, offset)))
    return dict(mass_kg=float(masses.sum()), com_camera_local_m=com.tolist(),
                inertia_about_com_camera_frame_kg_m2=np.sum(tensors, axis=0).tolist())


def batch(cache, source, arena_args, schedule):
    arena = PayloadArena(**arena_args)
    players = make_players(cache, source)
    pairs = []
    try:
        for index, seed in schedule:
            first, second = (0, 1) if index % 2 == 0 else (1, 0)
            frames = []
            left = play(arena, players[first], seed, frames)
            right = play(arena, players[second], seed, frames, left['actions'])
            shared = right['shared_rgb_frames']
            if left['optical'][:shared] != right['optical'][:shared]:
                raise ValueError('shared native pixels led to a different RGB track')
            rows = {MODES[first]: left, MODES[second]: right}
            pairs.append(dict(index=index, seed=seed, **rows))
            print(index + 1, seed, [rows[m]['result']['legal_returns'] for m in MODES], flush=True)
        dry_mass = [assembly_inertia(p.sim.arena) for p in players]
        if any(not np.isclose(info['mass_kg'], .035) for info in dry_mass):
            raise ValueError('the planner saw the test payload in its internal dynamics')
    finally:
        arena.close()
        for player in players:
            player.close()
    return pairs


def compare(nominal, payload):
    result = {}
    for mode in MODES:
        original = [bool(row[mode]['legal_return_frames']) for row in nominal]
        changed = [bool(row[mode]['legal_return_frames']) for row in payload]
        transitions = list(zip(original, changed, strict=True))
        result[mode] = dict(original=sum(original), payload=sum(changed),
                            lost=transitions.count((True, False)), gained=transitions.count((False, True)))
    return result


def verify(arena_args, pairs, earlier, mass_seed):
    original = SO101ViewArena(**arena_args)
    changed = PayloadArena(**arena_args)
    baseline, loaded = assembly_inertia(original), assembly_inertia(changed)
    accessory = changed.model.body(ACCESSORY)
    if not np.isclose(loaded['mass_kg'] - baseline['mass_kg'], MASS):
        raise ValueError('the accessory mass did not reach the compiled joint dynamics')
    np.testing.assert_allclose(accessory.inertia, INERTIA, rtol=1e-10)
    np.testing.assert_array_equal(accessory.pos, OFFSET)
    tape = next(row['gantry']['actions'] for row in earlier if row['seed'] == mass_seed)
    original.reset(seed=mass_seed, render=False)
    changed.reset(seed=mass_seed, render=False)
    deltas = []
    for action in tape:
        original.step(action, render=False)
        changed.step(action, render=False)
        deltas.append(float(np.linalg.norm(original.paddle - changed.paddle)))
        if original.terminated or original.truncated or changed.terminated or changed.truncated:
            break
    count = 0
    try:
        for row in pairs:
            for mode in MODES:
                point = row[mode]
                changed.reset(seed=row['seed'], server=-1, render=False)
                for action in point['actions']:
                    changed.step(action, render=False)
                if (point['result'] != changed.point_result() or
                        json.dumps(point['events']) != json.dumps(changed.events)):
                    raise ValueError(f"loaded-arm physical replay differs on {row['seed']} in {mode}")
                count += 1
    finally:
        original.close()
        changed.close()
    return dict(assembly_without_payload=baseline, assembly_with_payload=loaded,
                identical_tape=dict(seed=mass_seed, compared_frames=len(deltas), max_blade_delta_m=max(deltas)),
                replayed_points=count)


def run(folder, cache, workers):
    baseline_path = folder / 'direct-receive.json.gz'
    source = folder / 'flight-source.json.gz'
    baseline = _read(baseline_path)
    if _hash(cache) != baseline['catalogue_sha256'] or _hash(source) != baseline['gantry_flight_sha256']:
        raise ValueError('the published no-bank assay used different controller inputs')
    seeds = baseline['point_seeds'][:STARTS]
    old = baseline['pairs'][:STARTS]
    arena = baseline['arena']
    if len(old) != STARTS or seeds != [row['seed'] for row in old]:
        raise ValueError('the first 12 published starts are missing or reordered')
    schedule = list(enumerate(seeds))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        tasks = [pool.submit(batch, cache, source, arena, schedule[i::workers]) for i in range(workers)]
        pairs = sorted([row for task in tasks for row in task.result()], key=lambda row: row['index'])
    unchanged_replay = verify_original(dict(arena=arena, pairs=old))
    audit = verify(arena, pairs, old, seeds[0])
    receipt = dict(schema='tt19-receive-camera-payload-v1', point_seeds=seeds, pairs=pairs, arena=arena,
                   original_receipt_sha256=_hash(baseline_path), catalogue_sha256=_hash(cache),
                   source_priors_sha256=_hash(source), original_physical_replays=unchanged_replay,
                   payload=dict(mass_kg=MASS, camera_local_offset_m=OFFSET.tolist(),
                                half_size_m=HALF_SIZE.tolist(), inertia_kg_m2=INERTIA.tolist()),
                   inputs=dict(live=['RGB ball position', 'public control frame'],
                               supplied=['nominal dry catalogue', 'nominal camera and mount mass',
                                         'gantry flight law and impulse history or generic priors'],
                               evaluator_only=['payload model and state', 'held seeds', 'physical events',
                                               'prior unmodified outcomes']),
                   zero_target_practice=True, target_stroke_bank=None, matched_first_return=compare(old, pairs),
                   original_metrics=summarize(old), payload_metrics=summarize(pairs), audit=audit)
    _write(folder / 'direct-receive-payload.json.gz', receipt)
    print(receipt['matched_first_return'], receipt['audit'], flush=True)


def replay(folder):
    baseline_path = folder / 'direct-receive.json.gz'
    report = _read(folder / 'direct-receive-payload.json.gz')
    if _hash(baseline_path) != report['original_receipt_sha256']:
        raise ValueError('the nominal measurement receipt changed')
    baseline = _read(baseline_path)['pairs'][:STARTS]
    repeated = verify(report['arena'], report['pairs'], baseline, report['point_seeds'][0])
    if repeated != report['audit']:
        raise ValueError('the payload bench audit changed')
    return repeated['replayed_points']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('catalogue', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print('replayed', replay(args.folder))
    else:
        run(args.folder, args.catalogue.resolve(), args.workers)
