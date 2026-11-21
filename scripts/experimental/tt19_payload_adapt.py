"""Choose a camera-load hypothesis from one native RGB SO-101 movement."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt18_view import ImageColors
from .tt19_direct_receive import make_players, play, previous_seeds
from .tt19_direct_receive_payload import ACCESSORY, INERTIA, MASS, OFFSET, PayloadArena
from .tt19_model_rollout import DrySimulator
from .tt19_so101 import CAMERA_BODY, _vector, mujoco
from .tt19_view import detect_roles
from .tt19_view_trial import SO101ViewArena


SLATE = (0., .04, .08, .12)
PROBE_SEED = 9137
FRESH_SCHEDULE_SEED = 273574192
N_FRESH = 6
PROBE = np.repeat(np.array([
    [.26, -.54, .52, .78, .67, 0.],
    [-.48, .31, -.60, -.88, -.52, 0.],
    [.47, -.66, .63, .73, -.68, 0.],
    [-.55, .67, -.42, -.66, .62, 0.],
    [.44, .45, -.64, .39, .57, 0.],
    [-.14, -.23, .51, -.34, -.54, 0.],
]), 4, axis=0)


class HypothesisArena(PayloadArena):
    def __init__(self, hypothesis, **kwargs):
        self.hypothesis = hypothesis
        super().__init__(**kwargs)

    def _model_document(self):
        doc = super()._model_document()
        camera = doc.find(f".//body[@name='{CAMERA_BODY}']")
        component = camera.find(f"body[@name='{ACCESSORY}']")
        if self.hypothesis == 0.:
            camera.remove(component)
        else:
            inertial = component.find('inertial')
            inertial.set('mass', str(self.hypothesis))
            inertial.set('diaginertia', _vector(INERTIA * self.hypothesis / MASS))
        return doc


def blade_pixels(images):
    seen = detect_roles({k: ImageColors(im) for k, im in images.items()})
    return np.asarray([seen[role][view].center if view in seen[role] else [np.nan, np.nan]
                       for view in ('left', 'right') for role in ('tip', 'blade', 0, 1, 2, 3)])


def dry_trace(arena):
    arena.reset(seed=PROBE_SEED, render=False)
    geom = arena.model.geom('ball')
    geom.contype[0] = geom.conaffinity[0] = 0
    arena.ball[:] = [0., 0., -25.]
    arena.velocity[:] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    trace = [blade_pixels(arena.render_cameras())]
    ports = arena._arm_ids[1][2]
    steps = int(round(arena.control_dt / arena.model.opt.timestep))
    for row in PROBE:
        for _ in range(steps):
            arena.data.ctrl[ports] = np.clip(arena.data.ctrl[ports] + row * arena._joint_increments[ports],
                                            arena._joint_low[ports], arena._joint_high[ports])
            mujoco.mj_step(arena.model, arena.data)
        trace.append(blade_pixels(arena.render_cameras()))
    return np.stack(trace)


def calibration(arena_args):
    def capture(factory):
        instance = factory(**dict(arena_args, view_mode='fixed'))
        try:
            return dry_trace(instance)
        finally:
            instance.close()

    templates = [capture(lambda **kwargs: HypothesisArena(mass, **kwargs)) for mass in SLATE]
    measured = {'nominal': capture(SO101ViewArena), 'payload': capture(PayloadArena)}
    choices = {}
    for label, observations in measured.items():
        losses = []
        found = np.isfinite(observations).all(axis=-1)
        for hypothesis in templates:
            valid = found & np.isfinite(hypothesis).all(axis=-1)
            rms = float(np.sqrt(np.square(observations[valid] - hypothesis[valid]).sum(axis=-1).mean()))
            losses.append(rms + .60 * (1. - valid.sum() / found.sum()))
        order = np.argsort(losses)
        margin = float(losses[order[1]] - losses[order[0]])
        selected = SLATE[order[0]] if margin >= .02 and found.sum() >= 48 else 0.
        choices[label] = dict(scores_px=losses, observed_blobs=int(found.sum()), margin_px=margin,
                              selected_kg=selected, charged_frames=len(PROBE))
        print('identified', label, choices[label], flush=True)

    def values(trace):
        return np.where(np.isfinite(trace), trace, None).tolist()

    return dict(probe_seed=PROBE_SEED, commands=PROBE.tolist(), slate_kg=list(SLATE), minimum_margin_px=.02,
                features=['tip', 'blade', 'rim0', 'rim1', 'rim2', 'rim3'], views=['left', 'right'], choices=choices,
                templates=[values(trace) for trace in templates], observed={k: values(v) for k, v in measured.items()})


class WeightedSimulator(DrySimulator):
    """Apply the chosen rigid accessory inertia only to the independent planner."""

    def __init__(self, law, mass):
        super().__init__(law)
        if mass == 0.:
            return
        camera = self.model.body(CAMERA_BODY)
        old_mass = float(camera.mass[0])
        initial_com = camera.ipos.copy()
        total = old_mass + mass
        com = (old_mass * initial_com + mass * OFFSET) / total
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, camera.iquat)
        basis = rotation.reshape(3, 3)

        def shifted(point):
            delta = point - com
            return np.eye(3) * (delta @ delta) - np.outer(delta, delta)

        tensor = (basis @ np.diag(camera.inertia) @ basis.T + np.diag(INERTIA * mass / MASS)
                  + old_mass * shifted(initial_com) + mass * shifted(OFFSET))
        principal, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0.:
            axes[:, 0] *= -1.
        camera.mass[:] = total
        camera.inertia[:] = principal
        camera.ipos[:] = com
        mujoco.mju_mat2Quat(camera.iquat, np.ascontiguousarray(axes).ravel())
        mujoco.mj_setConst(self.model, self.data)
        self.arena.reset(render=False)
        self.arena.ball[:] = (1.7, 2., .6)
        self.arena.velocity[:] = 0.
        mujoco.mj_step(self.model, self.data, nstep=8 * self.timestep)
        mujoco.mj_getState(self.model, self.data, self.base, self.state_flags)
        self.base[0] = 0.


def batch(folder, cache, arena_args, selected, seeds, fresh):
    arena = PayloadArena(**arena_args)
    player, unused = make_players(cache, folder / 'flight-source.json.gz')
    unused.close()
    nominal, adapted = player.sim, WeightedSimulator(player.law, selected)
    rows = []
    try:
        for seed in seeds:
            row = dict(seed=seed)
            if fresh:
                player.sim = nominal
                images = []
                row['frozen'] = play(arena, player, seed, images)
                player.sim = adapted
                row['adapted'] = play(arena, player, seed, images, row['frozen']['actions'])
            else:
                player.sim = adapted
                row['adapted'] = play(arena, player, seed, [])
            rows.append(row)
            print('fresh' if fresh else 'matched', seed, {label: row[label]['legal_return_frames'] for label in row
                                                          if label != 'seed'}, flush=True)
    finally:
        nominal.close()
        adapted.close()
        arena.close()
    return rows


def replay(arena_args, sets):
    arena = PayloadArena(**arena_args)
    count = 0
    try:
        for group in sets:
            for item in group:
                for point in (item[label] for label in ('frozen', 'adapted') if label in item):
                    arena.reset(seed=point['seed'], server=-1, render=False)
                    for row in point['actions']:
                        arena.step(row, render=False)
                    if arena.point_result() != point['result'] or json.dumps(arena.events) != json.dumps(point['events']):
                        raise ValueError('identified-payload action tape changed its physical result')
                    count += 1
    finally:
        arena.close()
    return count


def run(folder, cache, workers):
    original_path = folder / 'direct-receive-payload.json.gz'
    earlier_path = folder / 'direct-receive.json.gz'
    original, earlier = _read(original_path), _read(earlier_path)
    atlas = _hash(cache)
    if atlas != original['catalogue_sha256'] or _hash(earlier_path) != original['original_receipt_sha256']:
        raise ValueError('baseline input differs from the published mass assay')
    arena_args = original['arena']
    fit = calibration(arena_args)
    selected = fit['choices']['payload']['selected_kg']
    seeds = original['point_seeds']
    fresh = np.random.default_rng(FRESH_SCHEDULE_SEED).integers(2**31, size=N_FRESH).tolist()
    excluded = previous_seeds(folder) | set(earlier['point_seeds']) | {PROBE_SEED}
    if len(set(fresh)) != N_FRESH or set(fresh) & excluded:
        raise ValueError('a fresh start repeats an earlier point or the calibration seed')
    path = folder / 'direct-receive-payload-adapt.json.gz'
    receipt = dict(schema='tt19-payload-adapt-v1', arena=arena_args, nominal_atlas_sha256=atlas,
                   payload_baseline_sha256=_hash(original_path), original_arm_sha256=_hash(earlier_path),
                   calibration=fit, fresh_seed=FRESH_SCHEDULE_SEED, prespecified_fresh_starts=fresh,
                   slots='gantry priors and nominal geometry catalogue', target_stroke_bank=None)
    _write(path, receipt)

    def run_points(pool, starts, unseen):
        tasks = [pool.submit(batch, folder, cache, arena_args, selected, starts[i::workers], unseen)
                 for i in range(min(workers, len(starts)))]
        by_seed = {row['seed']: row for task in tasks for row in task.result()}
        return [by_seed[seed] for seed in starts]

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        matched = run_points(pool, seeds, False)
        recovered = [row['seed'] for row, old, bare in zip(matched, original['pairs'], earlier['pairs'][:len(seeds)], strict=True)
                     if row['adapted']['legal_return_frames'] and not old['gantry']['legal_return_frames']
                     and bare['gantry']['legal_return_frames']]
        previous = sum(bool(row['gantry']['legal_return_frames']) for row in original['pairs'])
        after = sum(bool(row['adapted']['legal_return_frames']) for row in matched)
        follow = run_points(pool, fresh, True) if recovered and after > previous else []
    receipt.update(matched=matched, fresh=follow, loaded_frozen_returns=previous, loaded_adapted_returns=after,
                   recovered_prior_losses=recovered, replayed_points=replay(arena_args, (matched, follow)))
    _write(path, receipt)
    print('frozen', previous, 'adapted', after, 'recovered', recovered, 'fresh', len(follow), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('catalogue', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.folder, args.catalogue.resolve(), args.workers)
