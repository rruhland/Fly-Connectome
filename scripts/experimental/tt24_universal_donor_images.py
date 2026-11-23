"""Fit source feedback consistency from actual donor wrist images during contacts."""

import argparse
from pathlib import Path

import numpy as np

from .tt19_so101 import WRIST_CAMERA
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState, _unit_camera_ray
from .tt24_donor_trial import DESIGNS, DonorBounce, STOCK_HELD_RNG, _paired, _seeds, _summary
from .tt24_public_body import FOCAL, FUSED_OUTPUT, load_public
from .tt24_universal_noise import FRESH, ROOT, features
from .tt24_universal_trial import CHECKPOINT, UniversalNative, _store


LEARN = _seeds(638778472, len(DESIGNS), 6)
DEVELOPMENT = _seeds(2006364, len(DESIGNS), 3)
CONFIRMATION = _seeds(STOCK_HELD_RNG + 43872, 1, 16)[0]


class KnownSourceRig:
    """Keep a donor camera pose in the right frame for measuring optical timing."""

    def __init__(self, original):
        self.default, self.radius = original.default, original.radius
        self.image_height = original.image_height
        self.pose = np.zeros(17)

    def body(self, _joints):
        return self.pose

    def ball(self, ellipse, body):
        if ellipse is None:
            return None
        return body[:3] + body[3:12].reshape(3, 3) @ (_unit_camera_ray(ellipse, FOCAL) * self.radius)


def imaging_donor(design):
    arena = DonorBounce(design)
    arena.width, arena.height = 192, 144
    arena._model.vis.global_.offwidth = arena.width
    arena._model.vis.global_.offheight = arena.height
    return arena


def observe(arena, native, weights, seed, *, clean_drive):
    obs = arena.reset(seed=seed, split='held', regime='changed', render=False)
    truth = CleanState(arena)
    public = load_public(FUSED_OUTPUT)
    lens = KnownSourceRig(public.decoder)
    public.decoder = lens
    camera = arena._data.camera(WRIST_CAMERA)
    issued, action = np.zeros(5), np.zeros(5)
    actual, measured = [], []
    for _ in range(arena.max_steps):
        state = truth.observe(obs)
        picture = arena.render_wrist()
        lens.pose[:] = np.r_[camera.xpos, camera.xmat, truth.blade_home + state[6:9], state[9:]]
        estimated = np.ascontiguousarray(public.observe(obs | dict(rgb=picture)), dtype=float)
        actual.append(state)
        measured.append(estimated)
        frame = state if clean_drive else estimated
        native.command(frame.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
        obs, _, done = arena.step(action, render=False)
        if done:
            break
    judge = arena.judge_metrics()
    return np.asarray(actual), np.asarray(measured), judge


def fit(clean, visual, original, anchor, transform):
    exact = features(np.concatenate(clean), transform)
    realistic = features(np.concatenate(visual), transform)
    targets = exact @ original.T
    gram = realistic.T @ realistic + .55 * exact.T @ exact
    penalty = np.maximum(np.diag(gram) * .15, 6.)
    rhs = (realistic.T @ (targets-realistic @ anchor.T)
           + .55 * exact.T @ (targets-exact @ anchor.T))
    return anchor + np.linalg.solve(gram + np.diag(penalty), rhs).T


def _confirm(args):
    source = args.out / 'universal-donor-images.npz'
    previous = _read(args.out / 'universal-donor-images.json.gz')
    assert previous['checkpoint_sha256'] == _hash(source)
    assert not set(CONFIRMATION) & set(previous['held_seeds'])
    library = UniversalNative()
    arena = BounceArena(view='wrist', width=192, height=144)
    phases = (('frozen', args.source, ('public', 'clean', 'hold')),
              ('public_residual', args.prior_robust, ('public',)),
              ('donor_images', source, ('public', 'clean')))
    output = dict(schema='tt24-universal-consistency-confirmation-1', held_seeds=CONFIRMATION,
                  source_hashes={name: _hash(path) for name, path, _ in phases},
                  target_training=False, observer_sha256=_hash(FUSED_OUTPUT), held={}, summary={})
    try:
        for name, path, modes in phases:
            weights = library.load_checkpoint(path)
            data = _paired(arena, library, weights, CONFIRMATION, modes, 'standard')
            output['held'][name] = data
            output['summary'][name] = _summary(data)
            print('donor images fresh confirmation', name, output['summary'][name], flush=True)
            _write(args.out / 'universal-donor-images-confirm.json.gz', output)
    finally:
        arena.close()
        library.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=CHECKPOINT)
    parser.add_argument('--prior-robust', type=Path, default=ROOT/'universal-noise.npz')
    parser.add_argument('--previous', type=Path, default=ROOT/'universal-noise.json.gz')
    parser.add_argument('--out', type=Path, default=ROOT)
    parser.add_argument('--confirm', action='store_true', help='run the exported checkpoints on a disjoint stock cohort')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.confirm:
        _confirm(args)
        return
    native = UniversalNative()
    donors = [imaging_donor(design) for design in DESIGNS]
    try:
        weights = native.load_checkpoint(args.source)
        hashes = {key: _hash(path) for key, path in (('frozen', args.source), ('anchor', args.prior_robust))}
        with np.load(args.source, allow_pickle=False) as saved:
            transform = {key: saved[key] for key in ('center', 'scale', 'basis')}
        with np.load(args.prior_robust, allow_pickle=False) as saved:
            assert all(np.array_equal(saved[key], transform[key]) for key in transform)
            anchor = saved['weights']
        x, y, calibration = [], [], []
        for donor, seeds in zip(donors, LEARN, strict=True):
            for seed in seeds:
                clean, observed, judge = observe(donor, native, weights, seed, clean_drive=True)
                x.append(clean)
                y.append(observed)
                calibration.append(dict(seed=seed, frames=len(clean), donor_streak=judge['longest_consecutive_rebounds']))
                print('actual donor image exposure', seed, len(clean), flush=True)
        learned = fit(x, y, weights, anchor, transform)
        checkpoint = args.out/'universal-donor-images.npz'
        _store(checkpoint, learned, transform)

        diagnostics = {}
        for name, actor in (('frozen', weights), ('public_residual', anchor), ('donor_images', learned)):
            rows = []
            for donor, seeds in zip(donors, DEVELOPMENT, strict=True):
                rows.append([observe(donor, native, actor, seed, clean_drive=False)[2]['longest_consecutive_rebounds']
                             for seed in seeds])
            diagnostics[name] = dict(rows=rows, mean=float(np.mean(rows)))
            print('actual donor image validation', name, diagnostics[name], flush=True)

        baseline = _read(args.previous)
        assert baseline['held_seeds'] == FRESH and baseline['source_sha256'] == hashes['frozen']
        assert baseline['learned_sha256'] == hashes['anchor']
        held_arena = BounceArena(view='wrist', width=192, height=144)
        try:
            held = _paired(held_arena, native, learned, FRESH, ('public', 'clean'), 'standard')
        finally:
            held_arena.close()
        outcome = _summary(held)
        assert _hash(args.source) == hashes['frozen'] and _hash(args.prior_robust) == hashes['anchor']
        report = dict(schema='tt24-universal-consistency-donor-images-1', hashes=hashes,
                      frozen_target_observer=_hash(FUSED_OUTPUT), checkpoint_sha256=_hash(checkpoint),
                      source_pose_for_images='known exact simulator camera and blade pose on donors only',
                      training='closed form fit of self-consistency on actual donor images and clean donor observation',
                      motor_labels_from_other_policy=False, stock_training=False, source_images=calibration,
                      donor_validation=diagnostics, donor_validation_seeds=DEVELOPMENT, held_seeds=FRESH,
                      paired_prior_outcomes=baseline['summary'], held_summary=outcome,
                      held=held, same_stock_evaluation_as=str(args.previous))
        _write(args.out/'universal-donor-images.json.gz', report)
        print('donor images held', outcome, flush=True)
    finally:
        for donor in donors:
            donor.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
