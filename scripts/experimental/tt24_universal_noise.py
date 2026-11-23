"""Make the frozen generic feedback less sensitive to public visual errors."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState
from .tt24_donor_trial import DESIGNS, DonorBounce, STOCK_HELD_RNG, _paired, _seeds, _summary
from .tt24_metric_stereo import triangulate
from .tt24_public_body import FUSED_OUTPUT, blade_profile, disk_pose, load_public, probe, sight, stationary, stock
from .tt24_universal_trial import CHANNELS, CHECKPOINT, QUADRATIC, UniversalNative, _store
from .tt24_world_body_play import advance_body


ROOT = Path('artifacts/tt24')
OPTICAL_SEEDS = tuple(range(260_001, 260_011))
TRAIN = _seeds(48218730, len(DESIGNS), 8)
DEVELOPMENT = _seeds(19467128, len(DESIGNS), 4)
FRESH = _seeds(STOCK_HELD_RNG + 22611, 1, 16)[0]


def _visible_profile(arena):
    left, right = arena.fixed_images()
    images = (left, right)
    return np.asarray([sight(image) for image in images]), np.asarray([blade_profile(image) for image in images])


def public_probe_clips():
    """Use a separate collision-free imaging bench and ball-independent commands."""
    arena = stock()
    arena._model.geom_contype[arena._ball_geom] = 0
    arena._model.geom_conaffinity[arena._ball_geom] = 0
    clips = []
    try:
        for index, seed in enumerate(OPTICAL_SEEDS):
            obs = arena.reset(seed=seed, split='practice', render=False)
            if index % 2:
                # An optical-bench release samples ascending and descending views.
                # Its commanded setting is never used as a regression label.
                arena._ball_position[2] = .935
                arena._ball_velocity[2] = 1.30 + .12 * (index % 3)
            if index == 0:
                eyes = stationary(arena)
            estimate = load_public(FUSED_OUTPUT)
            home = obs['joints'].copy()
            retinal, missing, ball, paddle = [], [], [], []
            for tick in range(44):
                fixed_ball, fixed_blade = _visible_profile(arena)
                public = obs | dict(rgb=arena.render_wrist())
                before = estimate.gaps
                retinal.append(estimate.observe(public))
                missing.append(estimate.gaps != before)
                ball.append(fixed_ball)
                paddle.append(fixed_blade)
                obs = advance_body(arena, probe(seed, tick, home, obs['joints']))
            trips, _, _ = triangulate(eyes, np.asarray(ball)[:, :, :2], tolerance=.017)
            blade = np.full((len(paddle), 5), np.nan)
            last = None
            for tick in range(0, len(paddle), 2):
                fitted = disk_pose(eyes, paddle[tick], last, fixed_size=(.076, .006))
                if fitted is None:
                    continue
                last = fitted[0]
                direction = np.r_[last[3:5], 1.]
                blade[tick] = np.r_[last[:3], direction[:2] / np.linalg.norm(direction)]
            known = np.flatnonzero(np.isfinite(blade).all(axis=1))
            if len(known) < 3:
                continue
            for column in range(blade.shape[1]):
                blade[:, column] = np.interp(np.arange(len(blade)), known, blade[known, column])
            origin = blade[0, :3].copy()
            velocity = np.full_like(trips, np.nan)
            velocity[1:-1] = (trips[2:] - trips[:-2]) * 30.
            public_reference = np.c_[trips - origin, velocity, blade[:, :3] - origin, blade[:, 3:]]
            retained = np.isfinite(public_reference).all(axis=1)
            retained[:known[0]] = False
            retained[known[-1] + 1:] = False
            clips.append(dict(seed=seed, reference=public_reference, observed=np.asarray(retinal),
                              valid=retained, missing=np.asarray(missing)))
            print('unscored stereo calibration', seed, 'valid', int(retained.sum()),
                  'wrist gaps', int(np.asarray(missing)[retained].sum()), flush=True)
    finally:
        arena.close()
    return clips


def fit_noise(clips):
    trajectories, biases = [], []
    for clip in clips:
        valid, missed = clip['valid'], clip['missing']
        keep = valid & ~missed
        if keep.sum() < 6:
            continue
        dist = clip['observed'] - clip['reference']
        mean = np.median(dist[keep], axis=0)
        biases.append(mean)
        visible_pair = valid[1:] & valid[:-1]
        delta = clip['reference'][:-1] - clip['reference'][1:]
        trajectories.append(dict(error=dist[1:]-mean, delta=delta, visible=visible_pair, missing=missed[1:]))
    if not trajectories:
        raise ValueError('no disjoint public calibration tracks with visible stereo and wrist measurements')
    okay = np.concatenate([seq['visible'] for seq in trajectories])
    error = np.concatenate([seq['error'] for seq in trajectories])[okay]
    delta = np.concatenate([seq['delta'] for seq in trajectories])[okay]
    dropout = np.concatenate([seq['missing'] for seq in trajectories])[okay]
    size = np.maximum(np.quantile(abs(error), .90, axis=0), .002)
    motion_extent = np.maximum(np.quantile(abs(delta), .90, axis=0), .002)
    delta = np.clip(delta, -2.8 * motion_extent, 2.8 * motion_extent)
    error = np.clip(error, -2.8 * size, 2.8 * size)

    def slope(members):
        inputs, outputs = delta[members], error[members]
        return np.clip(np.sum(inputs * outputs, axis=0) /
                       (np.sum(inputs**2, axis=0) + 12 * motion_extent**2), 0., .95)

    lag = slope(~dropout)
    missing_lag = np.maximum(lag, slope(dropout))
    previous, current = [], []
    transitions = np.zeros((2, 2))
    for seq in trajectories:
        visible, active = seq['visible'], seq['missing']
        dev = np.clip(seq['error'] - np.where(active[:, None], missing_lag, lag) * seq['delta'], -3 * size, 3 * size)
        contiguous = visible[1:] & visible[:-1]
        previous.extend(dev[:-1][contiguous])
        current.extend(dev[1:][contiguous])
        np.add.at(transitions, (active[:-1][contiguous].astype(int), active[1:][contiguous].astype(int)), 1.)
    pre, post = np.asarray(previous), np.asarray(current)
    rho = np.clip(np.sum(pre*post, axis=0) / (np.sum(pre**2, axis=0) + .025*size**2), .08, .85)
    innovations = np.clip(post-rho*pre, -2.7*size, 2.7*size)
    transition = (transitions[:, 1] + 1.) / (transitions.sum(axis=1) + 4.)
    return dict(lag=lag, missing_lag=missing_lag, rho=rho, bias=np.asarray(biases), innovations=innovations,
                transition=transition, scale=size, visible_error_rms=np.sqrt(np.mean(error[~dropout]**2, axis=0)),
                exposure_rows=int(okay.sum()), missing_rows=int(dropout.sum()))


def corrupt(raw, noise, rng):
    error = np.zeros(CHANNELS)
    bias = noise['bias'][rng.integers(len(noise['bias']))]
    draws = noise['innovations'][rng.integers(len(noise['innovations']), size=len(raw))]
    amplitude = rng.uniform(.55, 1.05)
    mask = rng.random(len(raw))
    missing = False
    result = raw.copy()
    for index in range(len(raw)):
        missing = mask[index] < noise['transition'][int(missing)]
        echo = raw[index - 1] - raw[index] if index else np.zeros(CHANNELS)
        delay = noise['missing_lag'] if missing else noise['lag']
        error = noise['rho'] * error + draws[index]
        result[index] += amplitude * (delay * echo + bias + error)
    return result


def _physical_sequence(arena, native, weights, seed, noise=None):
    obs = arena.reset(seed=seed, split='held', regime='changed', render=False)
    observer = CleanState(arena)
    issued, command = np.zeros(5), np.zeros(5)
    inputs, completed = [], 0
    rng = np.random.default_rng(seed + 2227008)
    memory, last = np.zeros(CHANNELS), None
    missing = False
    if noise is not None:
        bias = noise['bias'][rng.integers(len(noise['bias']))]
        draws = noise['innovations'][rng.integers(len(noise['innovations']), size=arena.max_steps)]
        amplitude = rng.uniform(.55, 1.05)
        masks = rng.random(arena.max_steps)
    for index in range(arena.max_steps):
        true_input = np.ascontiguousarray(observer.observe(obs), dtype=float)
        inputs.append(true_input)
        current = true_input
        if noise is not None:
            missing = masks[index] < noise['transition'][int(missing)]
            delay = noise['missing_lag'] if missing else noise['lag']
            memory = noise['rho'] * memory + draws[index]
            echo = last - true_input if last is not None else np.zeros(CHANNELS)
            current = true_input + amplitude * (delay * echo + bias + memory)
        native.command(current.ctypes.data, weights.ctypes.data, issued.ctypes.data, command.ctypes.data)
        last = true_input
        obs, _, done = arena.step(command, render=False)
        completed += 1
        if done:
            break
    streak = arena.judge_metrics()['longest_consecutive_rebounds']
    return np.asarray(inputs), (streak, completed)


def features(x, transform):
    normalized = np.clip(np.arcsinh((x - transform['center']) / transform['scale']), -3.4, 3.4)
    rotated = np.clip(normalized @ transform['basis'].T, -3., 3.)
    i, j = np.asarray(QUADRATIC).T
    products = np.clip(.75 * rotated[:, i] * rotated[:, j], -3., 3.)
    return np.c_[np.ones(len(x)), rotated, products]


def fit_consistency(clips, weights, transform, noise):
    clean = features(np.concatenate(clips), transform)
    target = clean @ weights.T
    rng = np.random.default_rng(7735124)
    gram = .45 * clean.T @ clean
    direction = np.zeros_like(weights.T)
    for _ in range(12):
        augmented = features(np.concatenate([corrupt(clip, noise, rng) for clip in clips]), transform)
        gram += augmented.T @ augmented / 12
        direction += augmented.T @ (target - augmented @ weights.T) / 12
    penalty = np.maximum(np.diag(gram) * .11, 4.)
    delta = np.linalg.solve(gram + np.diag(penalty), direction).T
    return weights + delta


def _native_donor_validation(worlds, native, candidates, noise):
    diagnostics = {}
    for name, model in candidates.items():
        clean, uncertain = [], []
        for arena, seeds in zip(worlds, DEVELOPMENT, strict=True):
            clean.append([_physical_sequence(arena, native, model, seed)[1][0] for seed in seeds])
            uncertain.append([_physical_sequence(arena, native, model, seed, noise)[1][0] for seed in seeds])
        utility = .30 * np.mean(clean) + .70 * np.mean(uncertain)
        diagnostics[name] = dict(clean=clean, injected_noise=uncertain, utility=float(utility))
        print('universal noise donor validation', name, diagnostics[name], flush=True)
    selected = max((name for name in diagnostics if name != 'frozen'), key=lambda name: diagnostics[name]['utility'])
    return selected, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=CHECKPOINT)
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    native = UniversalNative()
    worlds = [DonorBounce(design) for design in DESIGNS]
    try:
        frozen = native.load_checkpoint(args.source)
        with np.load(args.source, allow_pickle=False) as saved:
            transform = {key: saved[key] for key in ('center', 'scale', 'basis')}
        source_hash = _hash(args.source)
        observations = public_probe_clips()
        noise = fit_noise(observations)
        sequences, exposure = [], []
        for arena, seeds in zip(worlds, TRAIN, strict=True):
            for seed in seeds:
                state, outcome = _physical_sequence(arena, native, frozen, seed)
                sequences.append(state)
                exposure.append(dict(seed=seed, decisions=outcome[1]))
        robust = fit_consistency(sequences, frozen, transform, noise)
        halfway = .5 * (frozen + robust)
        variants = dict(frozen=frozen, full=robust, halfway=halfway)
        selected, development = _native_donor_validation(worlds, native, variants, noise)
        learned = variants[selected]
        fit = args.out / 'universal-noise.npz'
        _store(fit, learned, transform)
        out = dict(schema='tt24-universal-sensor-consistency-1', stock_public_calibration_is_unscored=True,
                   stock_game_reward_during_learning=False, stock_numeric_scene_labels=False,
                   training_signal='own frozen donor readout on clean donor inputs, regularized consistency',
                   externally_supplied_motor_actions=False, source_sha256=source_hash, learned_sha256=_hash(fit),
                   optical_seeds=OPTICAL_SEEDS,
                   optical_count={str(clip['seed']): int(clip['valid'].sum()) for clip in observations},
                   noise={key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in noise.items() if key != 'innovations'},
                   learn_exposure=exposure, donor_validation=development, donor_development_seeds=DEVELOPMENT,
                   selected=selected, held_seeds=FRESH, held={}, summary={})
        _write(args.out / 'universal-noise.json.gz', out)
        arena = BounceArena(view='wrist', width=192, height=144)
        try:
            for name, pupil, modes in (('frozen', frozen, ('public', 'clean', 'hold')),
                                        ('consistent', learned, ('public', 'clean'))):
                rows = _paired(arena, native, pupil, FRESH, modes, 'standard')
                out['held'][name] = rows
                out['summary'][name] = _summary(rows)
                print('universal noise held', name, out['summary'][name], flush=True)
                _write(args.out / 'universal-noise.json.gz', out)
        finally:
            arena.close()
        assert _hash(args.source) == source_hash
    finally:
        for world in worlds:
            world.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
