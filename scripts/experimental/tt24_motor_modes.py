"""Discover motor directions from retinal changes during a small block of babble."""

import numpy as np


def scene_pixels(rgb):
    """Retinal luminance of relatively unsaturated surfaces at coarse resolution."""
    rgb = rgb.astype(np.float32) / 255.
    level = rgb.mean(axis=-1)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    level *= np.clip(1. - 3. * chroma, 0., 1.)
    rows = level.shape[0] // 4 * 4
    cols = level.shape[1] // 4 * 4
    return level[:rows, :cols].reshape(rows // 4, 4, cols // 4, 4).mean(axis=(1, 3)).ravel()


def survey(arena, seed, *, regime=None, scale=.24, ticks=11, concentration=.5, effects=None):
    """Fit changes in coarse retina to executed joint displacements.

    All joints receive short, balanced exploratory motor commands. The motor
    covariance is derived from a Jacobian of measured luminance, with a floor
    so that directions that move the retina more continue to be explored.
    """
    axes = arena.action_size
    actions = np.r_[np.eye(axes), -np.eye(axes)] * scale
    rng = np.random.default_rng(seed)
    observations, displacements = [], []
    launches = rng.integers(1, 2 ** 31 - 1, size=2).tolist()
    options = {} if regime is None else {'regime': regime}
    for index, command in enumerate(actions[rng.permutation(len(actions))]):
        obs = arena.reset(seed=launches[index % len(launches)], split='practice', **options)
        initial = obs['joints']
        previous = scene_pixels(obs['rgb'])
        if effects is not None:
            effects.reset()
            effects.observe(obs['joint_velocity'], plastic=True)
        for tick in range(ticks):
            sample = tick % 3 == 2
            obs, _, done = arena.step(command, render=sample)
            if effects is not None:
                effects.issued(command)
                effects.observe(obs['joint_velocity'], plastic=True)
            if done:
                break
            if sample:
                displacements.append(obs['joints'] - initial)
                observations.append(scene_pixels(obs['rgb']) - previous)
    q = np.asarray(displacements)
    pixels = np.asarray(observations)
    if len(q) < axes:
        raise ValueError('motor babble did not provide enough paired images and joint observations')
    variation = np.linalg.solve(q.T @ q + .001 * np.eye(axes), q.T @ pixels)
    directions, strength, _ = np.linalg.svd(variation, full_matrices=False)
    spread = ((np.median(strength) + .1) / (strength + .1)) ** concentration
    spread = np.maximum(spread, .18)
    spread *= np.sqrt(axes / np.sum(spread ** 2))
    receipt = dict(launches=launches, attempts=len(actions), samples=len(q), response=variation.tolist(),
                   strengths=strength.tolist(), spreads=spread.tolist(), concentration=concentration)
    return directions * spread, receipt
