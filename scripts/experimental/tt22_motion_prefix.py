"""Use a synchronized, unlabelled stereo motion track during the initial no-command interval."""

from hashlib import sha256

import numpy as np
from scipy import ndimage

from .tt18_view import triangulate_views
from .tt20_natural import NaturalViewTransport


def candidates(frames):
    # The learner has received the full short prefix before selecting an action.
    values = np.stack(frames).astype(np.int16)
    background = np.median(values, axis=0)
    results = []
    for frame in values:
        diff = np.max(np.abs(frame - background), axis=2)
        labels, n = ndimage.label(diff > 28, structure=np.ones((3, 3), int))
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        good = np.flatnonzero((sizes >= 2) & (sizes < 0.05 * labels.size))
        good = good[good != 0]
        centers = ndimage.center_of_mass(diff, labels, good) if len(good) else []
        results.append(np.asarray(centers).reshape(-1, 2)[:, ::-1])
    return results


def paths(frames):
    available = candidates(frames)
    if len(available) < 5:
        return []
    found = []
    # Pair the last two sightings; earlier images disambiguate transients and disappearing ghosts.
    for a in available[-1]:
        for b in available[-2]:
            if not 0.7 < np.linalg.norm(a - b) < 32:
                continue
            seq = [a, b]
            for points in reversed(available[:-2]):
                if not len(points):
                    break
                expected = 2 * seq[-1] - seq[-2]
                dist = np.linalg.norm(points - expected, axis=1)
                j = np.argmin(dist)
                if dist[j] > 5.0:
                    break
                seq.append(points[j])
            if len(seq) < 4:
                continue
            pixels = np.array(seq[::-1])
            deltas = np.diff(pixels, axis=0)
            movement = np.linalg.norm(pixels[-1] - pixels[0]) / (len(pixels) - 1)
            jerk = np.median(np.linalg.norm(np.diff(deltas, axis=0), axis=1))
            score = (len(pixels) - 3) * movement / (1 + jerk)
            found.append((score, pixels))
    return sorted(found, key=lambda p: -p[0])[:8]


def prefix(arena, seed, lane):
    from .tt22_joint_practice import READS

    arena.lane = lane
    arena.reset(seed=seed, server=-1, render=False)
    calibration = NaturalViewTransport(
        "fixed", lens_fov=arena.lens_intrinsics(), flight=False, associate=False
    )
    cameras = {}
    images = {"left": [], "right": []}
    digest = sha256()
    for t in range(READS):
        frames = arena.render_cameras()
        cameras = calibration.observe(frames, t, cameras)
        for key in images:
            digest.update(key.encode())
            digest.update(frames[key])
            images[key].append(frames[key])
        arena.step(np.zeros(6), render=False)
    best = None
    if len(cameras) == 2:
        for lscore, ll in paths(images["left"]):
            for rscore, rr in paths(images["right"]):
                length = min(len(ll), len(rr))
                trajectory = [
                    triangulate_views(cameras, {"left": l, "right": r}, max_error=2.5)
                    for l, r in zip(ll[-length:], rr[-length:])
                ]
                if any(x is None for x in trajectory):
                    continue
                ts = np.arange(1 - length, 1)
                pos = np.asarray(trajectory)
                design = np.column_stack((np.ones(length), ts))
                intercept, slope = np.linalg.lstsq(design, pos, rcond=None)[0]
                error = np.mean(np.linalg.norm(design @ np.stack((intercept, slope)) - pos, axis=1))
                score = np.sqrt(lscore * rscore) / (1 + error / 0.015)
                if best is None or score > best[0]:
                    last = pos[-4:]
                    estimate = np.linalg.lstsq(design[-4:], last, rcond=None)[0]
                    best = score, estimate, trajectory
    # Body origin comes from the robot's own command coordinate frame.
    trajectory = None if best is None else np.array(best[2]).tolist()
    mover = None if best is None or abs(best[1][1, 1]) < 0.001 else best[1].tolist()
    return dict(seed=seed, lane=lane, ready=None, mover=mover, rgb=digest.hexdigest(), track=trajectory)
