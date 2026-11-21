"""Fit a camera from the ordinary painted lines of a tabletop and its net."""

from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt18_view import CameraFit


_THETA = np.arange(180.) * np.pi / 180.
_ANGLES = np.array((np.cos(_THETA), np.sin(_THETA)))
_CONNECT = np.ones((3, 3), dtype=bool)


def _teal(r, g, b):
    selected = ((g - r > 19) & (b - g > 2) & (100 * r < 62 * g) &
                (100 * g > 75 * b) & (100 * g < 97 * b) & (g > 31))
    labels, count = ndimage.label(selected, structure=_CONNECT)
    if count == 0:
        return selected
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    keep = areas > max(8., .020 * areas.max())
    return keep[labels]


@lru_cache(maxsize=16)
def _court_geometry(bounds):
    w, length = bounds
    rail, end = w * (.752 / .7625), length * (1.359 / 1.37)
    across = np.linspace(-.95, .95, 21)
    down = np.linspace(-.95, .95, 27)
    seam = ([(x, y * end, .001) for y in down] for x in (-rail, rail, 0.))
    cross = ([(x * rail, y, .001) for x in across] for y in (-end, end))
    net = [(x * w * (0.915 / .7625), 0., .1485) for x in across]
    lines = np.asarray([point for line in (*seam, *cross, net) for point in line])
    surface = np.asarray([(x * w, y * length, 0.) for x in np.linspace(-.87, .87, 9)
                          for y in np.linspace(-.88, .88, 15) if abs(y) > .075])
    return rail, end, lines, surface


def _straight_segments(points, width, height):
    if len(points) < 35:
        return ()
    radius = int(np.ceil(np.hypot(width, height))) + 1
    angles = np.arange(len(_THETA))[None, :] * (2 * radius + 1)
    coordinates = np.rint(points @ _ANGLES).astype(int) + radius + angles
    counts = np.bincount(coordinates.ravel(), minlength=len(_THETA) * (2 * radius + 1)).reshape(len(_THETA), -1)
    peaks = np.argmax(counts, axis=1)
    supports = counts[np.arange(len(_THETA)), peaks]
    segments = []
    for _ in range(34):
        angle = int(np.argmax(supports))
        index = peaks[angle]
        if supports[angle] < 12 or len(segments) == 10:
            break
        direction = _ANGLES[:, angle]
        distance = index - radius
        nearby = points[abs(points @ direction - distance) <= 1.45]
        origin = nearby.mean(axis=0)
        eigenvalues, directions = np.linalg.eigh((nearby - origin).T @ (nearby - origin))
        normal = directions[:, 0]
        normal *= np.sign(normal @ direction)
        delta = float(normal @ origin)
        length = np.ptp(nearby @ directions[:, 1])
        if length >= 20 and not any(abs(normal @ prior[:2]) > .998 and
                                    abs(delta - np.sign(normal @ prior[:2]) * prior[2]) < 4.5
                                    for prior in segments):
            segments.append(np.r_[normal, delta])
        for adjacent in range(angle - 3, angle + 4):
            target = adjacent % len(_THETA)
            adjusted = int(round(origin @ _ANGLES[:, target])) + radius
            lower, upper = max(0, adjusted - 5), adjusted + 6
            counts[target, lower:upper] = 0
            # Clearing bins can only change a row whose first maximum was removed.
            if lower <= peaks[target] < upper:
                peaks[target] = np.argmax(counts[target])
                supports[target] = counts[target, peaks[target]]
    return tuple(np.r_[line[:2], -line[2]] for line in segments)


def _measurement(rgb):
    height, width = rgb.shape[:2]
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    table = _teal(r, g, b)
    if np.count_nonzero(table) < 220:
        return None
    closest = ndimage.distance_transform_edt(~table)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    darkest = np.minimum(np.minimum(red, green), blue)
    brightest = np.maximum(np.maximum(red, green), blue)
    white = (darkest > 105) & (brightest - darkest < 43) & (closest < max(5., .023 * height))
    yy, xx = np.nonzero(white)
    points = np.column_stack((xx, yy)).astype(float)
    if len(points) < 35:
        return None
    distance = ndimage.distance_transform_edt(~white)
    foreground = ((r > 65) & (g * 100 < 58 * r) & (b * 100 < 62 * r)) | (
        (r > 100) & (g * 100 > 65 * r) & (b * 100 < 32 * r))
    foreground = ndimage.binary_dilation(foreground, iterations=3)
    ys, xs = np.nonzero(table[::5, ::5])
    rays = np.column_stack((xs * 5, ys * 5))
    return distance, closest, points, rays, foreground, (height, width)


def _restore(parameters, focal, shape):
    height, width = shape
    return CameraFit(parameters[3:6], Rotation.from_rotvec(parameters[:3]).as_matrix(), focal, width, height)


def _project_samples(camera, model, field, foreground=None):
    projected = camera.project(model)
    u, v, z = projected.T
    visible = (z > .06) & (u > 1) & (u < camera.width - 2) & (v > 1) & (v < camera.height - 2)
    if foreground is not None:
        selected = np.flatnonzero(visible)
        visible[selected] &= ~foreground[np.rint(v[selected]).astype(int), np.rint(u[selected]).astype(int)]
    values = np.full(len(model), 10.)
    values[visible] = ndimage.map_coordinates(field, (v[visible], u[visible]), order=1, prefilter=False)
    return visible, values


def _cost(camera, measurements, lines, interior, bounds):
    distance, closest, _, pixels, foreground, _ = measurements
    visible, errors = _project_samples(camera, lines, distance, foreground)
    inside, colors = _project_samples(camera, interior, closest, foreground)
    if visible.sum() < 34 or inside.sum() < 16:
        return np.inf, np.inf, 0.
    clipped = np.sort(np.minimum(errors[visible], 9.))
    line_cost = float(np.mean(clipped[:max(1, int(.86 * len(clipped)))]))
    interior_support = float(np.mean(colors[inside] < 3.5))
    rays = np.column_stack(((pixels - camera.center) / camera.focal, np.ones(len(pixels)))) @ camera.rotation
    scale = -camera.eye[2] / np.minimum(-1e-6, rays[:, 2])
    xy = camera.eye[:2] + rays[:, :2] * scale[:, None]
    accepted = (rays[:, 2] < 0) & (np.abs(xy) <= 1.055 * np.asarray(bounds)).all(axis=1)
    exterior_support = float(np.mean(accepted))
    support = min(interior_support, exterior_support)
    cost = line_cost + 5. * (1. - interior_support) + 7. * (1. - exterior_support)
    return cost, line_cost, support


def _candidates(measurements, focal, rail, end):
    _, _, pixels, _, _, (height, width) = measurements
    center = (np.array((width, height)) - 1.) / 2.
    lines = np.asarray(_straight_segments(pixels, width, height))
    if len(lines) < 3:
        return
    intersections = np.cross(lines[:, None, :], lines[None, :, :])

    def camera_ray(pixel):
        return np.r_[((pixel[:2] - center * pixel[2]) / focal), pixel[2]]

    for i, first in enumerate(lines):
        for j, second in enumerate(lines[i + 1:], i + 1):
            if abs(first[:2] @ second[:2]) < .46:
                continue
            vanish = camera_ray(intersections[i, j])
            norm = np.linalg.norm(vanish)
            if norm < 1.e-6:
                continue
            axis = vanish / norm
            for k, cross in enumerate(lines):
                if k in (i, j) or max(abs(first[:2] @ cross[:2]), abs(second[:2] @ cross[:2])) > .90:
                    continue
                a, b = intersections[i, k], intersections[j, k]
                if min(abs(a[2]), abs(b[2])) < .05:
                    continue
                a, b = (pixel / pixel[2] for pixel in (a, b))
                if (np.linalg.norm(a[:2] - b[:2]) < .08 * width or
                        max(np.linalg.norm(a[:2] - center), np.linalg.norm(b[:2] - center)) > 1.1 * width):
                    continue
                ar, br = camera_ray(a), camera_ray(b)
                if abs(br @ axis) < .025:
                    continue
                ratio = (ar @ axis) / (br @ axis)
                private_delta = ratio * br - ar
                norm = np.linalg.norm(private_delta)
                if not .2 < ratio < 5. or norm < .02:
                    continue
                lateral = private_delta / norm
                for x0, x1 in ((-rail, rail), (rail, -rail), (-rail, 0.), (0., rail), (rail, 0.), (0., -rail)):
                    x_axis = np.sign(x1 - x0) * lateral
                    depth = abs(x1 - x0) / norm
                    for longitudinal in (axis, -axis):
                        rotation = np.column_stack((x_axis, longitudinal, np.cross(x_axis, longitudinal)))
                        for y in (-end, end):
                            eye = np.array((x0, y, .001)) - depth * (ar @ rotation)
                            if (abs(eye[0]) < 3.5 and -4.8 < eye[1] < -.37 and .045 < eye[2] < 3.1
                                    and rotation[2, 1] > .34 and abs(rotation[0, 2]) < .60):
                                yield CameraFit(eye, rotation, focal, width, height)


def register_table(rgb, bounds, focal, previous=None, *, search=True, motion=None):
    """Use a known lens and learner-side station orientation with no pose metadata."""
    measurement = _measurement(rgb)
    if measurement is None:
        return None
    shape = measurement[-1]
    rail, end, lines, interior = _court_geometry(tuple(bounds))

    def score(camera):
        return _cost(camera, measurement, lines, interior, bounds)

    def continuous(camera):
        if motion is None:
            return True
        earlier, elapsed_frames = motion
        if np.linalg.norm(camera.eye - earlier.eye) > min(.22, .032 + .042 * elapsed_frames):
            return False
        angular = np.clip((np.trace(camera.rotation @ earlier.rotation.T) - 1.) / 2., -1., 1.)
        return np.arccos(angular) <= min(.8, .15 + .15 * elapsed_frames)

    starts = []
    previous_cost = np.inf
    if (previous is not None and (previous.height, previous.width) == shape and
            abs(previous.focal - focal) < .01 and continuous(previous)):
        quality = score(previous)
        if quality[0] < .90 and quality[2] > .88:
            return previous
        previous_cost = quality[0]
        starts.append((quality, previous))
    if search and previous_cost > 3.4:
        starts.extend((score(item), item) for item in _candidates(measurement, focal, rail, end)
                      if continuous(item))
    starts = sorted((entry for entry in starts if entry[0][0] < 9.), key=lambda entry: entry[0][0])[:3]
    best = None
    for start_quality, start in starts:
        initial = np.r_[Rotation.from_matrix(start.rotation).as_rotvec(), start.eye]
        # Hold this visible subset fixed so refinement cannot reduce loss by
        # moving a marking outside the image.
        visible, distances = _project_samples(start, lines, measurement[0], measurement[4])
        chosen = lines[visible & (distances < 10.)]
        if len(chosen) < 27:
            continue

        def residual(parameters):
            _, errors = _project_samples(_restore(parameters, focal, shape), chosen, measurement[0])
            return np.r_[errors, .22 * (parameters - initial) / np.array((.1, .1, .1, .15, .18, .13))]

        fit = least_squares(residual, initial, loss='soft_l1', f_scale=.85, max_nfev=20, diff_step=1e-4)
        candidate = _restore(fit.x, focal, shape)
        quality = score(candidate)
        if quality[0] < start_quality[0]:
            selected = candidate
        else:
            selected, quality = start, start_quality
        x, y, z = selected.eye
        if (quality[0] < 2.75 and quality[1] < 1.8 and quality[2] > .76 and continuous(selected) and
                abs(x) < 3.5 and -4.8 < y < -.37 and .04 < z < 3.1 and selected.rotation[2, 1] > .34 and
                (best is None or quality[0] < best[0])):
            best = quality[0], CameraFit(selected.eye, selected.rotation, focal, selected.width,
                                        selected.height, quality[1], len(chosen))
    return None if best is None else best[1]
