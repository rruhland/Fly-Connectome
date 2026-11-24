"""Register a moved external camera from the white lines of a regulation table."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt18_view import CameraFit
from .tt4_arena import TABLE_HEIGHT


_STREAMS = ('left', 'right')
_HELD_SEED = 29087213
_PLAY_IDS = (4527, 8207)
_FRAME_SIZE = (384, 288)
_OUTPUT = Path('artifacts/tt29/court-registration.json.gz')
_CHECKPOINT = Path('artifacts/tt29/camera-action-memory.json.gz')
_LINE_HEIGHT = TABLE_HEIGHT + .001
_COURT_LINES = np.array([
    [(-.752, -1.36, _LINE_HEIGHT), (-.752, 1.36, _LINE_HEIGHT)],
    [(.752, -1.36, _LINE_HEIGHT), (.752, 1.36, _LINE_HEIGHT)],
    [(-.75, -1.359, _LINE_HEIGHT), (.75, -1.359, _LINE_HEIGHT)],
    [(-.75, 1.359, _LINE_HEIGHT), (.75, 1.359, _LINE_HEIGHT)],
])


@dataclass(frozen=True)
class Registration:
    camera: CameraFit
    supports: tuple[int, ...]
    line_error_px: float


def _project(points, camera):
    relative = (points - camera.eye) @ camera.rotation.T
    center = (np.array((camera.width, camera.height)) - 1.) / 2.
    return relative[..., :2] * (camera.focal / relative[..., 2:]) + center


def _pixels(rgb):
    red, green, blue = np.moveaxis(rgb.astype(np.float32), -1, 0)
    cloth = ((green > 39) & (red < green * .50) &
             (blue > green) & (blue < green * 1.195))
    pale = (red > 68) & (red > blue * .70) & (red < blue * 1.13) & (green > blue * .78)
    y, x = np.nonzero(pale & maximum_filter(cloth, size=5))
    return np.column_stack((x, y)), cloth


def _sample_mask(mask, pixels):
    width, height = mask.shape[::-1]
    rounded = np.rint(pixels).astype(int)
    x, y = np.clip(rounded, 0, (width - 1, height - 1)).T
    return mask[y, x]


def _line(start, end, table, pixels, cloth, scale, *, front=False):
    center = (start + end) / 2
    vector = end - start
    length = np.linalg.norm(vector)
    along = vector / length
    inward = np.array((-along[1], along[0]))
    if inward @ (table - center) < 0:
        inward = -inward
    offset = pixels - center
    width = 45. * scale
    chosen = (np.abs(offset @ inward) < width) & (np.abs(offset @ along) < length / 2 + width)
    pixels, offset = pixels[chosen], offset[chosen]
    inside = _sample_mask(cloth, pixels + inward * (3. * scale))
    outside = _sample_mask(cloth, pixels - inward * (3. * scale))
    # The near tabletop apron can share its playing surface's color.
    offset = offset[inside & (~outside | front)]
    if len(offset) < 10:
        raise ValueError('Too few boundary pixels near a table line')
    best = None
    for angle in np.deg2rad(np.linspace(-12., 12., 31)):
        normal = np.cos(angle) * inward + np.sin(angle) * along
        signed = offset @ normal
        counts, bins = np.histogram(signed, bins=np.arange(-width - 3, width + 4, max(.65, scale)))
        density = gaussian_filter1d(counts.astype(float), .8)
        index = int(np.argmax(density))
        peak = (bins[index] + bins[index + 1]) / 2
        support = abs(signed - peak) < 1.3 * scale
        measure = density[index] * (1. - .13 * abs(peak) / width)
        if best is None or measure > best[0]:
            best = measure, support
    selected = offset[best[1]]
    if len(selected) < 8 or np.ptp(selected @ along) < .30 * length:
        raise ValueError('A fitted table line has insufficient visible length')
    loc = selected.mean(axis=0)
    _, _, vectors = np.linalg.svd(selected - loc, full_matrices=False)
    normal = vectors[-1]
    return normal, float(normal @ (center + loc)), len(selected)


def register(rgb, nominal):
    """Recover the metric pose from one RGB frame, fixed intrinsics and a nominal survey."""
    if rgb.shape != (nominal.height, nominal.width, 3):
        raise ValueError('Camera dimensions differ from its nominal survey')
    pixels, cloth = _pixels(rgb)
    scale = nominal.height / 288.
    expected = _project(_COURT_LINES, nominal)
    middle = _project(np.array((0., 0., TABLE_HEIGHT)), nominal)
    observed = [_line(a, b, middle, pixels, cloth, scale, front=index == 2)
                for index, (a, b) in enumerate(expected)]
    axes = np.array([item[0] for item in observed])
    offsets = np.array([item[1] for item in observed])
    spans = np.linspace(.08, .92, 8)[None, :, None]
    landmarks = _COURT_LINES[:, :1] + spans * (_COURT_LINES[:, 1:] - _COURT_LINES[:, :1])

    def camera(delta):
        rotation = Rotation.from_rotvec(delta[3:]).as_matrix() @ nominal.rotation
        return CameraFit(nominal.eye + delta[:3], rotation, nominal.focal, nominal.width, nominal.height)

    def error(delta):
        projection = _project(landmarks, camera(delta))
        return (np.einsum('fpi,fi->fp', projection, axes) - offsets[:, None]).ravel()

    limit = np.array((.4, .4, .3, .25, .25, .25))
    fit = least_squares(error, np.zeros(6), bounds=(-limit, limit), max_nfev=50, loss='soft_l1')
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    if rms > 1.2 * scale or np.any(np.abs(fit.x) > .98 * limit):
        raise ValueError('The visible lines do not support a stable pose estimate')
    return Registration(camera(fit.x), tuple(item[2] for item in observed), rms)


def _perturbations():
    rng = np.random.default_rng(_HELD_SEED)
    moves = rng.uniform(-1., 1., (12, 2, 3)) * [.11, .13, .075]
    turns = rng.uniform(-1., 1., (12, 2, 3)) * [4.6, 4.6, 3.8]
    return tuple(zip(moves, turns, strict=True))


def _accuracy(camera, actual):
    rotation = Rotation.from_matrix(camera.rotation @ actual.rotation.T)
    grid = np.array([(x, y, z) for x in (-.4, 0., .4) for y in (-1.30, 0., 1.30)
                     for z in (.82, 1.03, 1.32)])
    errors = _project(grid, camera) - _project(grid, actual)
    return dict(position_mm=1000 * float(np.linalg.norm(camera.eye - actual.eye)),
                angle_deg=float(np.rad2deg(rotation.magnitude())),
                volume_reprojection_px=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))))


def run(feeds, checkpoint, output):
    """Evaluate unseen camera perturbations, followed by paired native live points."""
    from .tt4_transfer_trial import _hash, _read, _write
    from .tt19_view_trial import CAMERAS
    from .tt27_world_camera import CameraMounts
    from .tt29_material import MaterialEvidence, evaluate, restitution_grid
    from .tt29_partner import FOREHAND_LANES, PartnerArena
    from .tt29_rally_physics import RallyArena, measured_launches
    from .tt29_vision import FlightPractice, Rig

    result = dict(seed=_HELD_SEED, dimensions=_FRAME_SIZE, fits=[], plays=[])
    settings = dict(strike_ready='lateral', width=_FRAME_SIZE[0], height=_FRAME_SIZE[1],
                    fast_search=True, clearance_guard=True, max_steps=280)
    arena = PartnerArena(**settings, targets=FOREHAND_LANES, guard_table=True)
    reference = Rig.from_arena(arena)
    mounts = CameraMounts(arena)
    profiles = _perturbations()

    def move(index):
        offsets, angles = profiles[index]
        mounts.move(angles)
        for eye, offset in zip(_STREAMS, offsets, strict=True):
            arena.model.camera(CAMERAS[eye][0]).pos[:] = reference.fixed[eye].eye + offset

    def capture(index):
        move(index)
        arena.streams = _STREAMS
        return arena.render_cameras()

    try:
        cases = {}
        for index, (offsets, angles) in enumerate(profiles):
            arena.reset(seed=42, render=False)
            frames = capture(index)
            actual = Rig.from_arena(arena)
            recovered = reference.fixed.copy()
            row = dict(profile=index, translation_m=offsets.tolist(), euler_deg=angles.tolist(), cameras={})
            for eye in _STREAMS:
                prior = reference.fixed[eye]
                ceiling = actual.fixed[eye]
                scores = dict(stale=_accuracy(prior, ceiling), known_geometry=_accuracy(ceiling, ceiling))
                try:
                    fit = register(frames[eye], prior)
                    recovered[eye] = fit.camera
                    scores.update(recovered=_accuracy(fit.camera, ceiling), supports=fit.supports,
                                  line_error_px=fit.line_error_px)
                except ValueError as failure:
                    scores['failure'] = str(failure)
                row['cameras'][eye] = scores
            result['fits'].append(row)
            cases[index] = (reference.fixed.copy(), recovered, actual.fixed.copy())
        output.parent.mkdir(parents=True, exist_ok=True)
        _write(output, result)
        if feeds is None:
            return

        source = _read(checkpoint)
        prior_shots = [row['shot']['id'] for row in source['rows']]
        if set(prior_shots) & set(_PLAY_IDS):
            raise ValueError('Live opening appears in the frozen stroke checkpoint')
        launches = measured_launches(feeds)
        cap = 4
        result.update(feed_sha256=_hash(feeds), checkpoint_sha256=_hash(checkpoint), legal_return_cap=cap,
                      checkpoint_shot_ids=prior_shots, played_shot_ids=_PLAY_IDS)
        twin = RallyArena(**settings)
        try:
            bounce_reference = restitution_grid(twin)
            for profile, shot_id in enumerate(_PLAY_IDS):
                move(profile)
                for name, fixed in zip(('stale', 'recovered', 'known_geometry'), cases[profile], strict=True):
                    rig = Rig(fixed, reference.wrist)
                    observer = MaterialEvidence(bounce_reference)
                    played = evaluate(arena, twin, rig, observer, FlightPractice(), source, launches[shot_id],
                                      adapt=False, cap=cap)
                    metrics = played['score']
                    print(shot_id, name, metrics['returns'], metrics['learner'], flush=True)
                    result['plays'].append(dict(profile=profile, calibration=name, **played))
                    _write(output, result)
        finally:
            twin.close()
    finally:
        arena.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feeds', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=_OUTPUT)
    args = parser.parse_args()
    run(args.feeds, args.checkpoint, args.output)
