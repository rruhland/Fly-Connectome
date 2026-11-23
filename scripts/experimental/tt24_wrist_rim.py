"""Estimate the orientation of an existing circular tool from its visible wrist-camera rim."""
import numpy as np
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_public_body import FOCAL


def directions(count):
    angle = np.arange(count) * (2 * np.pi / count)
    return np.column_stack((np.cos(angle), np.sin(angle)))


RIM = directions(256)
SUPPORT = directions(64)


def visible_support(image):
    red, green, blue = image.astype(float).transpose(2, 0, 1)
    components, count = label((red > 45) & (green < .35 * red) & (blue < .4 * red))
    if not count:
        raise ValueError('no visible tool surface')
    counts = np.bincount(components.ravel())
    counts[0] = 0
    y, x = np.nonzero(components == counts.argmax())
    h, w = image.shape[:2]
    points = np.column_stack((x - (w - 1) / 2, (h - 1) / 2 - y))
    projection = points @ SUPPORT.T
    winners = points[projection.argmax(axis=0)]
    interior = (abs(winners[:, 0]) < w / 2 - 2) & (abs(winners[:, 1]) < h / 2 - 2)
    return projection.max(axis=0) + .35, interior


def project(center, normal, radius, thickness):
    axis = np.cross(normal, [0, 0, 1])
    axis /= np.linalg.norm(axis)
    disk = center + radius * RIM @ np.stack([axis, np.cross(normal, axis)])
    points = np.concatenate([disk + thickness * normal, disk - thickness * normal])
    pixels = FOCAL * points[:, :2] / -points[:, 2, None]
    return (pixels @ SUPPORT.T).max(axis=0)


def collect(body, radius, thickness):
    arena = BounceArena(arm=body, view='wrist', width=192, height=144)
    policy = CarriedWristPolicy()
    images, diagnostics = [], []
    try:
        for seed in range(766301, 766307):
            obs = arena.reset(seed=seed, split='held')
            policy.begin_release(obs)
            decoder = policy.sensor.decoder
            images.append(visible_support(obs['rgb']))
            # These measurements are held aside until both optical fits have finished.
            camera = arena._data.camera('tt19_wrist')
            world_normal = arena._data.geom_xmat[arena._blade].reshape(3, 3)[:, 2]
            diagnostics.append((camera.xmat.reshape(3, 3).T @ world_normal).copy())
    finally:
        arena.close()
    center = decoder.wrist_attitude.T @ (decoder.offset[3:] - decoder.translation)
    original = decoder.wrist_attitude.T @ decoder.normal
    normals = original * np.sign(original[1])
    axis = np.cross(normals, [0, 0, 1])
    axis /= np.linalg.norm(axis)
    tangent = np.stack([axis, np.cross(normals, axis)])
    def face(angles):
        return Rotation.from_rotvec(angles @ tangent).apply(normals)
    def fit(rows):
        target = np.median([s for s, _ in rows], axis=0)
        valid = np.all([v for _, v in rows], axis=0)
        if valid.sum() < 15:
            raise ValueError('too little rim is visible')
        def residual(x):
            predicted = project(center, face(x), radius, thickness)
            return np.r_[predicted[valid] - target[valid], x / .15]
        best = least_squares(residual, np.zeros(2), bounds=(-.28, .28), loss='soft_l1', f_scale=.7)
        def rms(x):
            return float(np.sqrt(np.mean(residual(x)[:-2] ** 2)))
        return face(best.x), dict(rim_directions=int(valid.sum()), before_px=rms(np.zeros(2)),
                                  after_px=rms(best.x), fitted_local_angles_rad=best.x.tolist())
    fitted, receipt = fit(images[:3])
    check, held = fit(images[3:])
    def angle(n, m):
        return float(np.degrees(np.arccos(np.clip(abs(n @ m), 0, 1))))
    truth = diagnostics[0]
    # Diagnostic checks never choose the fit or its sign.
    return dict(fit=receipt, repeat=held, repeat_degrees=angle(fitted, check),
                earlier_error_degrees=angle(normals, truth), fitted_error_degrees=angle(fitted, truth),
                measured_in_camera=fitted.tolist(), original_camera_normal=normals.tolist(),
                fitted_link_normal=(decoder.wrist_attitude @ (fitted * np.sign(original[1]))).tolist())


def usable(record):
    optical = record['fit']
    return (optical['rim_directions'] == len(SUPPORT) and record['repeat_degrees'] < 1
            and np.max(np.abs(optical['fitted_local_angles_rad'])) < .27
            and optical['after_px'] < optical['before_px'])


def main():
    familiar = _read(ROOT / 'public-body-fused.json.gz')['body']['shared_projection']
    result = {body: collect(body, familiar['radius_m'], familiar['half_thickness_m'])
              for body in ('stock', 'alternate')}
    result['source_radius_and_half_thickness_m'] = [familiar['radius_m'], familiar['half_thickness_m']]
    _write(ROOT / 'wrist-rim.json.gz', result)
    for body in ('stock', 'alternate'):
        print(body, result[body], flush=True)


if __name__ == '__main__':
    main()
