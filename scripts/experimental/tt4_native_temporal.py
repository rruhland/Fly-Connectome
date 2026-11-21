"""Fit a static RGB reference from the ordinary receiving-arm probe frames."""

import numpy as np

from .tt4_camera import PerspectiveTracker, _color_regions


def modal_image(images):
    """Find the most frequently observed RGB triplet at each fixed camera pixel."""
    frames = np.asarray(images, dtype=np.uint8)
    if frames.ndim != 5 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError('expected time × cameras × height × width × RGB')
    packed = frames[..., 0].astype(np.uint32) << 16
    packed |= frames[..., 1].astype(np.uint32) << 8
    packed |= frames[..., 2]
    packed.sort(axis=0)
    longest = np.zeros(packed.shape[1:], dtype=np.int32)
    run = np.zeros_like(longest)
    mode = np.zeros_like(packed[0])
    previous = mode.copy()
    for row in packed:
        run = np.where(row == previous, run + 1, 1)
        update = run > longest
        mode[update] = row[update]
        longest[update] = run[update]
        previous = row
    return np.stack((mode >> 16, (mode >> 8) & 255, mode & 255), axis=-1).astype(np.int16)


class ChangedRegions:
    """Segment recurrent changes relative to a reference learned from probe RGB."""

    def __init__(self, reference):
        self.reference = reference

    def __call__(self, frame):
        rgb = np.asarray(frame)
        if rgb.shape != self.reference.shape or rgb.dtype != np.uint8:
            raise ValueError('expected an RGB uint8 image at the calibrated resolution')
        difference = np.max(np.abs(rgb.astype(np.int16) - self.reference), axis=-1)
        changed = difference >= 9
        if not changed.any():
            return ()
        normalized = rgb.astype(np.float32) / 255
        lightness = normalized.mean(axis=-1, keepdims=True)
        chroma = (normalized - lightness) / np.maximum(3 * lightness, .18)
        appearance = np.concatenate((chroma, .17 * np.sqrt(lightness)), axis=-1)
        appearance /= np.maximum(.02, np.linalg.norm(appearance, axis=-1, keepdims=True))
        return tuple(region for region in _color_regions(difference, appearance, changed)
                     if region.area < .03 * changed.size)


def temporal_tracker(rig, reference):
    same_view = np.dot(rig[0].basis[2], rig[1].basis[2]) > .95
    return PerspectiveTracker(
        rig, coordinates='world', detector_factory=lambda camera: ChangedRegions(reference[rig.index(camera)]),
        pair_area_weight=.18 if same_view else 0., max_epipolar_error=2.6,
    )
