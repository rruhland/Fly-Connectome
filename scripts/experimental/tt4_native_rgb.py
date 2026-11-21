"""Extract anonymous salient regions from native shaded camera images."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .tt4_camera import ImageDetection, PerspectiveTracker


class NativeRegions:
    """Connect locally similar pixels using chromaticity and relative lightness.

    The scene supplies one RGB array per update. Color coordinates are continuous
    and frame-local. Motion is a ranking cue from recent RGB frames so stationary
    objects can remain visible when the action-to-motion model probes the scene.
    """

    def __init__(self, camera):
        self.shape = camera.height, camera.width
        pixels = np.arange(camera.height * camera.width, dtype=np.int32).reshape(self.shape)
        self.edge_from = np.concatenate((pixels[:, :-1].ravel(), pixels[:-1, :].ravel()))
        self.edge_to = np.concatenate((pixels[:, 1:].ravel(), pixels[1:, :].ravel()))
        self.x = (pixels % camera.width).ravel()
        self.y = (pixels // camera.width).ravel()
        self.previous = None
        self.background = None

    def __call__(self, frame):
        rgb = np.asarray(frame)
        if rgb.shape != (*self.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('expected an RGB uint8 image at the calibrated resolution')
        rgb = rgb.reshape(-1, 3).astype(np.float32) / 255
        total = rgb.sum(axis=1)
        chroma = (rgb - total[:, None] / 3) / np.maximum(total[:, None], .18)
        lightness = np.sqrt(total / 3)
        features = np.column_stack((chroma, lightness * .30))
        delta = features[self.edge_from] - features[self.edge_to]
        contrast = np.sqrt(np.einsum('ij,ij->i', delta, delta))
        edges = contrast <= .042
        graph = coo_matrix((np.ones(edges.sum(), dtype=np.int8),
                            (self.edge_from[edges], self.edge_to[edges])),
                           shape=(rgb.shape[0], rgb.shape[0])).tocsr()
        count, labels = connected_components(graph, directed=False, return_labels=True)
        area = np.bincount(labels, minlength=count)

        # Prefer an enclosing contour with a clear contrast against its surroundings.
        boundary_from, boundary_to = self.edge_from[~edges], self.edge_to[~edges]
        perimeter = np.bincount(labels[boundary_from], minlength=count)
        perimeter += np.bincount(labels[boundary_to], minlength=count)
        boundary_contrast = np.bincount(labels[boundary_from], weights=contrast[~edges], minlength=count)
        boundary_contrast += np.bincount(labels[boundary_to], weights=contrast[~edges], minlength=count)
        local_contrast = boundary_contrast / np.maximum(perimeter, 1)
        if self.previous is None:
            self.background = rgb.copy()
            motion = np.zeros(rgb.shape[0], dtype=np.float32)
        else:
            recent = np.abs(rgb - self.previous).max(axis=1)
            persistent = np.abs(rgb - self.background).max(axis=1)
            motion = np.maximum(recent, persistent * .6)
            self.background += .025 * (rgb - self.background)
        self.previous = rgb

        def by_region(values):
            return np.bincount(labels, weights=values, minlength=count) / area

        # One high-contrast pixel can be a small distant object or a glint.
        # Let it through with temporal evidence and keep a broad area range otherwise.
        movement = by_region(motion)
        compactness = np.minimum(1, 4 * np.sqrt(area) / np.maximum(perimeter, 1))
        height, width = self.shape
        x, y = by_region(self.x), by_region(self.y)
        valid = ((area <= .18 * rgb.shape[0]) & ((area >= 2) | (movement > .12)) &
                 (x > .5) & (x < width - 1.5) & (y > .5) & (y < height - 1.5))
        score = compactness * local_contrast + .45 * movement
        score *= np.minimum(1., np.sqrt(area / 3))
        score *= 1. + .10 * np.log1p(area)
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(-score[indices], kind='stable')[:16]]
        appearance = 2.4 * np.column_stack(tuple(by_region(channel) for channel in features.T))
        return tuple(ImageDetection((float(x[i]), float(y[i])), tuple(appearance[i]), float(area[i]))
                     for i in indices)


def native_tracker(rig, *, coordinates='world'):
    same_view = np.dot(rig[0].basis[2], rig[1].basis[2]) > .95
    return PerspectiveTracker(rig, coordinates=coordinates, detector_factory=NativeRegions,
                              pair_area_weight=.18 if same_view else 0., max_epipolar_error=2.6)
