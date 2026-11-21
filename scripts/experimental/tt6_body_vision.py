"""Find the rubber face amid the moving supports of unfamiliar arms."""

import numpy as np

from .tt5_vision import HallTracker, _Stereo, _response_apertures, modal_image


def _rubber_sample(image, difference, aperture):
    crop = image[aperture]
    r, g, b = np.moveaxis(crop, -1, 0)
    # Table tennis rubber is normally red or black. Use broad color ranges to
    # remove warm wood and cool structural hardware before learning its RGB.
    red = (r > 1.65 * g) & (r > 1.42 * b) & (r > 68)
    black = (r < 67) & (r >= g + 2) & (r > b)
    pixels = crop[(red | black) & (difference[aperture] >= 14)]
    if len(pixels) < 7:
        return None
    ids = (pixels.astype(np.int32) // 8) @ np.array((1024, 32, 1))
    return np.median(pixels[ids == np.bincount(ids).argmax()], axis=0)


class BodyTracker(HallTracker):
    """Identify the controlled face by appearance within a probe-responsive region."""

    def _actor_observation(self, rgb, differences):
        if self._palette is None:
            palettes = tuple(_rubber_sample(image, difference, aperture)
                             for image, difference, aperture in zip(rgb, differences, self.apertures, strict=True))
            if any(palette is None for palette in palettes):
                return
            if np.linalg.norm(palettes[0] - palettes[1]) > 65:
                return
            self._palette = palettes
        super()._actor_observation(rgb, differences)


def body_tracker_factory(rig, histories):
    images = np.asarray([frames for history in histories for _, frames in history[::2]])
    reference = modal_image(images)
    stereo = _Stereo(rig)
    apertures = _response_apertures(histories, images)

    def factory():
        return BodyTracker(stereo, reference, apertures)

    return factory
