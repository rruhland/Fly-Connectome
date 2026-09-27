"""Local sensory credibility and connected visual evidence."""
import torch
import torch.nn.functional as F

class LocalObservationModel:
    """Weight incoming signed events by locally learned sensory agreement."""

    def __init__(self, *, height=32, width=64):
        self.height = height
        self.width = width
        self.weights = torch.zeros((2, 8, 5, 5))
        self.bias = torch.zeros((2, 1, 1))
        self.reset_state()

    def reset_state(self):
        self.fast = torch.zeros((2, self.height, self.width))
        self.slow = torch.zeros_like(self.fast)
        self.contrast = torch.zeros((self.height, self.width))

    def _features(self, raw):
        appearance = torch.stack((self.contrast.clamp(min=0.),
                                  (-self.contrast).clamp(min=0.)))
        return torch.cat((raw, self.fast, self.slow, appearance))

    @torch.no_grad()
    def step(self, raw, *, absolute=None):
        features = self._features(raw)
        credibility = torch.sigmoid(F.conv2d(
            features[None], self.weights, padding=2)[0]+self.bias)
        filtered = raw*credibility
        self.fast.mul_(.65).add_(filtered).clamp_(0., 1.)
        self.slow.mul_(.95).add_(filtered).clamp_(0., 1.)
        self.contrast.add_(filtered[1]-filtered[0]).clamp_(-1., 1.)
        if absolute is not None:
            self.contrast.lerp_(absolute, .35)
        return filtered, features

    @torch.no_grad()
    def credit(self, features, raw, observed_change):
        credibility = torch.sigmoid(F.conv2d(
            features[None], self.weights, padding=2)[0]+self.bias)
        error = (observed_change-credibility)*raw
        patches = F.unfold(features[None], kernel_size=5,
                           padding=2)[0]
        count = raw.sum((1, 2)).clamp(min=1.)
        update = error.reshape(2, -1) @ patches.T
        self.weights.add_(.1*(update/count[:, None]).reshape_as(
            self.weights)).clamp_(-2., 2.)
        self.bias.add_(.1*error.sum((1, 2))[:, None, None]
                       / count[:, None, None]).clamp_(-2., 2.)


def local_motion_map(trace, current):
    """Signed local x/y evidence at each current-event pixel."""
    x = torch.zeros_like(current[0])
    y = torch.zeros_like(current[0])
    for distance in (1, 2):
        x[:, distance:] += (trace[:, :, :-distance]
                            * current[:, :, distance:]).sum(0)
        x[:, :-distance] -= (trace[:, :, distance:]
                             * current[:, :, :-distance]).sum(0)
        y[distance:, :] += (trace[:, :-distance, :]
                            * current[:, distance:, :]).sum(0)
        y[:-distance, :] -= (trace[:, distance:, :]
                             * current[:, :-distance, :]).sum(0)
    return torch.stack((x, y))


def surface_components(mask, contrast, changed, diagonal_links=None):
    """Connected sensory proposals; identity is assigned by the recurrent files."""
    remaining = set(map(tuple, mask.nonzero(as_tuple=False).tolist()))
    groups = []
    while remaining:
        stack = [remaining.pop()]
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            if diagonal_links:
                sign = 1. if contrast[y, x] >= 0 else -1.
                for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ny, nx = y+dy, x+dx
                    if (diagonal_links.get((sign, dy*dx), False) and
                            0 <= ny < mask.shape[0] and
                            0 <= nx < mask.shape[1] and
                            sign*contrast[ny, nx] > 0):
                        neighbors.append((ny, nx))
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        cy = sum(y for y, _ in pixels)/len(pixels)
        cx = sum(x for _, x in pixels)/len(pixels)
        feature = torch.zeros((7, 7))
        for y, x in pixels:
            fy, fx = y-round(cy)+3, x-round(cx)+3
            if 0 <= fy < 7 and 0 <= fx < 7:
                feature[fy, fx] = 1.
        groups.append(dict(center=(cy, cx), pixels=pixels,
                           sign=1. if sum(float(contrast[y, x])
                           for y, x in pixels) >= 0 else -1.,
                           feature=feature,
                           changed=any(bool(changed[y, x])
                                       for y, x in pixels)))
    return groups
