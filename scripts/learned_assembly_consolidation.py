"""Unsupervised local motion expectation and competing path consolidation."""

import math

import torch


class LocalDisplacementExpectation:
    """Hebbian feature-to-local-displacement association, without labels."""

    def __init__(self, *, features, radius=3):
        self.radius = radius
        self.sum = torch.zeros((features, 2))
        self.mass = torch.zeros(features)
        self.vector = None

    @torch.no_grad()
    def observe(self, previous, current):
        for sy, sx, source in previous:
            polarity = int(source[:2].argmax())
            for ty, tx, target in current:
                dy, dx = ty-sy, tx-sx
                if (max(abs(dy), abs(dx)) <= self.radius and
                        int(target[:2].argmax()) == polarity):
                    self.sum += source[:, None]*torch.tensor((dy, dx))
                    self.mass += source

    @torch.no_grad()
    def finalize(self):
        self.vector = self.sum/self.mass[:, None].clamp(min=1e-8)

    def score(self, previous, current):
        sy, sx, source = previous
        ty, tx, _ = current
        dy, dx = ty-sy, tx-sx
        displacement = math.hypot(dy, dx)
        if not displacement or displacement > self.radius:
            return 0.
        expected = source @ self.vector
        strength = float(expected.norm())
        if not strength:
            return 0.
        alignment = float(expected @ expected.new_tensor((dy, dx))) / (
            strength*displacement)
        return alignment*strength/(strength+.5)


def path_direction(history):
    first, last = history[0], history[-1]
    dy, dx = last[1]-first[1], last[2]-first[2]
    length = math.hypot(dy, dx)
    return (dy/length, dx/length) if length else (0., 0.)


def co_moving(first, second):
    ay, ax = path_direction(first['history'])
    by, bx = path_direction(second['history'])
    if ay*by+ax*bx < .7:
        return False
    positions = {frame: (y, x) for frame, y, x in first['history']}
    common = [(positions[frame], (y, x)) for frame, y, x in
              second['history'] if frame in positions]
    if len(common) < 2:
        return False
    near = sum(max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 4
               for a, b in common)
    return near*2 >= len(common)


def consolidate_paths(paths, *, max_tracks=4):
    ranked = sorted((row for row in paths if len(row['history']) >= 3),
                    key=lambda row: (len(row['history']), row['score']),
                    reverse=True)
    chosen = []
    for row in ranked:
        if any(co_moving(row, previous) for previous in chosen):
            continue
        chosen.append(row)
        if len(chosen) == max_tracks:
            break
    return chosen
