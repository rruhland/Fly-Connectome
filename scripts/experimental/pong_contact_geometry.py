"""Refine a shot target using recent sampled visual positions."""

from dataclasses import dataclass
import math

from pong_score_placement import ScorePlacementController


CAMERA_PIXEL = 1. / 64
VISIBLE_WALL = .5 - CAMERA_PIXEL / 2


@dataclass(frozen=True)
class ProjectedContact:
    position: float
    uncertainty: float


def project_contact(history, context, now):
    """Project the ball to the paddle's inward pixel edge from camera history."""
    if len(history) < 5 or now - history[-1][0] > .05:
        return ProjectedContact(context.intercept, CAMERA_PIXEL)
    t0, y0, x0 = history[0]
    t1, y1, x1 = history[-1]
    if t1 - t0 < .08 or abs(x1 - x0) < CAMERA_PIXEL:
        return ProjectedContact(context.intercept, CAMERA_PIXEL)
    vx = (x1 - x0) / (t1 - t0)
    # The original projection reports the time to the observed center. Move
    # that plane one image pixel into the court to target the facing edge.
    remaining = max(0., context.seconds - CAMERA_PIXEL / abs(vx))
    samples = [(t - t1, y) for t, y, _ in history]
    # A recent reflected path contains two slopes. Fit only the visible leg
    # since the last turn near the upper or lower image boundary.
    for i in range(len(samples) - 2, 0, -1):
        before, value, after = (samples[j][1] for j in (i - 1, i, i + 1))
        if abs(value) > .43 and (value - before) * (after - value) < 0.:
            if len(samples) - i < 4:
                return ProjectedContact(context.intercept, CAMERA_PIXEL)
            samples = samples[i:]
            break
    mean_t = sum(t for t, _ in samples) / len(samples)
    mean_y = sum(y for _, y in samples) / len(samples)
    time_variation = sum((t - mean_t) ** 2 for t, _ in samples)
    slope = sum((t - mean_t) * (y - mean_y) for t, y in samples) / time_variation
    if abs(slope) > .70:
        return ProjectedContact(context.intercept, CAMERA_PIXEL)
    age = now - t1
    projection = mean_y + slope * (remaining + age - mean_t)
    reflected = VISIBLE_WALL - abs((projection + VISIBLE_WALL) % (4 * VISIBLE_WALL)
                                   - 2 * VISIBLE_WALL)
    disagreement = reflected - context.intercept
    shift = max(-.04, min(.04, disagreement))
    residual = math.sqrt(sum((y - mean_y - slope * (t - mean_t)) ** 2 for t, y in samples)
                         / len(samples))
    uncertainty = min(.025, max(CAMERA_PIXEL / 2, 1.5 * residual) + .20 * abs(disagreement))
    return ProjectedContact(context.intercept + shift, uncertainty)


class ContactGeometryController(ScorePlacementController):
    """Preserve score rankings and use camera geometry to place active shots."""

    def __init__(self, timing, placement, *, forecast_center=False, guard=True):
        super().__init__(timing, placement, forecast_center=forecast_center)
        self.guard = guard

    def _offset(self, context):
        offset = super()._offset(context)
        if not offset:
            return 0.
        projection = project_contact(self.timing.inherited.history, context, self.timing.seconds)
        if self.guard and abs(offset) > .01:
            limit = max(.086, .106 - projection.uncertainty)
            offset = math.copysign(min(abs(offset), limit), offset)
        target = max(context.lower, min(context.upper, projection.position + offset))
        return target - context.intercept
