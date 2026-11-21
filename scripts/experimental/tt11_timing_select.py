"""Choose between two actuator transients from ball-free first-person stroke probes."""

from hashlib import sha256

import numpy as np

from .tt6_body_arena import TT6Wiring
from .tt6_body_control import CameraServo
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt10_series_trial import _restore_tracker


PREPARE = 7
SETTLE = 7
PROBE_COUNT = 8
STROKE_START = PREPARE + SETTLE
SHOT_FRAMES = (3, 5)
MIN_SPEED = .8  # m/s, from the optical blade's two-frame change in depth
MODES = ('inherited', 'cursor')


class StrokeProbeArena(TT8ServiceArena):
    """Park the ball outside the court while stepping the normal hall motors."""

    def _opening(self):
        self.ball[:] = (1.8, 2.15, .05)
        self.velocity[:] = 0.

    def _contacts(self, incoming):
        pass


def stroke_commands(model):
    """Request two paces of short, forward swings from varied poses."""
    jacobian = model.jacobian(np.zeros(model.action_size))
    pose_to_command = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + .015**2 * np.eye(4))
    cases = [(side, high, n) for n in SHOT_FRAMES for side in (-1., 1.) for high in (-1., 1.)]
    for side, high, n in cases:
        start = pose_to_command @ np.array([side * .075, -.13, .145 + high * .05, .09 * high])
        end = pose_to_command @ np.array([side * .055, .105, .18 + high * .05, -.11 * high])
        before, after = np.clip([start, end], -3.4, 3.4)
        preparation = np.tile(before / PREPARE, (PREPARE, 1))
        swing = np.tile(np.clip((after - before) / n, -1., 1.), (n, 1))
        commands = np.concatenate((preparation, np.zeros((SETTLE, model.action_size)), swing,
                                   np.zeros((3, model.action_size))))
        yield n, commands


def observe(arena_args, optics, model, seeds):
    """Read calibrated blade sightings and hash the stereo tape used to obtain them."""
    if len(seeds) != PROBE_COUNT or len(set(seeds)) != PROBE_COUNT:
        raise ValueError('the stroke probe requires eight distinct reset seeds')
    arena = StrokeProbeArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    rows = []
    try:
        cache_hall_stereo(arena, reflections=True)
        factory = _restore_tracker(arena.cameras, optics)
        for seed, (duration, commands) in zip(seeds, stroke_commands(model), strict=True):
            tracker = factory()
            history = collect_probe(wiring, seed, commands)
            if len(history) != len(commands) + 1:
                raise RuntimeError('dry stroke ended before its commanded swing completed')
            digest = sha256()
            poses = []
            for _, frames in history:
                digest.update(frames.tobytes())
                tracker.update(frames)
                actor = tracker.actor
                poses.append(None if actor is None or not actor.observed else
                             [*actor.position, tracker.actor_pitch])
            rows.append(dict(seed=seed, stroke_frames=duration, commands=commands.tolist(),
                             blade=poses, rgb_sha256=digest.hexdigest()))
    finally:
        arena.close()
    return rows


def _predictions(model, commands):
    servo = CameraServo(model)
    frames = [model.home]
    for action in commands:
        servo.advance(np.asarray(action))
        frames.append(model.pose(servo.effective))
    return np.asarray(frames)


def select(models, rows, control_dt):
    """Score two-frame displacement near the drive, with matched visibility for both models."""
    costs = {name: [] for name in MODES}
    stroke_scores = []
    speeds = []
    n_pairs = 0
    for row in rows:
        poses = row['blade']
        pairs = [(t - 2, t) for t in range(STROKE_START + 1, STROKE_START + row['stroke_frames'] + 3)
                 if poses[t - 2] is not None and poses[t] is not None]
        scores = {}
        speed = None
        if pairs:
            a, b = np.asarray(pairs).T
            observed = np.array(poses, dtype=object)
            p0 = np.asarray([observed[t] for t in a], dtype=float)
            p1 = np.asarray([observed[t] for t in b], dtype=float)
            actual = p1 - p0
            speed = float(np.max(actual[:, 1]) / (2. * control_dt))
            speeds.append(speed)
            n_pairs += len(pairs)
            for name in MODES:
                estimate = _predictions(models[name], row['commands'])
                residual = estimate[b] - estimate[a] - actual
                # Forward error sets the blade's launch velocity. Pitch is in
                # radians and gets a lever arm comparable to the blade radius.
                residual[:, 3] = np.nan_to_num(residual[:, 3])
                weighted = np.clip(residual * [1., 1.5, 1., .08], -.075, .075)
                loss = float(np.mean(np.sum(weighted**2, axis=1)))
                costs[name].append(loss)
                scores[name] = dict(error_m=float(np.sqrt(loss)),
                                    forward_error_mps=float(np.mean(np.abs(residual[:, 1])) / (2.*control_dt)))
        stroke_scores.append(dict(seed=row['seed'], pairs=len(pairs), peak_forward_mps=speed, errors=scores))
    aggregate = {name: float(np.sqrt(np.mean(values))) if values else None for name, values in costs.items()}
    supported = len(speeds) >= 4 and n_pairs >= 16 and np.median(speeds) >= MIN_SPEED
    selected = min(MODES, key=aggregate.__getitem__) if supported else 'inherited'
    return dict(selected=selected, supported=bool(supported), error_m=aggregate, scores=stroke_scores,
                visible_pairs=n_pairs, median_peak_forward_mps=float(np.median(speeds)) if speeds else None)
