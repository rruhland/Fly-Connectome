"""Learn an age-dependent extrapolation gain from actual native Pong points."""

from collections import deque
from hashlib import sha256

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import CAP, COLUMNS, HEIGHT, PLAN as WARM_PLAN, ROWS, STRIDE, WARM, WIDTH, arcade, field
from .tt24_pong_temporal_wrist import DIAGNOSTIC, advance, packet_ticks, starts


PLAN = ROOT / 'pong-temporal-plan.json'
LOCK = ROOT / 'pong-temporal-source.json.gz'
GRID = np.linspace(0., 2., 9)
NATIVE_HZ = 120
CAMERA_HZ = NATIVE_HZ // STRIDE
SHUFFLE = 123998124
SCHEDULE = dict(practice=(1410058499, 80), delayed_held=(41319819, 64), prompt_held=(509932642, 32))


def pixels(frame):
    interior = frame[:, :, 3:-3]
    counts = interior.sum(axis=(1, 2))
    measured = np.column_stack((interior.sum(axis=1) @ COLUMNS,
                                interior.sum(axis=2) @ ROWS)) / np.maximum(1, counts[:, None])
    return measured, counts > 0


class TaggedBall:
    """Emit only the mover detections corresponding to delivered source packets."""

    def __init__(self, seeds, repeats):
        self.names = seeds
        self.repeats = repeats
        size = len(seeds) * repeats
        self.indices = np.arange(size)
        self.capture = np.full(size, -1, dtype=int)
        self.seen = np.zeros(size, dtype=bool)
        self.history = deque(maxlen=5)

    def update(self, tick, frame, position, velocity):
        paddle = frame[:, :, 1]
        actor = (paddle @ ROWS) / np.maximum(1, paddle.sum(axis=1))
        self.history.appendleft(pixels(frame))
        requested = np.array([packet_ticks(tick, seed, CAMERA_HZ) for seed in self.names])
        capture = np.maximum(self.capture, np.maximum(0, tick - np.tile(requested, self.repeats)))
        age = tick - capture
        measured, visible = zip(*self.history, strict=True)
        measured = np.asarray(measured)[age, self.indices]
        visible = np.asarray(visible)[age, self.indices]
        keep = capture > self.capture
        gap = capture - self.capture
        valid = keep & visible & self.seen
        smooth = .69 ** gap[valid, None]
        velocity[valid] = smooth * velocity[valid] + ((1 - smooth) * (measured[valid] - position[valid])
                                                     * CAMERA_HZ / gap[valid, None])
        velocity[keep & ~valid] = 0.
        position[keep & visible] = measured[keep & visible]
        self.seen[keep] = visible[keep]
        self.capture = capture
        return actor, self.seen, age


@torch.no_grad()
def points(weights, gains, seeds, delay):
    """Keep current paddle optics and run the frozen native policy on delivered mover optics."""
    repeats = len(gains)
    gain = np.repeat(gains, len(seeds))
    size = len(gain)
    game = Pong(seeds * repeats, Physics(opponent_speed=.45))
    packets = TaggedBall(seeds, repeats) if delay else None
    live = np.ones(size, dtype=bool)
    position = np.zeros((size, 2))
    velocity = np.zeros_like(position)
    visible = np.zeros(size, dtype=bool)
    score, hits, elapsed, action, paddle_speed = [np.zeros(size) for _ in range(5)]
    action_tape = np.zeros((CAP // STRIDE, size), dtype='float32')
    for step in range(CAP):
        if step % STRIDE == 0:
            tick = step // STRIDE
            frame = game.render(HEIGHT, WIDTH).numpy()
            if packets is None:
                paddle, seen = field(frame, position, velocity, visible)
                predicted = position
            else:
                paddle, seen, age = packets.update(tick, frame, position, velocity)
                predicted = advance(position, velocity, age[:, None] / CAMERA_HZ, gain[:, None])
            if step == 0:
                anchor = paddle.copy()
                previous_paddle = paddle.copy()
            gap = np.clip((predicted[:, 1] - paddle) * .76 / .065, -3., 3.)
            tangent = np.clip(velocity[:, 1] * 3. / .38, -3., 3.)
            pose = np.clip((paddle - anchor) * .76 / .04, -3., 3.)
            normal = np.clip(velocity[:, 0] * 3. / 1.5, -3., 3.)
            features = np.stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)), axis=1)
            offset = .36 * np.tanh((features * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (NATIVE_HZ / STRIDE)
            request = np.clip(-11. * (anchor + offset - paddle) + 1.35 * paddle_speed, -1., 1.)
            action[seen] = request[seen]
            action_tape[tick, live] = action[live]
            previous_paddle = paddle
        result = game.step(torch.from_numpy(action.astype(np.float32)), auto_reset=False)
        scored = result.scores.numpy()
        elapsed += live
        hits += result.hits.numpy() * live
        score += scored * live
        live &= scored == 0.
        if not live.any():
            break
    stats = np.column_stack((score, hits, elapsed)).reshape(repeats, len(seeds), 3)
    digests = [sha256(np.ascontiguousarray(col, dtype='<f4').tobytes()).hexdigest() for col in action_tape.T]
    return stats, np.asarray(digests).reshape(repeats, len(seeds))


def fit(payoffs):
    x = GRID - 1.
    design = np.column_stack((np.ones(len(x)), x, x**2))
    curve = payoffs.mean(axis=1)
    coefficients = np.linalg.solve(design.T @ design + np.diag((0., .10, .10)), design.T @ curve)
    _, slope, bend = coefficients
    best = np.clip(1. - slope / (2. * bend), 0., 2.) if bend < 0 else (2. if slope > 0 else 0.)
    return dict(gain=float(best), score_curve=curve.tolist(), curve_coefficients=coefficients.tolist())


def summary(points):
    scores, hits, duration = points.T
    return dict(wins=int((scores > 0).sum()), losses=int((scores < 0).sum()), net=int(scores.sum()),
                returns=int((hits > 0).sum()), total_returns=int(hits.sum()), ticks=int(duration.sum()))


def run():
    if not _read(DIAGNOSTIC)['passed']:
        raise ValueError('registered visual-latency diagnostic failed')
    torch.set_num_threads(1)
    rosters = {label: starts(*args) for label, args in SCHEDULE.items()}
    old = _read(WARM_PLAN)
    earlier = {seed for group in old['warm_train'] for seed in group} | set(old['warm_held'])
    all_seeds = [seed for row in rosters.values() for seed in row]
    if set(all_seeds) & earlier or len(set(all_seeds)) != len(all_seeds):
        raise ValueError('the temporal course overlaps an earlier native roster')
    plan = dict(schema='tt24-pong-temporal-source-1', rosters=rosters, gains=GRID.tolist(),
                shuffle_seed=SHUFFLE, payoff='terminal native point score only', native_parent_sha256=_hash(WARM),
                diagnostic_sha256=_hash(DIAGNOSTIC), forecast='p_capture + gain * elapsed_seconds * v_capture',
                wrist_practice_for_gain=0, max_native_physics_ticks=CAP, source_camera_hz=CAMERA_HZ,
                fit='quadratic ridge on matched source means, maximize on [0,2]')
    if PLAN.exists() and plan != _read(PLAN):
        raise ValueError('registered temporal source plan changed')
    _write(PLAN, plan)
    weights = np.asarray(_read(WARM)['weights'])
    practice, practice_tape = points(weights, GRID, rosters['practice'], True)
    payoffs = practice[..., 0]
    rng = np.random.default_rng(SHUFFLE)
    order = np.column_stack([rng.permutation(len(GRID)) for _ in rosters['practice']])
    fits = dict(pong=fit(payoffs), shuffled=fit(np.take_along_axis(payoffs, order, axis=0)))
    gains = dict(pong=fits['pong']['gain'], shuffled=fits['shuffled']['gain'], cold=0., always=1.)
    print('native practice', fits, 'frozen gains', gains, flush=True)
    held, held_tape = points(weights, list(gains.values()), rosters['delayed_held'], True)
    prompt, prompt_tape = points(weights, list(gains.values()), rosters['prompt_held'], False)
    parity = arcade([weights], rosters['prompt_held'])[0]
    if not (np.all(prompt == parity) and np.all(prompt_tape == prompt_tape[0])):
        raise ValueError('prompt Pong retention or original source parity failed')
    receipt = dict(plan_sha256=_hash(PLAN), gains=gains, fits=fits, practice=practice.tolist(),
                   practice_action_sha256=practice_tape.tolist(), source_practice_points=int(practice.shape[0]*practice.shape[1]),
                   source_practice_ticks=int(practice[..., 2].sum()),
                   delayed_held=held.tolist(), delayed_action_sha256=held_tape.tolist(),
                   delayed_summary={name: summary(row) for name, row in zip(gains, held, strict=True)},
                   prompt_held=prompt.tolist(), prompt_action_sha256=prompt_tape.tolist(),
                   prompt_summary=summary(prompt[0]), exact_prompt_action_retention=True)
    _write(LOCK, receipt)
    print('native held', receipt['delayed_summary'], 'prompt original parity', receipt['prompt_summary'], flush=True)


if __name__ == '__main__':
    run()
