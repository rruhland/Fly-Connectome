"""Test visual-motion compensation in a paddle-mounted native Pong camera."""

import argparse
from hashlib import sha256

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import HEIGHT, STRIDE, WARM, WIDTH, arcade, field
from .tt24_pong_workspace import paired, summary


PLAN = ROOT / 'pong-egomotion-source-plan.json'
PILOT = ROOT / 'pong-egomotion-source-pilot.json.gz'
SEED = 1962348329
CAP = 12 * 120
GAINS = np.asarray((-.5, 0., .5, 1., 1.5))
PAD = HEIGHT // 2
CAMERA_HZ = 120 / STRIDE


def register():
    rng = np.random.default_rng(SEED)
    rotor = rng.choice(2**31 - 2, 96, replace=False) + 1
    plan = dict(schema='tt24-moving-native-camera-1', moving=rotor[:64].tolist(), static=rotor[64:].tolist(),
                gains=GAINS.tolist(), camera='one public own paddle encoder, pixel-quantized full-FOV image translation',
                base_actor=_hash(WARM), arena='native official starts and motion, 12 s terminal score, timeout=0',
                velocity='raw mover displacement plus gain times calibrated ego displacement; fully restored positions for all',
                pilot_gate='nonunity improves on exact full correction by >=4 combined terminal points and improves at least 3 starts',
                analytic_heuristic='unity is exact full geometric correction for the declared translation')
    if PLAN.exists() and _read(PLAN) != plan:
        raise ValueError('the camera pilot differs from its registered plan')
    _write(PLAN, plan)
    return plan


def camera(outside, movement):
    """Translate actual native RGB with a generous field of view rather than wrap or crop."""
    offset = np.rint(movement * HEIGHT).astype(int)
    row = PAD+np.arange(HEIGHT)[None,:]-offset[:,None]
    channel = np.arange(len(offset))[:, None]
    view = np.zeros((len(offset), HEIGHT + 2 * PAD, WIDTH), dtype=outside.dtype)
    view[channel, row, :] = outside
    registered = view[channel, row, :]
    return view, registered, offset / HEIGHT


def mover_rows(frame):
    """Apply the same known interior mover channel to a padded native raster."""
    margin = (frame.shape[1] - HEIGHT) // 2
    axis = (np.arange(frame.shape[1]) + .5 - margin) / HEIGHT
    per_row = frame[:, :, 3:-3].sum(axis=2)
    count = per_row.sum(axis=1)
    return (per_row @ axis) / np.maximum(1, count), count > 0


@torch.no_grad()
def points(weights, roster, gains, *, move):
    size = len(roster) * len(gains)
    gain = np.repeat(gains, len(roster))
    game = Pong(roster * len(gains), Physics(opponent_speed=.45))
    live = np.ones(size, dtype=bool)
    old, velocity = [np.zeros((size, 2)) for _ in range(2)]
    visible = np.zeros(size, dtype=bool)
    score, hits, elapsed, action, paddle_speed = [np.zeros(size) for _ in range(5)]
    ego_speed = np.zeros(size)
    previous_camera = np.zeros(size)
    tapes = np.zeros((CAP // STRIDE, size), dtype='<f4')
    signals = []
    past_raw, past_outside = np.zeros(size), np.zeros(size)
    maximum_registration_error = 0.
    for step in range(CAP):
        if step % STRIDE == 0:
            clock = step // STRIDE
            outside = game.render(HEIGHT, WIDTH).numpy()
            # Treat only own paddle coordinate as an encoder packet; images carry all mover information.
            own_movement = game.body.position.numpy() - .5 if move else np.zeros(size)
            raw, restored, optical_camera = camera(outside, own_movement)
            previous_visibility = visible.copy()
            paddle, seen = field(restored, old, velocity, visible)
            continuing = seen & previous_visibility
            camera_delta = optical_camera-previous_camera
            ego_speed[continuing] = .69 * ego_speed[continuing] + .31 * camera_delta[continuing] * CAMERA_HZ
            ego_speed[~continuing] = 0.
            previous_camera = optical_camera
            raw_y, raw_seen = mover_rows(raw)
            expected_y, expected_seen = mover_rows(outside)
            if not np.array_equal(seen, raw_seen) or not np.array_equal(seen, expected_seen):
                raise ValueError('camera translation clipped a mover')
            maximum_registration_error = max(maximum_registration_error,
                                             np.max(np.abs(raw_y[seen] + optical_camera[seen] - expected_y[seen]), initial=0.))
            keep = continuing & live
            true_delta = expected_y - past_outside
            raw_delta = raw_y - past_raw
            signals.append(np.c_[camera_delta[keep], (true_delta-raw_delta)[keep]])
            past_raw, past_outside = raw_y, expected_y
            if step == 0:
                anchor, previous_paddle = paddle.copy(), paddle.copy()
            gap = np.clip((old[:, 1] - paddle) * .76 / .065, -3., 3.)
            relative = velocity[:, 1] + (gain-1.) * ego_speed
            tangent = np.clip(relative * 3. / .38, -3., 3.)
            pose = np.clip((paddle - anchor) * .76 / .04, -3., 3.)
            normal = np.clip(velocity[:, 0] * 3. / 1.5, -3., 3.)
            features = np.stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)), axis=1)
            target = .36 * np.tanh((features * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (120 / STRIDE)
            request = np.clip(-11. * (anchor + target - paddle) + 1.35 * paddle_speed, -1., 1.)
            action[seen] = request[seen]
            tapes[clock, live] = action[live]
            previous_paddle = paddle
        result = game.step(torch.from_numpy(action.astype(np.float32)), auto_reset=False)
        scored = result.scores.numpy()
        hits += result.hits.numpy() * live
        score += scored * live
        elapsed += live
        live &= scored == 0.
        if not live.any():
            break
    if maximum_registration_error > 1.e-12:
        raise ValueError('known local translation did not match the outside raster')
    actions = np.asarray([sha256(np.ascontiguousarray(col).tobytes()).hexdigest() for col in tapes.T])
    return (np.c_[score, hits, elapsed].reshape(len(gains), len(roster), 3), actions.reshape(len(gains), len(roster)),
            np.concatenate(signals), maximum_registration_error)


def visual_fit(signals):
    predictor, required = signals.T
    use = np.abs(predictor) > 1.e-12
    denominator = predictor @ predictor
    rng = np.random.default_rng(SEED + 1)
    shuffled = rng.permutation(required)
    return dict(matched_gain=float(predictor @ required / denominator),
                shuffled_gain=float(predictor @ shuffled / denominator), samples=len(signals),
                nonzero_motion_frames=int(use.sum()))


def pilot():
    torch.set_num_threads(1)
    plan = register()
    weights = np.array(_read(WARM)['weights'])
    dynamic, dynamic_tape, signals, error = points(weights, plan['moving'], GAINS, move=True)
    static, static_tapes, _, _ = points(weights, plan['static'], GAINS, move=False)
    full_index = int(np.flatnonzero(GAINS == 1.)[0])
    outside_baseline = arcade([weights], plan['moving'], cap=CAP)[0]
    if not np.array_equal(outside_baseline, dynamic[full_index]) or not np.all(static_tapes == static_tapes[0]):
        raise ValueError('unity correction or static camera changed the original controller outcome')
    net = dynamic[..., 0].sum(1)
    restored = dynamic[full_index, :, 0]
    passed = [float(g) for i, g in enumerate(GAINS) if net[i] - net[full_index] >= 4
              and np.sum(dynamic[i, :, 0] > restored) >= 3]
    view = visual_fit(signals)
    report = dict(plan_sha256=_hash(PLAN), gate=bool(passed), promising_nonunity=passed,
                  source_visual_consistency=view, maximum_optical_registration_error=error,
                  moving=dict(outcomes=dynamic.tolist(), action_sha256=dynamic_tape.tolist(),
                              summary={str(k): summary(row) for k, row in zip(GAINS, dynamic, strict=True)},
                              paired_vs_full={str(k): paired(row, dynamic[full_index]) for k, row in zip(GAINS, dynamic, strict=True)}),
                  static=dict(outcomes=static.tolist(), action_sha256=static_tapes.tolist(),
                              summary=summary(static[0]), exact_action_parity=True))
    _write(PILOT, report)
    print('moving camera point nets', net.tolist(), 'static', report['static']['summary'],
          'optical consistency', view, 'practical gain gate', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    pilot()
