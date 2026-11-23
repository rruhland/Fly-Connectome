"""Transport a frozen learned motor law between two public optical link fits."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_donor_assess import CHECKPOINT, ShiftedBounce
from .tt24_donor_trial import Native, ROOT
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_public_morph import OUTPUT as EARLIER, load_morph
from .tt24_donor_morph import FIT as TARGET


RESULT = ROOT / 'dual-morph.json.gz'
HELD_RNG = 266413109
TICK_INCREMENT = 5.2 / 60.
RATE_CAP = .65
POSE_SCALE = np.array((.04, .04, .04, .14, .14))


def pose(decoder, joints):
    """Read the modeled blade pose from the frozen optical linkage alone."""
    link_position, orientation = decoder.links.read(np.atleast_2d(joints))
    blade_center = link_position + decoder.offset[:3] + orientation @ decoder.offset[3:]
    normal = orientation @ decoder.normal
    tilts = np.where(normal[:, 2, None] < 0., -normal, normal)[:, :2]
    return np.column_stack((blade_center, tilts))


class Transport:
    """Map rates locally and correct integration drift in the same measured units."""

    def __init__(self, source, target, home, *, converse=False):
        self.models = source, target
        self.home = np.asarray(home, dtype=float).copy()
        self.origin = np.asarray([pose(model, self.home)[0] for model in self.models])
        self.differences = np.eye(5) * .003
        self.target_issued = np.zeros(5)
        self.converse = converse
        self.errors = []
        self.limited = 0

    def _read(self, decoder, joint, accumulated):
        joint = np.clip(joint, self.home - .40, self.home + .40)
        current = self.home + accumulated
        states = np.vstack((current, joint + self.differences, joint - self.differences))
        values = pose(decoder, states) / POSE_SCALE
        jacobian = (values[1:6] - values[6:]).T / .006
        return values[0], jacobian

    @staticmethod
    def _damped_map(target, expected, anchor):
        penalty = np.trace(target.T @ target) / 5. * .07 + .02
        regularizer = np.eye(5) * penalty
        return np.linalg.solve(target.T @ target + regularizer, target.T @ expected + penalty * anchor)

    def command(self, request, source_issued, measured):
        source_pose, source_jac = self._read(self.models[0], self.home + source_issued, source_issued)
        target_pose, target_jac = self._read(self.models[1], measured, self.target_issued)
        source_effect = source_pose - self.origin[0] / POSE_SCALE
        actual_effect = target_pose - self.origin[1] / POSE_SCALE
        error = source_effect - actual_effect
        if self.converse:
            desired = target_jac @ request - .28 * error / TICK_INCREMENT
            raw = self._damped_map(source_jac, desired, request)
        else:
            desired = source_jac @ request + .28 * error / TICK_INCREMENT
            raw = self._damped_map(target_jac, desired, request)
        safe = request + np.clip(raw - request, -.35, .35)
        limit = 1.5 * np.linalg.norm(request) + .045
        norm = np.linalg.norm(safe)
        if norm > limit:
            safe *= limit / norm
        answer = np.clip(safe, -RATE_CAP, RATE_CAP)
        self.limited += int(np.any(abs(answer - raw) > 1.e-7))
        required = source_jac @ request
        self.errors.append([float(np.linalg.norm(target_jac @ request - required)),
                            float(np.linalg.norm(target_jac @ answer - required)),
                            float(np.linalg.norm(error))])
        self.target_issued += TICK_INCREMENT * answer
        return answer


def readout(arena, native, weights, start, mode):
    draw = mode != 'hold'
    obs = arena.reset(seed=start, split='held', regime='changed', render=draw)
    eye = load_public(FUSED_OUTPUT) if mode == 'untouched' else load_morph(TARGET) if draw else None
    transport = None
    if mode in ('both', 'converse'):
        original_link = load_morph(EARLIER).decoder
        transport = Transport(original_link, eye.decoder, obs['joints'], converse=mode == 'converse')
    source_issued = np.zeros(5)
    raw = np.zeros(5)
    actions = []
    done = False
    while not done:
        if draw:
            state = np.ascontiguousarray(eye.observe(obs), dtype=np.float64)
            previous = source_issued.copy()
            native.command(state.ctypes.data, weights.ctypes.data, source_issued.ctypes.data, raw.ctypes.data)
            command = raw if transport is None else transport.command(raw, previous, obs['joints'])
        else:
            command = raw
        obs, _, done = arena.step(command, render=draw)
        actions.append(command.tolist())
    checks = arena.judge_metrics()
    arena.reset(seed=start, split='held', regime='changed', render=False)
    for index, command in enumerate(actions):
        _, _, done = arena.step(command, render=False)
        if bool(done) != (index + 1 == len(actions)):
            raise ValueError(f'transported command tape changed terminal tick on {start}')
    if digest_json(checks) != digest_json(arena.judge_metrics()):
        raise ValueError(f'transported command tape changed physical judge result on {start}')
    diagnostic = None
    if transport is not None:
        diagnostic = dict(error=np.asarray(transport.errors).mean(axis=0).tolist(), bounds=transport.limited)
    return dict(seed=start, run=checks['longest_consecutive_rebounds'], score=checks['rebounds'],
                ticks=checks['steps'], gaps=eye.gaps if eye is not None else 0,
                modeled_alignment=diagnostic, physical=checks, actions=actions)


def summarize(rows):
    scores = np.asarray([row['run'] for row in rows])
    return dict(mean=float(scores.mean()), total=int(scores.sum()), max=int(scores.max()),
                at_least_ten=int(np.sum(scores >= 10)), counts=scores.tolist())


def fit_fields(name):
    from .tt4_transfer_trial import _read

    fit = _read(ROOT / name)
    return dict(metadata=name, image_frames=sum(row['ticks'] for row in fit['data']),
                probe_seeds=[row['seed'] for row in fit['data']], prior=fit['prior_geometry'],
                target_rewards_used=fit['target_rewards_used'], new_teacher_actions=fit['new_teacher_actions'])


def run(path):
    with np.load(CHECKPOINT, allow_pickle=False) as source:
        weights = np.ascontiguousarray(source['weights'], dtype=np.float64)
    starts = np.random.default_rng(HELD_RNG).integers(1, 2**31 - 1, 12).tolist()
    reference = fit_fields('public-morph-calibration.json.gz')
    target = fit_fields('donor-morph-fit.json.gz')
    previous = set(reference['probe_seeds']) | set(target['probe_seeds'])
    if previous.intersection(starts):
        raise ValueError('held release overlaps an optical probe')
    result = dict(schema='tt24-dual-morph-1', source_fit=reference, target_fit=target, start_seeds=starts,
                  motor_sha256=sha256(CHECKPOINT.read_bytes()).hexdigest(), extra_target_practice=0,
                  map='pose differences from own observed reset, damped local Jacobian transport',
                  held={}, summary={})
    arena = ShiftedBounce()
    arena.max_steps = 450
    native = Native()
    try:
        for mode in ('both', 'optics', 'untouched', 'converse', 'hold'):
            rows = []
            for seed in starts:
                row = readout(arena, native, weights, seed, mode)
                rows.append(row)
                print('dual morphology', mode, seed, row['run'], row['modeled_alignment'], flush=True)
            result['held'][mode] = rows
            result['summary'][mode] = summarize(rows)
            print('dual morphology summary', mode, result['summary'][mode], flush=True)
            with gzip.open(path, 'wt') as stream:
                json.dump(result, stream, separators=(',', ':'))
    finally:
        native.working.cleanup()
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULT)
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
