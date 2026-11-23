"""Test source-fitted normal outcome dynamics in an otherwise matched reactive bounce controller."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BALL_RADIUS, BounceArena, mujoco
from .tt24_bounce_trial import verify_replay
from .tt24_motor import MotorPolicy
from .tt24_recurrence_arm import EMBODIMENT, target_effects
from .tt24_spatial_carry import OUTPUT, RESULT as CONTACTS


RECEIPT = OUTPUT / 'spatial-carry-clean-control.json.gz'
SCHEDULE = 808100392
MODES = ('coherent', 'shuffled', 'constant_source', 'cold', 'warm', 'passive')
STARTS = 12
OUTGOING_SPEED = 1.55
PADDLE_RADIUS = .076
FACE_DISTANCE = BALL_RADIUS + .006


def normal_response(closure, slip, radial, law, mode):
    if mode == 'cold':
        return 1.
    if mode == 'constant_source':
        return law['constant']
    features = np.array((np.log(closure), slip / closure, radial / PADDLE_RADIUS))
    x = np.r_[1., np.clip((features - law['median']) / law['spread'], -3., 3.)]
    return float(np.clip(x @ law['weights'][mode], .55, 1.10))


class CleanFrame:
    """Diagnostic only. Measure current body/ball geometry and clean actuator leverage."""

    def __init__(self, arena):
        self.arena = arena
        self.jac = np.zeros((3, arena._model.nv))
        self.angular = np.zeros_like(self.jac)

    def project(self, _obs):
        arena, m, d = self.arena, self.arena._model, self.arena._data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        blade = d.geom_xpos[arena._blade].copy()
        normal = d.geom_xmat[arena._blade].reshape(3, 3)[:, 2].copy()
        if normal[2] < 0.:
            normal *= -1.
        twist = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, arena._blade, twist, 0)
        mujoco.mj_jac(m, d, self.jac, self.angular, blade, arena._grip)
        controlled = arena._joint_velocity[:arena.action_size]
        jac = normal @ self.jac[:, controlled]
        return (arena._ball_position - blade).copy(), arena._ball_velocity.copy(), normal, twist[3:], jac


def residual(state, law, mode):
    if state is None:
        return None, None
    position, velocity, normal, paddle_velocity, jacobian = state
    height = position @ normal - FACE_DISTANCE
    relative = velocity - paddle_velocity
    closure = -relative @ normal
    if not (.015 < height < .14 and .55 < closure < 4.4):
        return None, None
    tangent = relative + closure * normal
    time = height / closure
    projected_contact = position - (position @ normal) * normal + tangent * time
    radial = np.linalg.norm(projected_contact)
    if radial > .084:
        return None, None
    response = normal_response(closure, np.linalg.norm(tangent), radial, law, mode)
    predicted_outgoing = response * closure + paddle_velocity @ normal
    desired_paddle_change = (OUTGOING_SPEED - predicted_outgoing) / (1. + response)
    # Convert one desired normal-speed change to an instantaneous rate residual.
    # The active wrist anchor supplies all other stabilization and lateral action.
    leverage = jacobian * 5.2
    correction = .53 * desired_paddle_change * leverage / (leverage @ leverage + .16)
    correction *= min(1., .16 / max(1.e-9, np.linalg.norm(correction)))
    return correction, float(response)


def drop(arena, seed, body, mode, law):
    held = mode == 'passive'
    use_geometry = mode in MODES[:4]
    obs = arena.reset(seed=seed, split='held', render=not held)
    pupil = MotorPolicy(np.asarray(body['weights']), target_effects(body, arena.action_size),
                        readout=body['readout'], retinal=body['retinal'])
    frame = CleanFrame(arena) if use_geometry else None
    actions, influences, values = [], [], []
    done, score = False, 0.
    while not done:
        tick = len(actions)
        sampled = tick % body['stride'] == 0
        if held:
            command = np.zeros(arena.action_size)
        else:
            projected = frame.project(obs) if frame is not None else None
            command = pupil.command(obs, sampled=sampled)
            if use_geometry:
                correction, response = residual(projected, law, mode)
                if correction is not None:
                    before = command
                    command = np.clip(before + correction, -1., 1.)
                    influences.append(float(np.linalg.norm(command - before)))
                    values.append(response)
        actions.append(command.tolist())
        obs, reward, done = arena.step(command, render=not held and (tick + 1) % body['stride'] == 0)
        score += reward
    return arena.judge_metrics() | dict(actions=actions, score=score, changed_frames=len(influences),
                                        mean_abs_residual=float(np.mean(influences)) if influences else 0.,
                                        response_median=float(np.median(values)) if values else None)


def statistic(rows):
    lengths = {mode: np.array([r['longest_consecutive_rebounds'] for r in episodes]) for mode, episodes in rows.items()}
    rng = np.random.default_rng(1810099477)
    resample = rng.integers(0, STARTS, size=(4096, STARTS))
    differences = {}
    for mode in MODES[1:]:
        delta = lengths['coherent'] - lengths[mode]
        differences[mode] = dict(mean=float(delta.mean()), higher=int((delta > 0).sum()), lower=int((delta < 0).sum()),
                                 bootstrap95=np.quantile(delta[resample].mean(axis=1), (.025, .975)).tolist())
    modes = {mode: dict(consecutive=x.tolist(), mean=float(x.mean()), best=int(x.max()),
                        changed_frames=sum(r['changed_frames'] for r in rows[mode])) for mode, x in lengths.items()}
    return dict(modes=modes, coherent_minus=differences,
                qualified=all(differences[mode]['mean'] > 0. for mode in ('cold', 'shuffled', 'warm')) and
                min(differences[mode]['bootstrap95'][0] for mode in ('cold', 'shuffled')) > 0.)


def run(output):
    began = perf_counter()
    data = _read(CONTACTS)
    body = _read(EMBODIMENT)
    starts = np.random.default_rng(SCHEDULE).choice(2147483646, STARTS, replace=False).tolist()
    known = {r['seed'] for rows in data['target'].values() for r in rows}
    known |= set(body['practice_seeds']) | {e['seed'] for rows in body['held'].values() for e in rows}
    if known.intersection(starts):
        raise ValueError('assessment overlaps earlier exposure')
    metadata = dict(schema='tt24-spatial-carry-reactive-exit-1', contact_fit_sha256=_hash(CONTACTS),
                    wrist_body_sha256=_hash(EMBODIMENT), learned_body_attempts=body['total_initial_attempts'],
                    additional_practice_attempts=0, sensor='clean', held=starts, outgoing_speed=OUTGOING_SPEED)
    report = _read(output) if output.exists() else dict(metadata=metadata, held={mode: [] for mode in MODES})
    if report['metadata'] != metadata:
        raise ValueError('saved assessment has another source or schedule')
    arena = BounceArena(view='wrist', width=64, height=48)
    try:
        for i, launch in enumerate(starts):
            for mode in MODES if i % 2 == 0 else MODES[::-1]:
                if len(report['held'][mode]) > i:
                    continue
                row = drop(arena, launch, body, mode, data['normal_model'])
                report['held'][mode].append(row)
                _write(output, report)
                print('clean', launch, mode, row['longest_consecutive_rebounds'], row['changed_frames'], flush=True)
        for rows in report['held'].values():
            for row in rows:
                verify_replay(arena, row)
    finally:
        arena.close()
    report.update(summary=statistic(report['held']), verified_tapes=STARTS * len(MODES), seconds=perf_counter() - began)
    _write(output, report)
    print(report['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
